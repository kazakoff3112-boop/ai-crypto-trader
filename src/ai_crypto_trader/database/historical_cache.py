"""Independent SQLite cache for public candles; never opens the trade ledger DB."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
import sqlite3

from ..market_data.historical import (
    Candle, HistoricalDataUnavailable, INTERVAL_MINUTES, epoch_ms, utc_from_ms,
    validate_candle, validate_range,
)


class HistoricalCache:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, 2):
                raise HistoricalDataUnavailable("Unsupported historical cache schema")
            db.execute("""CREATE TABLE IF NOT EXISTS candles (
                symbol TEXT NOT NULL, interval TEXT NOT NULL, open_time_ms INTEGER NOT NULL,
                open TEXT NOT NULL, high TEXT NOT NULL, low TEXT NOT NULL, close TEXT NOT NULL,
                volume TEXT NOT NULL, turnover TEXT NOT NULL,
                source TEXT NOT NULL, fetched_at_utc TEXT NOT NULL,
                PRIMARY KEY(symbol, interval, open_time_ms))""")
            db.execute("""CREATE TABLE IF NOT EXISTS fetch_batches (
                symbol TEXT NOT NULL, interval TEXT NOT NULL,
                first_open_ms INTEGER NOT NULL, last_open_ms INTEGER NOT NULL,
                candle_count INTEGER NOT NULL, source TEXT NOT NULL,
                fetched_at_utc TEXT NOT NULL,
                PRIMARY KEY(symbol, interval, first_open_ms, last_open_ms, candle_count))""")
            db.execute("PRAGMA user_version = 2")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def save(self, candles: list[Candle], *, fetched_at: datetime | None = None) -> None:
        if not candles:
            return
        stamp = (fetched_at or datetime.now(timezone.utc)).isoformat()
        if datetime.fromisoformat(stamp).utcoffset().total_seconds() != 0:
            raise ValueError("Fetch timestamp must be UTC")
        first = candles[0]
        step = None
        rows = []
        for candle in candles:
            validate_candle(candle)
            if candle.symbol != first.symbol or candle.interval != first.interval:
                raise HistoricalDataUnavailable("Mixed candle batch")
            ms = epoch_ms(candle.open_time_utc)
            if step is not None and ms != step:
                raise HistoricalDataUnavailable("Non-contiguous candle batch")
            step = ms + INTERVAL_MINUTES[candle.interval] * 60_000
            rows.append((candle.symbol, candle.interval, ms, *(str(x) for x in (
                candle.open, candle.high, candle.low, candle.close, candle.volume, candle.turnover
            )), "Bybit", stamp))
        with self._connect() as db:
            for row in rows:
                prior = db.execute("""SELECT open,high,low,close,volume,turnover,source
                    FROM candles WHERE symbol=? AND interval=? AND open_time_ms=?""", row[:3]).fetchone()
                if prior is not None:
                    try:
                        same = (tuple(Decimal(v) for v in prior[:6]) == tuple(Decimal(v) for v in row[3:9])
                                and prior[6] == "Bybit")
                    except (InvalidOperation, TypeError):
                        same = False
                    if not same:
                        raise HistoricalDataUnavailable("Cached candle conflicts with downloaded data")
                else:
                    db.execute("""INSERT INTO candles VALUES (?,?,?,?,?,?,?,?,?,?,?)""", row)
            db.execute("""INSERT INTO fetch_batches VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(symbol,interval,first_open_ms,last_open_ms,candle_count)
                DO UPDATE SET fetched_at_utc=excluded.fetched_at_utc""", (
                first.symbol, first.interval, rows[0][2], rows[-1][2], len(rows),
                "Bybit", stamp,
            ))

    def load(self, symbol: str, interval: str, start: datetime, end: datetime,
             *, category: str = "spot") -> list[Candle]:
        symbol, step, start_ms, end_ms = validate_range(symbol, category, interval, start, end)
        with self._connect() as db:
            rows = db.execute("""SELECT open_time_ms,open,high,low,close,volume,turnover,source
                FROM candles WHERE symbol=? AND interval=? AND open_time_ms>=? AND open_time_ms<?
                ORDER BY open_time_ms""", (symbol, interval, start_ms, end_ms)).fetchall()
        if len(rows) != (end_ms - start_ms) // step:
            raise HistoricalDataUnavailable("Cached range is incomplete")
        result = []
        for index, row in enumerate(rows):
            ms, *values, source = row
            if type(ms) is not int or ms != start_ms + index * step or source != "Bybit":
                raise HistoricalDataUnavailable("Cached candle sequence or source is invalid")
            try:
                numbers = [Decimal(v) if isinstance(v, str) else None for v in values]
            except InvalidOperation:
                raise HistoricalDataUnavailable("Cached candle has an invalid number") from None
            candle = Candle(symbol, interval, utc_from_ms(ms), *numbers)
            validate_candle(candle)
            result.append(candle)
        return result

    def summary(self) -> list[tuple[str, str, int, int, int, str]]:
        with self._connect() as db:
            return db.execute("""SELECT symbol,interval,COUNT(*),MIN(open_time_ms),
                MAX(open_time_ms),MAX(fetched_at_utc) FROM candles
                GROUP BY symbol,interval ORDER BY symbol,interval""").fetchall()

    def batches(self) -> list[tuple[str, str, int, int, int, str, str]]:
        with self._connect() as db:
            return db.execute("""SELECT symbol,interval,first_open_ms,last_open_ms,
                candle_count,source,fetched_at_utc FROM fetch_batches
                ORDER BY fetched_at_utc,symbol,interval""").fetchall()
