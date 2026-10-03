"""Independent append-only SQLite history of non-trading shadow decisions."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
import sqlite3
from uuid import uuid4

from ..market_data.candle import Candle, HistoricalDataUnavailable, validate_candle
from ..market_data.market_regime import MarketRegime

STRATEGY_NAME = "1h BULL Regime Hold"
STRATEGY_VERSION = "regime-bull-hold-v1"
SYMBOL = "BTCUSDT"
TIMEFRAME = "1h"
HOUR = timedelta(hours=1)
ACTIONS = frozenset({"BUY_SIGNAL", "SELL_SIGNAL", "HOLD", "NOT_READY"})
STATES = frozenset({"FLAT", "LONG"})


class ShadowUnavailable(RuntimeError):
    """Shadow history or current data cannot safely produce a decision."""


class ShadowGapUnavailable(ShadowUnavailable):
    """A closed hour was skipped; controlled catch-up is required."""


@dataclass(frozen=True)
class ShadowDecision:
    decision_id: str
    strategy_name: str
    strategy_version: str
    symbol: str
    timeframe: str
    candle_close_utc: datetime
    decision_utc: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    current_regime: MarketRegime
    previous_regime: MarketRegime
    action: str
    reason: str
    sma50: Decimal
    sma200: Decimal
    previous_sma200: Decimal
    state_before: str
    state_after: str
    position_would_be_open: bool
    software_version: str = "0.1.0"


@dataclass(frozen=True)
class ShadowStatus:
    state: str
    last_candle_utc: datetime | None
    last_regime: MarketRegime | None
    last_action: str | None
    processed: int
    buy_signals: int
    sell_signals: int


_COLUMNS = ("decision_id", "strategy_name", "strategy_version", "symbol", "timeframe",
            "candle_close_utc", "decision_utc", "open", "high", "low", "close",
            "current_regime", "previous_regime", "action", "reason", "sma50", "sma200",
            "previous_sma200", "state_before", "state_after", "position_would_be_open",
            "software_version")


def _values(decision: ShadowDecision) -> tuple:
    values = (
        decision.decision_id, decision.strategy_name, decision.strategy_version,
        decision.symbol, decision.timeframe, decision.candle_close_utc.isoformat(),
        decision.decision_utc.isoformat(),
        *(str(value) for value in (decision.open, decision.high, decision.low, decision.close)),
        decision.current_regime.value, decision.previous_regime.value,
        decision.action, decision.reason,
        *(str(value) for value in (decision.sma50, decision.sma200,
                                   decision.previous_sma200)),
        decision.state_before, decision.state_after,
        int(decision.position_would_be_open), decision.software_version,
    )
    _read(values)  # Validate the exact on-disk representation before INSERT.
    return values


def _utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        raise ShadowUnavailable("Invalid stored UTC timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ShadowUnavailable("Stored timestamp is not UTC")
    return parsed


def _decimal(value: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except (InvalidOperation, TypeError):
        raise ShadowUnavailable("Invalid stored Decimal") from None
    if not parsed.is_finite():
        raise ShadowUnavailable("Non-finite stored Decimal")
    return parsed


def _read(row: tuple) -> ShadowDecision:
    try:
        data = dict(zip(_COLUMNS, row))
        decision = ShadowDecision(
            data["decision_id"], data["strategy_name"], data["strategy_version"],
            data["symbol"], data["timeframe"], _utc(data["candle_close_utc"]),
            _utc(data["decision_utc"]), *(_decimal(data[name]) for name in
                                        ("open", "high", "low", "close")),
            MarketRegime(data["current_regime"]), MarketRegime(data["previous_regime"]),
            data["action"], data["reason"],
            *(_decimal(data[name]) for name in ("sma50", "sma200", "previous_sma200")),
            data["state_before"], data["state_after"],
            data["position_would_be_open"] == 1, data["software_version"],
        )
        validate_candle(Candle(SYMBOL, TIMEFRAME, decision.candle_close_utc - HOUR,
                               decision.open, decision.high, decision.low, decision.close,
                               Decimal("0"), Decimal("0")))
    except (KeyError, ValueError, TypeError, HistoricalDataUnavailable):
        raise ShadowUnavailable("Invalid stored shadow decision") from None
    if (not decision.decision_id or decision.strategy_name != STRATEGY_NAME
            or decision.strategy_version != STRATEGY_VERSION or decision.symbol != SYMBOL
            or decision.timeframe != TIMEFRAME or decision.action not in ACTIONS
            or decision.state_before not in STATES or decision.state_after not in STATES
            or type(data["position_would_be_open"]) is not int
            or data["position_would_be_open"] not in (0, 1)
            or decision.position_would_be_open != (decision.state_after == "LONG")
            or any(value <= 0 for value in (decision.sma50, decision.sma200,
                                            decision.previous_sma200))
            or decision.decision_utc < decision.candle_close_utc):
        raise ShadowUnavailable("Stored shadow decision violates invariants")
    return decision


def _validate_chain(decisions: list[ShadowDecision]) -> ShadowStatus:
    state = "FLAT"
    last: ShadowDecision | None = None
    buys = sells = 0
    for decision in decisions:
        if decision.state_before != state:
            raise ShadowUnavailable("Shadow state chain is inconsistent")
        if last is not None and decision.candle_close_utc != last.candle_close_utc + HOUR:
            raise ShadowUnavailable("Shadow decision history has a gap or duplicate")
        if last is not None and decision.previous_regime is not last.current_regime:
            raise ShadowUnavailable("Shadow regime chain is inconsistent")
        expected = "HOLD"
        if state == "FLAT" and decision.previous_regime is not MarketRegime.BULL \
                and decision.current_regime is MarketRegime.BULL:
            expected = "BUY_SIGNAL"
        elif state == "LONG" and decision.previous_regime is MarketRegime.BULL \
                and decision.current_regime is not MarketRegime.BULL:
            expected = "SELL_SIGNAL"
        if decision.action != expected:
            raise ShadowUnavailable("Stored shadow action contradicts frozen strategy rules")
        if decision.action == "BUY_SIGNAL":
            if (state != "FLAT" or decision.previous_regime is MarketRegime.BULL
                    or decision.current_regime is not MarketRegime.BULL
                    or decision.state_after != "LONG"):
                raise ShadowUnavailable("Impossible stored BUY signal")
            buys += 1
        elif decision.action == "SELL_SIGNAL":
            if (state != "LONG" or decision.previous_regime is not MarketRegime.BULL
                    or decision.current_regime is MarketRegime.BULL
                    or decision.state_after != "FLAT"):
                raise ShadowUnavailable("Impossible stored SELL signal")
            sells += 1
        elif decision.state_after != state:
            raise ShadowUnavailable("Non-trading decision changed shadow state")
        state = decision.state_after
        last = decision
    return ShadowStatus(state, last.candle_close_utc if last else None,
                        last.current_regime if last else None, last.action if last else None,
                        len(decisions), buys, sells)


class ShadowStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ShadowUnavailable("Unsupported shadow database schema")
            with db:
                db.execute("""CREATE TABLE IF NOT EXISTS shadow_decisions (
                    decision_id TEXT PRIMARY KEY, strategy_name TEXT NOT NULL,
                    strategy_version TEXT NOT NULL, symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
                    candle_close_utc TEXT NOT NULL, decision_utc TEXT NOT NULL,
                    open TEXT NOT NULL, high TEXT NOT NULL, low TEXT NOT NULL, close TEXT NOT NULL,
                    current_regime TEXT NOT NULL, previous_regime TEXT NOT NULL,
                    action TEXT NOT NULL, reason TEXT NOT NULL,
                    sma50 TEXT NOT NULL, sma200 TEXT NOT NULL, previous_sma200 TEXT NOT NULL,
                    state_before TEXT NOT NULL, state_after TEXT NOT NULL,
                    position_would_be_open INTEGER NOT NULL, software_version TEXT NOT NULL,
                    UNIQUE(strategy_version, symbol, timeframe, candle_close_utc)
                )""")
                db.execute("""CREATE TRIGGER IF NOT EXISTS shadow_no_update
                    BEFORE UPDATE ON shadow_decisions BEGIN
                    SELECT RAISE(ABORT, 'shadow decisions are append-only'); END""")
                db.execute("""CREATE TRIGGER IF NOT EXISTS shadow_no_delete
                    BEFORE DELETE ON shadow_decisions BEGIN
                    SELECT RAISE(ABORT, 'shadow decisions are append-only'); END""")
                db.execute("PRAGMA user_version = 1")
        self.status()  # Refuse to continue from damaged or incompatible history.

    def _history(self, db: sqlite3.Connection) -> tuple[list[ShadowDecision], ShadowStatus]:
        rows = db.execute(f"SELECT {', '.join(_COLUMNS)} FROM shadow_decisions "
                          "ORDER BY candle_close_utc").fetchall()
        decisions = [_read(row) for row in rows]
        return decisions, _validate_chain(decisions)

    def status(self) -> ShadowStatus:
        with closing(sqlite3.connect(self.path)) as db:
            return self._history(db)[1]

    def append(self, decision: ShadowDecision) -> tuple[bool, ShadowStatus]:
        with closing(sqlite3.connect(self.path, timeout=10)) as db:
            try:
                db.execute("BEGIN IMMEDIATE")
                history, status = self._history(db)
                if status.last_candle_utc == decision.candle_close_utc:
                    db.rollback()
                    return False, status
                if (status.last_candle_utc is not None and
                        decision.candle_close_utc != status.last_candle_utc + HOUR):
                    raise ShadowGapUnavailable("Gap before new shadow candle; controlled catch-up required")
                if decision.state_before != status.state:
                    raise ShadowUnavailable("Shadow state changed before decision was stored")
                _validate_chain([*history, decision])
                values = _values(decision)
                db.execute(f"INSERT INTO shadow_decisions ({', '.join(_COLUMNS)}) "
                           f"VALUES ({', '.join('?' for _ in _COLUMNS)})", values)
                db.commit()
                return True, _validate_chain([*history, decision])
            except Exception:
                db.rollback()
                raise

    def append_batch(self, decisions: tuple[ShadowDecision, ...], *,
                     expected_status: ShadowStatus) -> ShadowStatus:
        """Atomically append one prevalidated, chronological catch-up batch."""
        if not decisions:
            raise ShadowUnavailable("Catch-up batch is empty")
        with closing(sqlite3.connect(self.path, timeout=10)) as db:
            try:
                db.execute("BEGIN IMMEDIATE")
                history, status = self._history(db)
                if status != expected_status:
                    raise ShadowUnavailable("Shadow state changed after catch-up preview")
                candidate = [*history, *decisions]
                updated = _validate_chain(candidate)
                values = [_values(decision) for decision in decisions]
                db.executemany(
                    f"INSERT INTO shadow_decisions ({', '.join(_COLUMNS)}) "
                    f"VALUES ({', '.join('?' for _ in _COLUMNS)})", values,
                )
                db.commit()
                return updated
            except Exception:
                db.rollback()
                raise


def new_decision(*, candle: Candle, decision_utc: datetime, previous_regime: MarketRegime,
                 current_regime: MarketRegime, action: str, reason: str,
                 sma50: Decimal, sma200: Decimal, previous_sma200: Decimal,
                 state_before: str, state_after: str) -> ShadowDecision:
    return ShadowDecision(str(uuid4()), STRATEGY_NAME, STRATEGY_VERSION, SYMBOL, TIMEFRAME,
                          candle.close_time_utc, decision_utc, candle.open, candle.high,
                          candle.low, candle.close, current_regime, previous_regime, action,
                          reason, sma50, sma200, previous_sma200, state_before, state_after,
                          state_after == "LONG")
