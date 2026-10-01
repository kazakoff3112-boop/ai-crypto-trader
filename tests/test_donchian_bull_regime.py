"""Synthetic-only regression tests for the pre-registered BULL-entry hypothesis."""

from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from io import StringIO
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from ai_crypto_trader.backtesting import BacktestConfig, BacktestEngine, PortfolioSnapshot, Signal
from ai_crypto_trader.backtesting.analytics import analyze_backtest
from ai_crypto_trader.bybit.client import BybitReadOnlyClient
from ai_crypto_trader.database.historical_cache import HistoricalCache
from ai_crypto_trader.diagnostics.backtest import main
from ai_crypto_trader.market_data.candle import Candle
from ai_crypto_trader.market_data.market_regime import MarketRegime, MarketRegimeReport, RegimePoint
from ai_crypto_trader.strategies.donchian_breakout import DonchianBreakoutStrategy
from ai_crypto_trader.strategies.donchian_bull_regime import DonchianBullRegimeStrategy

D = Decimal
START = datetime(2024, 1, 10, 10, tzinfo=timezone.utc)
STEP = timedelta(minutes=15)


def candles():
    rows = [Candle("BTCUSDT", "15m", START + index * STEP,
                   D("10"), D("10"), D("9"), D("10"), D("1"), D("1"))
            for index in range(100)]
    rows[96] = replace(rows[96], high=D("11"), close=D("11"))
    rows[97] = replace(rows[97], open=D("20"), high=D("20"), low=D("19"), close=D("20"))
    rows[98] = replace(rows[98], low=D("8"), close=D("8"))
    rows[99] = replace(rows[99], open=D("7"), high=D("7"), low=D("6"), close=D("7"))
    return tuple(rows)


def view(quantity="0"):
    return PortfolioSnapshot("BTCUSDT", D("1000"), D(quantity), D("0"), D("0"), D("1000"))


class BullRegimeTests(unittest.TestCase):
    def setUp(self):
        self.history = candles()

    def run_engine(self, strategy, *, data=None, config=None):
        return BacktestEngine().run(
            self.history if data is None else data, strategy, symbol="BTCUSDT", interval="15m",
            config=config or BacktestConfig(D("0.001"), D("5")),
            as_of_utc=START + timedelta(days=2),
        )

    def test_buy_only_on_bull_and_filter_counts(self):
        for regime, expected in ((MarketRegime.BULL, "BUY"), (MarketRegime.BEAR, "HOLD"),
                                 (MarketRegime.SIDEWAYS, "HOLD"), (None, "HOLD")):
            with self.subTest(regime=regime):
                seen = []
                strategy = DonchianBullRegimeStrategy(lambda stamp: (seen.append(stamp), regime)[1])
                signal = strategy.on_candle(self.history[:97], view())
                self.assertEqual(signal.action, expected)
                self.assertEqual(seen, [self.history[96].close_time_utc])
                stats = strategy.filter_stats
                self.assertEqual(stats.raw_buy_signals, 1)
                self.assertEqual((stats.allowed_bull, stats.blocked_bear,
                                  stats.blocked_sideways, stats.unavailable),
                                 (int(regime is MarketRegime.BULL), int(regime is MarketRegime.BEAR),
                                  int(regime is MarketRegime.SIDEWAYS), int(regime is None)))
        self.assertEqual(dict(DonchianBullRegimeStrategy(lambda _: MarketRegime.BULL).parameters),
                         {"entry_lookback": "96", "exit_lookback": "48",
                          "position_size_pct": "25", "entry_regime": "BULL"})

    def test_no_breakout_never_queries_regime(self):
        rows = (*self.history[:96], replace(self.history[96], high=D("10"), close=D("10")))
        strategy = DonchianBullRegimeStrategy(lambda _: self.fail("unexpected regime lookup"))
        self.assertEqual(strategy.on_candle(rows, view()), Signal())
        self.assertEqual(strategy.filter_stats.raw_buy_signals, 0)

    def test_sell_is_original_donchian_and_regime_independent(self):
        for regime in (MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.SIDEWAYS, None):
            with self.subTest(regime=regime):
                strategy = DonchianBullRegimeStrategy(lambda _: self.fail("SELL must not query regime"))
                self.assertEqual(strategy.on_candle(self.history[:99], view("1")),
                                 Signal("SELL", quantity=D("1")))
                self.assertEqual(strategy.filter_stats.raw_buy_signals, 0)
                # A regime change alone does not create an exit.
                self.assertEqual(strategy.on_candle(self.history[:98], view("1")), Signal())
                equal = replace(self.history[98], low=D("9"), close=D("9"))
                self.assertEqual(strategy.on_candle((*self.history[:98], equal), view("1")), Signal())

    def test_last_closed_hour_at_1045_not_current_hour_or_future(self):
        signal_at = self.history[96].close_time_utc
        self.assertEqual(signal_at.minute, 15)  # Use an explicit 10:45 signal below.
        older = datetime(2024, 1, 11, 10, tzinfo=timezone.utc)
        future = older + timedelta(hours=1)
        points = (
            RegimePoint(older, "BTCUSDT", "1h", D("10"), D("10"), D("10"), D("9"), MarketRegime.BULL),
            RegimePoint(future, "BTCUSDT", "1h", D("10"), D("10"), D("10"), D("9"), MarketRegime.BEAR),
        )
        report = MarketRegimeReport(older - timedelta(hours=1), future, points, (), (), 0)
        # Shift only the synthetic clock so the breakout candle closes at 10:45.
        shift = timedelta(minutes=30)
        history = tuple(replace(row, open_time_utc=row.open_time_utc + shift) for row in self.history)
        self.assertEqual(history[96].close_time_utc, older + timedelta(minutes=45))
        first = DonchianBullRegimeStrategy(report.regime_at)
        self.assertEqual(first.on_candle(history[:97], view()).action, "BUY")
        changed_future = replace(report, points=(points[0], replace(points[1], regime=MarketRegime.SIDEWAYS)))
        second = DonchianBullRegimeStrategy(changed_future.regime_at)
        changed_15m = (*history[:97], replace(history[97], high=D("999"), close=D("999")))
        self.assertEqual(second.on_candle(changed_15m[:97], view()),
                         Signal("BUY", budget_usdt=D("250")))
        self.assertEqual(report.regime_at(older + timedelta(minutes=45)), MarketRegime.BULL)
        self.assertEqual(report.regime_at(future), MarketRegime.BEAR)

    def test_engine_next_open_sizing_costs_and_blocked_ledger(self):
        bullish = DonchianBullRegimeStrategy(lambda _: MarketRegime.BULL)
        result = self.run_engine(bullish)
        original = self.run_engine(DonchianBreakoutStrategy())
        self.assertEqual(result.ledger, original.ledger)
        self.assertEqual(result.final_equity, original.final_equity)
        self.assertEqual([trade.side for trade in result.ledger], ["BUY", "SELL"])
        buy, sell = result.ledger
        self.assertEqual(buy.execution_candle_timestamp_utc, self.history[97].open_time_utc)
        self.assertEqual(sell.execution_candle_timestamp_utc, self.history[99].open_time_utc)
        self.assertLessEqual(buy.gross_value + buy.commission, D("250"))
        self.assertEqual(buy.execution_price, D("20") * D("1.0005"))
        self.assertEqual(sell.execution_price, D("7") * D("0.9995"))
        self.assertEqual(result.total_commissions, sum((trade.commission for trade in result.ledger), D("0")))
        self.assertEqual(result.total_slippage, sum((trade.slippage_cost for trade in result.ledger), D("0")))
        self.assertEqual(analyze_backtest(result).completed_trade_count, 1)
        blocked = DonchianBullRegimeStrategy(lambda _: MarketRegime.BEAR)
        no_trade = self.run_engine(blocked)
        self.assertEqual((no_trade.buy_operations, no_trade.sell_operations, no_trade.ledger_entries), (0, 0, 0))

    def test_engine_future_15m_candle_cannot_change_earlier_signal(self):
        class RecordingStrategy(DonchianBullRegimeStrategy):
            def __init__(self):
                super().__init__(lambda _: MarketRegime.BULL)
                self.signal_at_breakout = None

            def on_candle(self, history, portfolio):
                signal = super().on_candle(history, portfolio)
                if history[-1].open_time_utc == START + 96 * STEP:
                    self.signal_at_breakout = signal
                return signal

        changed = list(self.history)
        changed[97] = replace(changed[97], high=D("999"), close=D("999"))
        original_strategy, changed_strategy = RecordingStrategy(), RecordingStrategy()
        self.run_engine(original_strategy)
        self.run_engine(changed_strategy, data=changed)
        self.assertEqual(original_strategy.signal_at_breakout,
                         Signal("BUY", budget_usdt=D("250")))
        self.assertEqual(changed_strategy.signal_at_breakout,
                         original_strategy.signal_at_breakout)

    def test_cli_readonly_dual_warmup_and_fixed_parameters(self):
        rows = self.history
        evaluation_start = rows[96].open_time_utc
        evaluation_end = rows[-1].close_time_utc
        hourly_start = evaluation_start.replace(minute=0) - timedelta(hours=1)
        first_hour = hourly_start - timedelta(hours=223)
        hours = int((evaluation_end.replace(minute=0) - first_hour) / timedelta(hours=1))
        hourly = [Candle("BTCUSDT", "1h", first_hour + index * timedelta(hours=1),
                         D(100 + index), D(100 + index), D(100 + index), D(100 + index), D("1"), D("1"))
                  for index in range(hours)]
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "history.sqlite3"
            paper_path = Path(directory) / "paper.sqlite3"
            cache = HistoricalCache(cache_path)
            cache.save(list(rows))
            cache.save(hourly)
            paper_path.write_bytes(b"paper-ledger-sentinel")
            cache_before, paper_before = cache_path.read_bytes(), paper_path.read_bytes()
            flags = ["--symbol", "BTCUSDT", "--interval", "15m", "--strategy", "donchian-bull-regime",
                     "--entry-lookback", "96", "--exit-lookback", "48", "--position-size-pct", "25",
                     "--start", evaluation_start.isoformat(), "--end", evaluation_end.isoformat()]
            with patch.dict("os.environ", {"HISTORICAL_CACHE_PATH": str(cache_path),
                                        "DATABASE_PATH": str(paper_path)}):
                with patch.object(socket.socket, "connect", side_effect=AssertionError("network called")):
                    output = StringIO()
                    with redirect_stdout(output):
                        self.assertEqual(main(flags), 0)
                    for phrase in ("REGIME FILTER:", "Entry allowed only in BULL",
                                   "Exit: original Donchian rule, regime-independent",
                                   "Raw Donchian BUY signals: 1", "Allowed BULL BUY signals: 1",
                                   "Blocked BEAR: 0", "Blocked SIDEWAYS: 0", "Unavailable regime: 0",
                                   "TRADE QUALITY", "BENCHMARKS", "MATCHED BUY & HOLD", "FULL BUY & HOLD"):
                        self.assertIn(phrase, output.getvalue())
                    self.assertEqual(cache_path.read_bytes(), cache_before)
                    self.assertEqual(paper_path.read_bytes(), paper_before)
                    for extra in (["--entry-lookback", "95"], ["--exit-lookback", "47"],
                                  ["--position-size-pct", "30"]):
                        altered = flags[:]
                        altered[altered.index(extra[0]) + 1] = extra[1]
                        with self.assertRaises(SystemExit), redirect_stdout(StringIO()), patch("sys.stderr", StringIO()):
                            main(altered)
                    # Missing 1h warm-up must fail without touching the PAPER ledger.
                    with cache._connect() as db:
                        db.execute("DELETE FROM candles WHERE interval='1h' AND open_time_ms=(SELECT MIN(open_time_ms) FROM candles WHERE interval='1h')")
                    with redirect_stdout(StringIO()):
                        self.assertEqual(main(flags), 1)
            self.assertEqual(paper_path.read_bytes(), paper_before)

    def test_readonly_client_has_no_order_methods(self):
        for name in ("create_order", "place_order", "amend_order", "cancel_order"):
            self.assertFalse(hasattr(BybitReadOnlyClient, name))


if __name__ == "__main__":
    unittest.main()
