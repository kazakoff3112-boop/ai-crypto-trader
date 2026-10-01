"""Synthetic, offline checks of the pre-registered 1h BULL hold strategy."""

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

from ai_crypto_trader.backtesting import BacktestConfig, BacktestEngine, BacktestError, PortfolioSnapshot, Signal
from ai_crypto_trader.backtesting.analytics import analyze_backtest
from ai_crypto_trader.backtesting.benchmark import run_buy_and_hold, run_cash_benchmark
from ai_crypto_trader.backtesting.periods import BacktestPeriod
from ai_crypto_trader.bybit.client import BybitReadOnlyClient
from ai_crypto_trader.database.historical_cache import HistoricalCache
from ai_crypto_trader.diagnostics.backtest import main
from ai_crypto_trader.market_data.candle import Candle
from ai_crypto_trader.market_data.market_regime import MarketRegime, MarketRegimeAnalyzer, RegimeUnavailable
from ai_crypto_trader.strategies.regime_bull_hold import RegimeBullHoldStrategy

D = Decimal
START = datetime(2024, 1, 1, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)


def candle(index, *, open_="100", high="100", low="100", close="100"):
    return Candle("BTCUSDT", "1h", START + index * HOUR,
                  D(open_), D(high), D(low), D(close), D("1"), D("1"))


def view(quantity="0", cash="1000", equity="1000"):
    return PortfolioSnapshot("BTCUSDT", D(cash), D(quantity), D("0"), D("0"), D(equity))


def run_candles():
    rows = [candle(index) for index in range(228)]
    rows[225] = candle(225, open_="150", high="150", low="100", close="100")
    rows[227] = candle(227, open_="90", high="100", low="90", close="100")
    return tuple(rows)


def sequence_lookup(rows):
    regimes = (MarketRegime.SIDEWAYS, MarketRegime.BULL, MarketRegime.BULL,
               MarketRegime.BEAR, MarketRegime.BEAR)
    return dict(zip((row.close_time_utc for row in rows[223:]), regimes)).get


class RegimeBullHoldTests(unittest.TestCase):
    def test_entry_transitions_only_when_flat(self):
        for previous, current, quantity, action in (
            (MarketRegime.SIDEWAYS, MarketRegime.BULL, "0", "BUY"),
            (MarketRegime.BEAR, MarketRegime.BULL, "0", "BUY"),
            (MarketRegime.BULL, MarketRegime.BULL, "0", "HOLD"),
            (MarketRegime.SIDEWAYS, MarketRegime.SIDEWAYS, "0", "HOLD"),
            (MarketRegime.BEAR, MarketRegime.BEAR, "0", "HOLD"),
            (MarketRegime.BEAR, MarketRegime.SIDEWAYS, "0", "HOLD"),
            (MarketRegime.SIDEWAYS, MarketRegime.BULL, "1", "HOLD"),
        ):
            with self.subTest(previous=previous, current=current, quantity=quantity):
                rows = (candle(0), candle(1))
                lookup = {rows[0].close_time_utc: previous, rows[1].close_time_utc: current}.get
                strategy = RegimeBullHoldStrategy(lookup)
                signal = strategy.on_candle(rows, view(quantity))
                self.assertEqual(signal.action, action)
                if action == "BUY":
                    self.assertEqual(signal.budget_usdt, D("250"))
                    self.assertEqual(strategy.transition_stats.buy_signals, 1)

    def test_exit_transitions_only_with_position(self):
        for current in (MarketRegime.SIDEWAYS, MarketRegime.BEAR):
            with self.subTest(current=current):
                rows = (candle(0), candle(1))
                lookup = {rows[0].close_time_utc: MarketRegime.BULL,
                          rows[1].close_time_utc: current}.get
                self.assertEqual(RegimeBullHoldStrategy(lookup).on_candle(rows, view("2")),
                                 Signal("SELL", quantity=D("2")))
                self.assertEqual(RegimeBullHoldStrategy(lookup).on_candle(rows, view()), Signal())
        rows = (candle(0), candle(1))
        lookup = {rows[0].close_time_utc: MarketRegime.BULL,
                  rows[1].close_time_utc: MarketRegime.BULL}.get
        self.assertEqual(RegimeBullHoldStrategy(lookup).on_candle(rows, view("1")), Signal())

    def test_missing_regime_skips_and_unknown_regime_fails_closed(self):
        rows = (candle(0), candle(1))
        for lookup in (lambda _: None, lambda _: (_ for _ in ()).throw(RegimeUnavailable("missing"))):
            strategy = RegimeBullHoldStrategy(lookup)
            self.assertEqual(strategy.on_candle(rows, view()), Signal())
            self.assertEqual(strategy.transition_stats.skipped_unavailable, 1)
            self.assertEqual(strategy.transition_stats.buy_signals, 0)
        with self.assertRaises(RegimeUnavailable):
            RegimeBullHoldStrategy(lambda _: "BULL").on_candle(rows, view())
        with self.assertRaises(RegimeUnavailable):
            RegimeBullHoldStrategy(lambda _: MarketRegime.BULL).on_candle(
                rows, view("-1"))

    def test_engine_next_open_warmup_costs_analytics_and_benchmarks(self):
        rows = run_candles()
        period = BacktestPeriod.create(rows[224].open_time_utc, rows[-1].close_time_utc,
                                       "1h", warmup_candles=224)
        config = BacktestConfig(D("0.001"), D("5"))
        strategy = RegimeBullHoldStrategy(sequence_lookup(rows))
        result = BacktestEngine().run(rows, strategy, symbol="BTCUSDT", interval="1h",
                                      period=period, config=config, as_of_utc=START + 300 * HOUR)
        self.assertEqual(result.strategy_name, "1h BULL Regime Hold")
        self.assertEqual(result.warmup_candles, 224)
        self.assertEqual(result.evaluation_candles, 4)
        self.assertEqual(len(result.equity_curve), 4)
        self.assertEqual([trade.side for trade in result.ledger], ["BUY", "SELL"])
        buy, sell = result.ledger
        self.assertEqual(buy.signal_candle_timestamp_utc, rows[224].open_time_utc)
        self.assertEqual(buy.execution_candle_timestamp_utc, rows[225].open_time_utc)
        self.assertEqual(sell.signal_candle_timestamp_utc, rows[226].open_time_utc)
        self.assertEqual(sell.execution_candle_timestamp_utc, rows[227].open_time_utc)
        self.assertEqual(buy.execution_price, rows[225].open * D("1.0005"))
        self.assertEqual(sell.execution_price, rows[227].open * D("0.9995"))
        self.assertLessEqual(buy.gross_value + buy.commission, D("250"))
        self.assertEqual(sell.quantity, buy.quantity)
        self.assertEqual(result.final_quantity, D("0"))
        self.assertGreaterEqual(result.final_cash, D("0"))
        self.assertEqual(result.total_commissions,
                         sum((trade.commission for trade in result.ledger), D("0")))
        self.assertEqual(result.total_slippage,
                         sum((trade.slippage_cost for trade in result.ledger), D("0")))
        quality = analyze_backtest(result)
        self.assertEqual(quality.completed_trade_count, 1)
        self.assertEqual(quality.strategy_pnl_after_costs, result.total_pnl)
        stats = strategy.transition_stats
        self.assertEqual((stats.total_valid_transitions, stats.transitions_into_bull,
                          stats.buy_signals, stats.transitions_out_of_bull, stats.sell_signals,
                          stats.bull_hours_while_position_open, stats.skipped_unavailable),
                         (2, 1, 1, 1, 1, 1, 0))
        evaluation = rows[224:]
        matched = run_buy_and_hold(evaluation, symbol="BTCUSDT", interval="1h",
                                   config=config, initial_capital=result.initial_capital,
                                   allocation_pct=D("25"))
        full = run_buy_and_hold(evaluation, symbol="BTCUSDT", interval="1h",
                                config=config, initial_capital=result.initial_capital)
        cash = run_cash_benchmark(evaluation, symbol="BTCUSDT", interval="1h",
                                  config=config, initial_capital=result.initial_capital)
        for benchmark in (matched, full, cash):
            self.assertEqual((benchmark.evaluation_start, benchmark.evaluation_end),
                             (result.evaluation_start, result.evaluation_end))
        self.assertEqual(matched.initial_capital, D("1000"))

    def test_future_candle_does_not_change_signal(self):
        rows = run_candles()

        class Recording(RegimeBullHoldStrategy):
            def __init__(self, lookup):
                super().__init__(lookup)
                self.first_signal = None

            def on_candle(self, history, portfolio):
                signal = super().on_candle(history, portfolio)
                if history[-1].open_time_utc == rows[224].open_time_utc:
                    self.first_signal = signal
                return signal

        altered = list(rows)
        altered[225] = replace(altered[225], high=D("999"), close=D("999"))
        original, changed = Recording(sequence_lookup(rows)), Recording(sequence_lookup(rows))
        period = BacktestPeriod.create(rows[224].open_time_utc, rows[-1].close_time_utc,
                                       "1h", warmup_candles=224)
        for data, strategy in ((rows, original), (altered, changed)):
            BacktestEngine().run(data, strategy, symbol="BTCUSDT", interval="1h", period=period,
                                 as_of_utc=START + 300 * HOUR)
        self.assertEqual(original.first_signal, Signal("BUY", budget_usdt=D("250")))
        self.assertEqual(changed.first_signal, original.first_signal)

    def test_analyzer_future_hour_cannot_change_prior_transition(self):
        rows = [candle(index) for index in range(228)]
        rows[224] = candle(224, open_="100", high="200", low="100", close="200")
        changed = rows[:]
        changed[225] = candle(225, open_="100", high="999", low="100", close="999")
        signals = []
        for data in (rows, changed):
            report = MarketRegimeAnalyzer().analyze(
                data, evaluation_start=data[223].open_time_utc,
                evaluation_end=data[-1].close_time_utc,
                as_of_utc=START + 300 * HOUR,
            )
            strategy = RegimeBullHoldStrategy(report.regime_at)
            signals.append(strategy.on_candle(tuple(data[:225]), view()))
        self.assertEqual(signals, [Signal("BUY", budget_usdt=D("250"))] * 2)

    def test_engine_rejects_missing_warmup_unclosed_and_bad_candles(self):
        rows = run_candles()
        period = BacktestPeriod.create(rows[224].open_time_utc, rows[-1].close_time_utc,
                                       "1h", warmup_candles=224)
        strategy = RegimeBullHoldStrategy(sequence_lookup(rows))
        for data, clock in ((rows[1:], START + 300 * HOUR),
                            (rows, rows[-1].open_time_utc + timedelta(minutes=30)),
                            ((*rows[:100], rows[99], *rows[101:]), START + 300 * HOUR),
                            ((*rows[:100], replace(rows[100], low=D("200")), *rows[101:]),
                             START + 300 * HOUR)):
            with self.subTest(data=data[0].open_time_utc, clock=clock), self.assertRaises(BacktestError):
                BacktestEngine().run(data, strategy, symbol="BTCUSDT", interval="1h",
                                     period=period, as_of_utc=clock)
        with self.assertRaises(BacktestError):
            BacktestEngine().run(rows, strategy, symbol="BTCUSDT", interval="1h",
                                 period=BacktestPeriod.create(rows[224].open_time_utc,
                                                              rows[-1].close_time_utc, "1h",
                                                              warmup_candles=223),
                                 as_of_utc=START + 300 * HOUR)

    def test_cli_analyzer_offline_cache_and_paper_unchanged(self):
        rows = [candle(index) for index in range(228)]
        rows[224] = candle(224, open_="100", high="200", low="100", close="200")
        evaluation_start = rows[224].open_time_utc
        evaluation_end = rows[-1].close_time_utc
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "historical.sqlite3"
            paper_path = Path(directory) / "paper.sqlite3"
            HistoricalCache(cache_path).save(rows)
            paper_path.write_bytes(b"paper-ledger-sentinel")
            cache_before, paper_before = cache_path.read_bytes(), paper_path.read_bytes()
            flags = ["--symbol", "BTCUSDT", "--interval", "1h", "--strategy", "regime-bull-hold",
                     "--position-size-pct", "25", "--start", evaluation_start.isoformat(),
                     "--end", evaluation_end.isoformat()]
            with patch.dict("os.environ", {"HISTORICAL_CACHE_PATH": str(cache_path),
                                        "DATABASE_PATH": str(paper_path)}):
                with patch.object(socket.socket, "connect", side_effect=AssertionError("network called")):
                    output = StringIO()
                    with redirect_stdout(output):
                        self.assertEqual(main(flags), 0, output.getvalue())
                    for phrase in ("OFFLINE BACKTEST — 1h BULL Regime Hold",
                                   "Regime definition: SMA50 / SMA200 / SMA200 slope 24h",
                                   "Entry: non-BULL -> BULL", "Exit: BULL -> non-BULL",
                                   "Position size: 25%", "Total valid regime transitions: 2",
                                   "BUY signals generated: 1", "SELL signals generated: 1",
                                   "TRADE QUALITY", "Completed trades: 1", "BENCHMARKS",
                                   "MATCHED BUY & HOLD — 25% allocation", "FULL BUY & HOLD — 100% allocation"):
                        self.assertIn(phrase, output.getvalue())
                    self.assertEqual(cache_path.read_bytes(), cache_before)
                    self.assertEqual(paper_path.read_bytes(), paper_before)
                    for target, replacement in (("--interval", "15m"), ("--symbol", "ETHUSDT"),
                                                ("--position-size-pct", "30")):
                        changed = flags[:]
                        changed[changed.index(target) + 1] = replacement
                        with self.assertRaises(SystemExit), patch("sys.stderr", StringIO()):
                            main(changed)
                    # A separate temporary cache with one missing warm-up hour fails closed.
                    partial_path = Path(directory) / "partial.sqlite3"
                    HistoricalCache(partial_path).save(rows[1:])
                    with patch.dict("os.environ", {"HISTORICAL_CACHE_PATH": str(partial_path)}):
                        with redirect_stdout(StringIO()):
                            self.assertEqual(main(flags), 1)
            self.assertEqual(cache_path.read_bytes(), cache_before)
            self.assertEqual(paper_path.read_bytes(), paper_before)

    def test_bybit_client_exposes_no_order_methods(self):
        for name in ("create_order", "place_order", "amend_order", "cancel_order", "submit_order"):
            self.assertFalse(hasattr(BybitReadOnlyClient, name))


if __name__ == "__main__":
    unittest.main()
