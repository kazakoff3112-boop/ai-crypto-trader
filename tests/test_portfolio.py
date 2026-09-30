from dataclasses import FrozenInstanceError
from datetime import timezone
from decimal import Decimal
import unittest

from ai_crypto_trader.config import Settings
from ai_crypto_trader.portfolio import InsufficientAssets, InsufficientFunds, Ledger, Portfolio, PortfolioError


class PortfolioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.portfolio = Portfolio()

    def snapshot(self) -> tuple:
        return (
            self.portfolio.cash,
            self.portfolio.btc,
            self.portfolio.eth,
            self.portfolio.realized_pnl,
            self.portfolio.total_commission,
            self.portfolio.total_slippage,
            self.portfolio.ledger.entries,
        )

    def test_starting_balance_is_internal_1000_usdt(self) -> None:
        self.assertEqual(self.portfolio.cash, Decimal("1000"))
        self.assertEqual(self.portfolio.btc, Decimal("0"))
        self.assertEqual(self.portfolio.eth, Decimal("0"))

    def test_successful_btc_buy_and_ledger_fields(self) -> None:
        entry = self.portfolio.buy("BTCUSDT", "0.01", "20000")
        self.assertEqual(self.portfolio.btc, Decimal("0.01"))
        self.assertEqual(self.portfolio.cash, Decimal("799.80000"))
        self.assertEqual(entry.gross_value, Decimal("200.00"))
        self.assertEqual(entry.commission, Decimal("0.20000"))
        self.assertEqual(entry.position_after, Decimal("0.01"))
        self.assertEqual(entry.timestamp_utc.tzinfo, timezone.utc)
        self.assertTrue(entry.id)

    def test_successful_eth_buy(self) -> None:
        self.portfolio.buy("ETHUSDT", "0.1", "1000")
        self.assertEqual(self.portfolio.eth, Decimal("0.1"))
        self.assertEqual(self.portfolio.btc, Decimal("0"))

    def test_commission_and_slippage_reduce_cash_and_are_tracked(self) -> None:
        entry = self.portfolio.buy("BTCUSDT", "0.01", "10000", slippage_cost="2")
        self.assertEqual(self.portfolio.cash, Decimal("897.90000"))
        self.assertEqual(self.portfolio.total_commission, Decimal("0.10000"))
        self.assertEqual(self.portfolio.total_slippage, Decimal("2"))
        self.assertEqual(entry.slippage_cost, Decimal("2"))

    def test_successful_sell_and_realized_pnl(self) -> None:
        self.portfolio.buy("BTCUSDT", "0.01", "10000")
        entry = self.portfolio.sell("BTCUSDT", "0.01", "12000")
        self.assertEqual(self.portfolio.btc, Decimal("0"))
        self.assertEqual(self.portfolio.cash, Decimal("1019.78000"))
        self.assertEqual(entry.realized_pnl, Decimal("19.78000"))
        self.assertEqual(self.portfolio.realized_pnl, Decimal("19.78000"))

    def test_unrealized_pnl_and_equity(self) -> None:
        self.portfolio.buy("BTCUSDT", "0.01", "10000")
        self.assertEqual(self.portfolio.average_entry_price("BTCUSDT"), Decimal("10010.000"))
        self.assertEqual(self.portfolio.unrealized_pnl("BTCUSDT", "11000"), Decimal("9.90000"))
        self.assertEqual(self.portfolio.equity({"BTCUSDT": "11000"}), Decimal("1009.90000"))

    def test_buy_above_cash_is_atomic(self) -> None:
        before = self.snapshot()
        with self.assertRaises(InsufficientFunds):
            self.portfolio.buy("BTCUSDT", "1", "1000")
        self.assertEqual(self.snapshot(), before)

    def test_sell_above_position_is_atomic(self) -> None:
        self.portfolio.buy("BTCUSDT", "0.01", "10000")
        before = self.snapshot()
        with self.assertRaises(InsufficientAssets):
            self.portfolio.sell("BTCUSDT", "0.02", "11000")
        self.assertEqual(self.snapshot(), before)

    def test_unknown_symbol_is_atomic(self) -> None:
        before = self.snapshot()
        with self.assertRaises(PortfolioError):
            self.portfolio.buy("SOLUSDT", "1", "10")
        self.assertEqual(self.snapshot(), before)

    def test_zero_and_negative_quantity_are_atomic(self) -> None:
        for quantity in ("0", "-1"):
            with self.subTest(quantity=quantity):
                before = self.snapshot()
                with self.assertRaises(PortfolioError):
                    self.portfolio.buy("BTCUSDT", quantity, "100")
                self.assertEqual(self.snapshot(), before)

    def test_balances_never_become_negative(self) -> None:
        self.portfolio.buy("BTCUSDT", "0.01", "10000")
        self.portfolio.sell("BTCUSDT", "0.01", "9000")
        self.assertGreaterEqual(self.portfolio.cash, Decimal("0"))
        self.assertGreaterEqual(self.portfolio.btc, Decimal("0"))
        self.assertGreaterEqual(self.portfolio.eth, Decimal("0"))

    def test_btc_and_eth_are_independent(self) -> None:
        self.portfolio.buy("BTCUSDT", "0.01", "10000")
        self.portfolio.buy("ETHUSDT", "0.1", "1000")
        self.portfolio.sell("BTCUSDT", "0.005", "11000")
        self.assertEqual(self.portfolio.btc, Decimal("0.005"))
        self.assertEqual(self.portfolio.eth, Decimal("0.1"))

    def test_fee_rate_comes_from_configuration(self) -> None:
        portfolio = Portfolio(Settings(spot_fee_rate=Decimal("0.002")))
        entry = portfolio.buy("BTCUSDT", "0.01", "10000")
        self.assertEqual(entry.commission, Decimal("0.20000"))

    def test_explicit_test_starting_cash_is_supported(self) -> None:
        portfolio = Portfolio(initial_cash="50")
        self.assertEqual(portfolio.cash, Decimal("50"))

    def test_injected_empty_ledger_is_used(self) -> None:
        ledger = Ledger()
        portfolio = Portfolio(ledger=ledger)
        portfolio.buy("BTCUSDT", "0.01", "10000")
        self.assertIs(portfolio.ledger, ledger)
        self.assertEqual(len(ledger), 1)

    def test_float_inputs_are_rejected(self) -> None:
        with self.assertRaises(PortfolioError):
            self.portfolio.buy("BTCUSDT", 0.01, "10000")

    def test_ledger_entries_are_immutable_and_collection_is_read_only(self) -> None:
        entry = self.portfolio.buy("BTCUSDT", "0.01", "10000")
        with self.assertRaises(FrozenInstanceError):
            entry.quantity = Decimal("99")  # type: ignore[misc]
        self.assertIsInstance(self.portfolio.ledger.entries, tuple)


if __name__ == "__main__":
    unittest.main()
