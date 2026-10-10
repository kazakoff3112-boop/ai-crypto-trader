# AI Crypto Trader

A production-style crypto trading research platform with offline backtesting,
market-regime analysis, execution simulation and an order-free **Shadow Mode**.
The name is a project label: the current strategies are deterministic rules,
not an AI model or a promise of profitable trading.

## What it does

- Validated BTC/ETH spot market data through the Bybit V5 public API.
- Offline historical backtesting with SMA crossover, Donchian breakout and
  Bollinger mean-reversion baselines.
- A BTCUSDT 1h Market Regime Analyzer: SMA50, SMA200 and 24h SMA200 slope,
  classifying BULL, BEAR and SIDEWAYS conditions.
- Regime-aware strategies, including BULL-filtered Donchian and BULL Regime Hold.
- Stateful Shadow Mode, gap detection and explicit controlled catch-up.
- A separate, manually confirmed paper trading engine and virtual 1,000 USDT
  portfolio, independent of any exchange balance.
- SQLite append-only ledgers, trade analytics and benchmark comparisons.
- Outgoing Telegram alerts and hardened systemd deployment templates.

## Safety design

- **Spot only:** BTCUSDT / ETHUSDT allowlist; derivatives, leverage, margin and
  short positions are rejected.
- The private read-only connectivity check permits **Bybit Demo only**.
  The application exposes no create/amend/cancel order methods.
- Shadow signals are observations, **not trades**. Shadow never calls the paper
  engine or executes exchange orders.
- Telegram is notifications-only: no incoming commands, webhooks or remote trades.
- Fail-closed checks reject invalid prices, stale quotes, malformed candles,
  gaps, duplicates and inconsistent persisted state.
- Financial calculations use `Decimal`; SQLite stores decimal values as text.
- Append-only financial and decision ledgers, transactional writes, unique
  decision keys and a process lock protect against partial writes and duplicates.
- Paper, historical-cache, shadow and notification databases are separate.
- Secrets are loaded from environment / an ignored `.env`. No runtime data,
  credentials or production database is distributed with this repository.

## Architecture

```text
Public Market Data (validated closed candles)
   ↓
Market Regime Analyzer
   ↓
Strategy
   ↓
Shadow Runner (FLAT / LONG; no execution)
   ↓
SQLite Decision Log
   ↓
Telegram Alerts (outgoing only)

Historical Cache (read-only) → Backtester → Trade / Regime Analytics
Explicit manual request     → Paper Engine → Portfolio / SQLite Ledger
```

These paths are deliberately separate: a shadow signal does not become a paper
trade, and a backtest cannot mutate the live shadow or paper ledger.

## Local setup

Python **3.9+**; the application has no third-party runtime dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp .env.example .env
python -m unittest discover -s tests -v
```

Keep `.env` local and never commit it. Public market data and Shadow Mode do
not require exchange credentials. Telegram is disabled until both notification
environment values are configured. Do not paste credentials into shell commands,
screenshots, issues or logs.

The test command is offline. Data downloads, live shadow runs, Telegram test
messages and paper executions are separate, explicit operator actions.

## Backtesting

Backtests read an existing validated historical cache without network access.
Each run uses an isolated virtual portfolio. Strategies see only closed candles
and an immutable portfolio snapshot. A signal at candle N executes at **N+1
open**, never at the signal candle's close.

Fees and adverse slippage are included. Slippage is embedded in execution price
and is not charged twice. Reports include completed-trade statistics, profit
factor, turnover, cost drag, equity, maximum drawdown, exposure and separately
marked open positions. Accounting identities are checked with Decimal tolerance.
Comparisons include Buy & Hold and allocation-matched benchmarks, with their
equity, costs and drawdowns. Regime attribution is diagnostic, not evidence that
a post-hoc filter will generalize.

Example using an **already populated local cache**:

```bash
python -m ai_crypto_trader.diagnostics.backtest \
  --symbol BTCUSDT --interval 15m --strategy donchian-breakout \
  --entry-lookback 96 --exit-lookback 48 --position-size-pct 25
```

No price history is included. See [the detailed usage guide](docs/usage.md) for
data preparation, explicit evaluation periods and analytics. The
[experiment registry](docs/experiment_registry.md) records hypotheses and
pre-selected validation periods. A viewed period is no longer independent OOS;
one positive backtest does not prove profitability.

## Forward Shadow Testing

Shadow Mode is designed for continuous VPS observation of fully closed BTCUSDT
1h candles. It persists a logical FLAT/LONG state, detects missing hours and
requires preview plus explicit confirmation for controlled catch-up (at most
72 hours). It never silently skips a gap or fabricates missing candles.

An illustrative cycle is `BUY_SIGNAL → LONG → SELL_SIGNAL → FLAT`.
This is a state transition example, **not a claim of an executed trade**.
Notifications say `NO TRADE EXECUTED`. HOLD and ALREADY_PROCESSED do not produce
routine alerts. Event keys deduplicate normal retries; a rare duplicate remains
possible if Telegram accepts a message immediately before local delivery-state
commit. Pending messages retry on later runs. Notification storage failure may
lose an alert but does not undo the recorded shadow decision.

Only one host should run live shadow observation. Keep development separate.
See [deployment templates and operator procedures](deploy/README.md). Deployment
is manual; cloning this repository does not install or start a service.

## Testing

**256 / 256 tests passed** using Python's standard-library **unittest**.
Fixtures and mocks cover safety guards, precision, portfolio accounting,
SQLite replay / transactions, market-data failures, no-look-ahead execution,
strategies, analytics, shadow state / catch-up, locks and notification dedup.
The full suite also passes with outbound socket connections blocked.

```bash
python -m unittest discover -s tests -v
```

## Tech stack

Python · Decimal · sqlite3 · unittest · Bybit V5 public API · Telegram Bot API ·
Linux systemd. Build tooling: setuptools. No ORM or optimizer is required.

## License

[MIT](LICENSE).

## Disclaimer

Research, educational and portfolio project. Not financial advice. No guarantee
of future profitability. Simulated fills do not reproduce all real-world market
conditions. The repository contains no exchange-order execution implementation.
