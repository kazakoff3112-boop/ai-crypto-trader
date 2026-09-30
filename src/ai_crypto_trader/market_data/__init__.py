"""Validated market data. Network transport is imported only when requested."""

__all__ = ["MarketDataClient", "MarketDataUnavailable", "SpotQuote"]


def __getattr__(name: str):
    if name in __all__:
        from .client import MarketDataClient, MarketDataUnavailable, SpotQuote
        return {
            "MarketDataClient": MarketDataClient,
            "MarketDataUnavailable": MarketDataUnavailable,
            "SpotQuote": SpotQuote,
        }[name]
    raise AttributeError(name)
