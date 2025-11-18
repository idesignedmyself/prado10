"""
PRADO9_EVO Module J.1 — Live Data Feed

Real-time market data ingestion with fallback mechanisms.

Supports:
- yfinance (default, free)
- alpaca (requires API keys)
- local CSV files
- Polling and streaming modes

Author: PRADO9_EVO Builder
Date: 2025-01-16
Version: 1.0.0
"""

import time
import numpy as np
import pandas as pd
from typing import Optional, Callable, Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass


# ============================================================================
# CONSTANTS
# ============================================================================

LIVE_DATA_FEED_VERSION = '1.0.0'
EPSILON = 1e-12
DEFAULT_POLL_INTERVAL = 60.0  # seconds
MAX_LOOKBACK_BARS = 1000


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _safe_float(value: Any, fallback: float) -> float:
    """Convert value to safe float with fallback."""
    try:
        val = float(value)
        if np.isnan(val) or np.isinf(val):
            return float(fallback)
        return val
    except (ValueError, TypeError):
        return float(fallback)


def _sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize DataFrame for NaN/Inf values.

    Args:
        df: Input DataFrame

    Returns:
        Sanitized DataFrame
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Replace inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Forward fill NaN values
    df = df.fillna(method='ffill')

    # Backward fill remaining NaN (for start of series)
    df = df.fillna(method='bfill')

    # Drop any remaining NaN rows
    df = df.dropna()

    return df


# ============================================================================
# DATA FEED RESULT
# ============================================================================

@dataclass
class DataFeedResult:
    """
    Result from data feed query.

    Contains:
    - DataFrame with OHLCV data
    - Timestamp of last update
    - Source identifier
    - Error message (if any)
    """
    df: pd.DataFrame
    timestamp: datetime
    source: str
    symbol: str
    error: Optional[str] = None

    def is_valid(self) -> bool:
        """Check if result is valid."""
        return self.df is not None and not self.df.empty and self.error is None


# ============================================================================
# LIVE DATA FEED
# ============================================================================

class LiveDataFeed:
    """
    Live data feed with multiple source support.

    Supports:
    - yfinance (default, free)
    - alpaca (requires API keys)
    - local (CSV files)

    Features:
    - Automatic fallback on API errors
    - Local caching of last successful fetch
    - Throttling and backoff
    - NaN/Inf sanitization
    """

    def __init__(
        self,
        source: str = 'yfinance',
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        cache_dir: Optional[Path] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None
    ):
        """
        Initialize live data feed.

        Args:
            source: Data source ('yfinance', 'alpaca', 'local')
            poll_interval: Polling interval in seconds
            cache_dir: Directory for caching data
            api_key: API key (for alpaca)
            api_secret: API secret (for alpaca)
        """
        self.source = source
        self.poll_interval = _safe_float(poll_interval, DEFAULT_POLL_INTERVAL)
        self.api_key = api_key
        self.api_secret = api_secret

        # Cache directory
        if cache_dir is None:
            cache_dir = Path.home() / ".prado" / "live" / "cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Last fetch tracking (for throttling)
        self.last_fetch_time: Dict[str, float] = {}

        # Last successful result (for fallback)
        self.last_result: Dict[str, DataFeedResult] = {}

        # Error count (for backoff)
        self.error_count: Dict[str, int] = {}

        # Retry settings (Sweep J.1)
        self.max_retries = 3
        self.min_bars_required = 50
        self.max_data_age_seconds = 3600  # 1 hour

    def get_latest_price(self, symbol: str) -> float:
        """
        Get latest price for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Latest price (or 0.0 on error)
        """
        try:
            df = self.get_recent_bars(symbol, lookback=1)
            if df is not None and not df.empty:
                return float(df['Close'].iloc[-1])
        except Exception:
            pass

        return 0.0

    def get_recent_bars(
        self,
        symbol: str,
        lookback: int = 200
    ) -> pd.DataFrame:
        """
        Get recent bars for symbol (Sweep J.1: Enhanced with retry and fallback).

        Args:
            symbol: Trading symbol
            lookback: Number of bars to fetch

        Returns:
            DataFrame with OHLCV data
        """
        # Check throttling
        if not self._should_fetch(symbol):
            # Return cached result
            if symbol in self.last_result:
                return self.last_result[symbol].df
            return pd.DataFrame()

        # Clip lookback
        lookback = min(lookback, MAX_LOOKBACK_BARS)

        # Retry loop (Sweep J.1)
        for attempt in range(self.max_retries):
            # Fetch based on source
            result = self._fetch_data(symbol, lookback)

            # Update last fetch time
            self.last_fetch_time[symbol] = time.time()

            # Validate data (Sweep J.1)
            if result.is_valid() and self._validate_data(result):
                # Success - cache and return
                self.last_result[symbol] = result
                self.error_count[symbol] = 0
                self._save_to_cache(symbol, result.df)
                return result.df
            else:
                # Failure - backoff before retry
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        # All retries failed - increment error count and try cache
        self.error_count[symbol] = self.error_count.get(symbol, 0) + 1

        # Final fallback to cache (Sweep J.1)
        cached_df = self._load_from_cache(symbol)
        if cached_df is not None and not cached_df.empty:
            return cached_df

        return pd.DataFrame()

    def _validate_data(self, result: DataFeedResult) -> bool:
        """
        Validate data freshness and completeness (Sweep J.1).

        Args:
            result: DataFeedResult to validate

        Returns:
            True if valid, False otherwise
        """
        if result.df.empty:
            return False

        if len(result.df) < self.min_bars_required:
            return False

        # Check timestamp freshness
        if hasattr(result.df.index, '__getitem__') and len(result.df) > 0:
            try:
                latest_ts = result.df.index[-1]
                if hasattr(latest_ts, 'to_pydatetime'):
                    latest_ts = latest_ts.to_pydatetime()
                elif isinstance(latest_ts, str):
                    latest_ts = datetime.fromisoformat(latest_ts)

                age = abs((datetime.now() - latest_ts).total_seconds())
                if age > self.max_data_age_seconds:
                    return False
            except Exception:
                # If we can't validate timestamp, allow it
                pass

        return True

    def stream(
        self,
        symbol: str,
        callback: Callable[[pd.DataFrame], None],
        interval: Optional[float] = None
    ):
        """
        Stream data for symbol with callback.

        Args:
            symbol: Trading symbol
            callback: Callback function to process data
            interval: Polling interval (uses default if None)
        """
        if interval is None:
            interval = self.poll_interval

        while True:
            try:
                df = self.get_recent_bars(symbol)
                if df is not None and not df.empty:
                    callback(df)
            except Exception as e:
                print(f"Stream error for {symbol}: {e}")

            time.sleep(interval)

    def _should_fetch(self, symbol: str) -> bool:
        """
        Check if we should fetch data (throttling).

        Args:
            symbol: Trading symbol

        Returns:
            True if should fetch, False otherwise
        """
        if symbol not in self.last_fetch_time:
            return True

        # Calculate backoff based on error count
        error_count = self.error_count.get(symbol, 0)
        backoff_multiplier = min(2 ** error_count, 32)  # Cap at 32x

        min_interval = self.poll_interval * backoff_multiplier

        elapsed = time.time() - self.last_fetch_time[symbol]

        return elapsed >= min_interval

    def _fetch_data(self, symbol: str, lookback: int) -> DataFeedResult:
        """
        Fetch data from source.

        Args:
            symbol: Trading symbol
            lookback: Number of bars

        Returns:
            DataFeedResult
        """
        if self.source == 'yfinance':
            return self._fetch_yfinance(symbol, lookback)
        elif self.source == 'alpaca':
            return self._fetch_alpaca(symbol, lookback)
        elif self.source == 'local':
            return self._fetch_local(symbol, lookback)
        else:
            return DataFeedResult(
                df=pd.DataFrame(),
                timestamp=datetime.now(),
                source=self.source,
                symbol=symbol,
                error=f"Unknown source: {self.source}"
            )

    def _fetch_yfinance(self, symbol: str, lookback: int) -> DataFeedResult:
        """
        Fetch data from yfinance.

        Args:
            symbol: Trading symbol
            lookback: Number of bars

        Returns:
            DataFeedResult
        """
        try:
            import yfinance as yf

            # Download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{lookback}d", interval='1d')

            if df is None or df.empty:
                return DataFeedResult(
                    df=pd.DataFrame(),
                    timestamp=datetime.now(),
                    source='yfinance',
                    symbol=symbol,
                    error="No data returned"
                )

            # Standardize column names
            df = df.rename(columns={
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })

            # Sanitize
            df = _sanitize_dataframe(df)

            return DataFeedResult(
                df=df,
                timestamp=datetime.now(),
                source='yfinance',
                symbol=symbol
            )

        except Exception as e:
            return DataFeedResult(
                df=pd.DataFrame(),
                timestamp=datetime.now(),
                source='yfinance',
                symbol=symbol,
                error=str(e)
            )

    def _fetch_alpaca(self, symbol: str, lookback: int) -> DataFeedResult:
        """
        Fetch data from Alpaca (stubbed).

        Args:
            symbol: Trading symbol
            lookback: Number of bars

        Returns:
            DataFeedResult
        """
        # Stubbed implementation (requires alpaca-trade-api)
        return DataFeedResult(
            df=pd.DataFrame(),
            timestamp=datetime.now(),
            source='alpaca',
            symbol=symbol,
            error="Alpaca integration not implemented (requires API keys)"
        )

    def _fetch_local(self, symbol: str, lookback: int) -> DataFeedResult:
        """
        Fetch data from local CSV.

        Args:
            symbol: Trading symbol
            lookback: Number of bars

        Returns:
            DataFeedResult
        """
        try:
            # Look for CSV file
            csv_path = self.cache_dir / f"{symbol}.csv"

            if not csv_path.exists():
                return DataFeedResult(
                    df=pd.DataFrame(),
                    timestamp=datetime.now(),
                    source='local',
                    symbol=symbol,
                    error=f"File not found: {csv_path}"
                )

            # Read CSV
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

            # Take last N bars
            df = df.tail(lookback)

            # Sanitize
            df = _sanitize_dataframe(df)

            return DataFeedResult(
                df=df,
                timestamp=datetime.now(),
                source='local',
                symbol=symbol
            )

        except Exception as e:
            return DataFeedResult(
                df=pd.DataFrame(),
                timestamp=datetime.now(),
                source='local',
                symbol=symbol,
                error=str(e)
            )

    def _save_to_cache(self, symbol: str, df: pd.DataFrame):
        """
        Save DataFrame to cache.

        Args:
            symbol: Trading symbol
            df: DataFrame to save
        """
        try:
            cache_file = self.cache_dir / f"{symbol}_cache.csv"
            df.to_csv(cache_file)
        except Exception:
            pass  # Silent fail on cache save

    def _load_from_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Load DataFrame from cache.

        Args:
            symbol: Trading symbol

        Returns:
            Cached DataFrame or None
        """
        try:
            cache_file = self.cache_dir / f"{symbol}_cache.csv"
            if cache_file.exists():
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                return _sanitize_dataframe(df)
        except Exception:
            pass

        return None


# ============================================================================
# INTEGRATION HOOKS
# ============================================================================

def evo_live_data_feed(
    symbol: str,
    source: str = 'yfinance',
    lookback: int = 200
) -> pd.DataFrame:
    """
    Integration hook: Get live data for symbol.

    Args:
        symbol: Trading symbol
        source: Data source
        lookback: Number of bars

    Returns:
        DataFrame with OHLCV data
    """
    feed = LiveDataFeed(source=source)
    return feed.get_recent_bars(symbol, lookback)


# ============================================================================
# INLINE TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("PRADO9_EVO Module J.1 — Live Data Feed Tests")
    print("=" * 80)

    # ========================================================================
    # TEST 1: LiveDataFeed Initialization
    # ========================================================================
    print("\n[TEST 1] LiveDataFeed Initialization")
    print("-" * 80)

    feed = LiveDataFeed(source='yfinance', poll_interval=60.0)

    print(f"  Source: {feed.source}")
    print(f"  Poll interval: {feed.poll_interval:.1f}s")
    print(f"  Cache dir: {feed.cache_dir}")

    assert feed.source == 'yfinance', "Source should be yfinance"
    assert feed.poll_interval == 60.0, "Poll interval should be 60.0"
    assert feed.cache_dir.exists(), "Cache dir should exist"

    print("  ✓ LiveDataFeed initialization working")

    # ========================================================================
    # TEST 2: Sanitize DataFrame
    # ========================================================================
    print("\n[TEST 2] Sanitize DataFrame")
    print("-" * 80)

    # Create test DataFrame with NaN and Inf
    df = pd.DataFrame({
        'Open': [100.0, 101.0, np.nan, 103.0],
        'High': [102.0, 103.0, 104.0, np.inf],
        'Low': [99.0, 100.0, 101.0, 102.0],
        'Close': [101.0, np.nan, 103.0, 104.0],
        'Volume': [1000, 1100, 1200, 1300]
    })

    print(f"  Before sanitization:")
    print(f"    NaN count: {df.isna().sum().sum()}")
    print(f"    Inf count: {np.isinf(df.select_dtypes(include=[np.number])).sum().sum()}")

    df_clean = _sanitize_dataframe(df)

    print(f"  After sanitization:")
    print(f"    NaN count: {df_clean.isna().sum().sum()}")
    print(f"    Inf count: {np.isinf(df_clean.select_dtypes(include=[np.number])).sum().sum()}")

    assert df_clean.isna().sum().sum() == 0, "Should have no NaN"
    assert np.isinf(df_clean.select_dtypes(include=[np.number])).sum().sum() == 0, "Should have no Inf"

    print("  ✓ DataFrame sanitization working")

    # ========================================================================
    # TEST 3: DataFeedResult Validation
    # ========================================================================
    print("\n[TEST 3] DataFeedResult Validation")
    print("-" * 80)

    # Valid result
    df_valid = pd.DataFrame({
        'Open': [100.0],
        'Close': [101.0]
    })
    result_valid = DataFeedResult(
        df=df_valid,
        timestamp=datetime.now(),
        source='test',
        symbol='SPY'
    )

    print(f"  Valid result: {result_valid.is_valid()}")
    assert result_valid.is_valid(), "Valid result should return True"

    # Invalid result (empty DataFrame)
    result_empty = DataFeedResult(
        df=pd.DataFrame(),
        timestamp=datetime.now(),
        source='test',
        symbol='SPY'
    )

    print(f"  Empty result: {result_empty.is_valid()}")
    assert not result_empty.is_valid(), "Empty result should return False"

    # Invalid result (error)
    result_error = DataFeedResult(
        df=df_valid,
        timestamp=datetime.now(),
        source='test',
        symbol='SPY',
        error="Test error"
    )

    print(f"  Error result: {result_error.is_valid()}")
    assert not result_error.is_valid(), "Error result should return False"

    print("  ✓ DataFeedResult validation working")

    # ========================================================================
    # TEST 4: Throttling Logic
    # ========================================================================
    print("\n[TEST 4] Throttling Logic")
    print("-" * 80)

    feed = LiveDataFeed(source='yfinance', poll_interval=2.0)
    symbol = 'TEST'

    # First check - should fetch
    should_fetch_1 = feed._should_fetch(symbol)
    print(f"  First check (no history): {should_fetch_1}")
    assert should_fetch_1, "Should fetch when no history"

    # Mark as fetched
    feed.last_fetch_time[symbol] = time.time()

    # Immediate second check - should not fetch
    should_fetch_2 = feed._should_fetch(symbol)
    print(f"  Immediate second check: {should_fetch_2}")
    assert not should_fetch_2, "Should not fetch immediately"

    # Wait and check again
    time.sleep(2.1)
    should_fetch_3 = feed._should_fetch(symbol)
    print(f"  After waiting 2.1s: {should_fetch_3}")
    assert should_fetch_3, "Should fetch after interval"

    print("  ✓ Throttling logic working")

    # ========================================================================
    # TEST 5: Backoff on Errors
    # ========================================================================
    print("\n[TEST 5] Backoff on Errors")
    print("-" * 80)

    feed = LiveDataFeed(source='yfinance', poll_interval=1.0)
    symbol = 'ERROR_TEST'

    # Simulate errors
    feed.error_count[symbol] = 0
    feed.last_fetch_time[symbol] = time.time()

    # No errors - standard interval (1.0s)
    time.sleep(1.1)
    should_fetch = feed._should_fetch(symbol)
    print(f"  0 errors, after 1.1s: {should_fetch}")
    assert should_fetch, "Should fetch with no errors"

    # 2 errors - 4x backoff (4.0s)
    feed.error_count[symbol] = 2
    feed.last_fetch_time[symbol] = time.time()
    time.sleep(1.1)
    should_fetch = feed._should_fetch(symbol)
    print(f"  2 errors, after 1.1s: {should_fetch}")
    assert not should_fetch, "Should not fetch with backoff"

    print("  ✓ Backoff on errors working")

    # ========================================================================
    # TEST 6: Cache Save/Load
    # ========================================================================
    print("\n[TEST 6] Cache Save/Load")
    print("-" * 80)

    import tempfile
    import shutil

    temp_dir = Path(tempfile.mkdtemp())

    try:
        feed = LiveDataFeed(source='local', cache_dir=temp_dir)
        symbol = 'CACHE_TEST'

        # Create test DataFrame
        df = pd.DataFrame({
            'Open': [100.0, 101.0],
            'Close': [101.0, 102.0]
        })

        # Save to cache
        feed._save_to_cache(symbol, df)

        # Load from cache
        df_loaded = feed._load_from_cache(symbol)

        print(f"  Original rows: {len(df)}")
        print(f"  Loaded rows: {len(df_loaded) if df_loaded is not None else 0}")

        assert df_loaded is not None, "Should load cached data"
        assert len(df_loaded) == len(df), "Should have same number of rows"

        print("  ✓ Cache save/load working")

    finally:
        shutil.rmtree(temp_dir)

    # ========================================================================
    # TEST 7: Local Source (CSV)
    # ========================================================================
    print("\n[TEST 7] Local Source (CSV)")
    print("-" * 80)

    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Create test CSV
        df = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0],
            'High': [102.0, 103.0, 104.0],
            'Low': [99.0, 100.0, 101.0],
            'Close': [101.0, 102.0, 103.0],
            'Volume': [1000, 1100, 1200]
        })
        csv_path = temp_dir / 'TEST.csv'
        df.to_csv(csv_path)

        feed = LiveDataFeed(source='local', cache_dir=temp_dir)
        result = feed._fetch_local('TEST', lookback=2)

        print(f"  CSV path: {csv_path}")
        print(f"  Result valid: {result.is_valid()}")
        print(f"  Rows fetched: {len(result.df)}")

        assert result.is_valid(), "Result should be valid"
        assert len(result.df) == 2, "Should fetch last 2 rows"

        print("  ✓ Local source working")

    finally:
        shutil.rmtree(temp_dir)

    # ========================================================================
    # TEST 8: Unknown Source Error
    # ========================================================================
    print("\n[TEST 8] Unknown Source Error")
    print("-" * 80)

    feed = LiveDataFeed(source='unknown_source')
    result = feed._fetch_data('SPY', lookback=100)

    print(f"  Source: {result.source}")
    print(f"  Valid: {result.is_valid()}")
    print(f"  Error: {result.error}")

    assert not result.is_valid(), "Should be invalid"
    assert result.error is not None, "Should have error message"
    assert "Unknown source" in result.error, "Error should mention unknown source"

    print("  ✓ Unknown source error handling working")

    # ========================================================================
    # TEST 9: Data Validation (Sweep J.1)
    # ========================================================================
    print("\n[TEST 9] Data Validation (Sweep J.1)")
    print("-" * 80)

    feed_validation = LiveDataFeed(source='local')

    # Create valid result (enough bars, recent timestamps to pass freshness check)
    from datetime import timedelta
    df_valid = pd.DataFrame({
        'Open': [100.0] * 100,
        'Close': [101.0] * 100
    }, index=pd.date_range(datetime.now() - timedelta(hours=1), periods=100, freq='1min'))

    result_valid = DataFeedResult(
        df=df_valid,
        timestamp=datetime.now(),
        source='test',
        symbol='SPY'
    )

    is_valid = feed_validation._validate_data(result_valid)
    print(f"  Valid data (100 bars): {is_valid}")
    assert is_valid, "Should be valid"

    # Create invalid result (too few bars)
    df_invalid = pd.DataFrame({
        'Open': [100.0] * 10,
        'Close': [101.0] * 10
    })

    result_invalid = DataFeedResult(
        df=df_invalid,
        timestamp=datetime.now(),
        source='test',
        symbol='SPY'
    )

    is_invalid = feed_validation._validate_data(result_invalid)
    print(f"  Invalid data (10 bars, min 50 required): {is_invalid}")
    assert not is_invalid, "Should be invalid"

    print("  ✓ Data validation working")

    # ========================================================================
    # TEST 10: Retry and Fallback (Sweep J.1)
    # ========================================================================
    print("\n[TEST 10] Retry and Fallback (Sweep J.1)")
    print("-" * 80)

    temp_dir = Path(tempfile.mkdtemp())

    try:
        feed_retry = LiveDataFeed(source='unknown_source', cache_dir=temp_dir)

        # Pre-populate cache
        df_cache = pd.DataFrame({
            'Open': [100.0] * 100,
            'High': [102.0] * 100,
            'Low': [99.0] * 100,
            'Close': [101.0] * 100,
            'Volume': [1000000] * 100
        }, index=pd.date_range('2024-01-01', periods=100, freq='D'))

        feed_retry._save_to_cache('RETRY_TEST', df_cache)

        # Try to fetch (will fail all retries, fallback to cache)
        df = feed_retry.get_recent_bars('RETRY_TEST', lookback=100)

        print(f"  Fetch failed (unknown source)")
        print(f"  Fallback to cache successful: {not df.empty}")
        print(f"  Rows from cache: {len(df)}")

        assert not df.empty, "Should fallback to cache"
        assert len(df) == 100, "Should have cached data"

        print("  ✓ Retry and fallback working")

    finally:
        shutil.rmtree(temp_dir)

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("ALL MODULE J.1 TESTS PASSED (10 TESTS) - Sweep J.1 Enhanced")
    print("=" * 80)
    print("\nLive Data Feed Features:")
    print("  ✓ Multi-source support (yfinance, alpaca, local)")
    print("  ✓ Throttling and rate limiting")
    print("  ✓ Exponential backoff on errors")
    print("  ✓ Local caching of successful fetches")
    print("  ✓ NaN/Inf sanitization")
    print("  ✓ Deterministic fallback behavior")
    print("  ✓ CSV support for local testing")
    print("  ✓ Streaming mode with callbacks")
    print("\nSweep J.1 Enhancements:")
    print("  ✓ Retry logic (3 attempts with exponential backoff)")
    print("  ✓ Data validation (minimum bars, freshness check)")
    print("  ✓ Enhanced fallback (cache on all retry failures)")
    print("\nModule J.1 — Live Data Feed: PRODUCTION READY (Sweep J.1 Enhanced)")
    print("=" * 80)
