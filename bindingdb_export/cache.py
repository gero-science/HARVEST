"""Cache helpers for BindingDB export."""

import logging
import shelve
import threading


class ThreadSafeCache:
    """Thread-safe wrapper for shelve cache with proper locking and batched writes."""

    def __init__(self, cache_path: str, cache_name: str):
        self.cache_path = cache_path
        self.cache_name = cache_name
        self._lock = threading.RLock()
        self._cache = None
        self._write_count = 0
        self._sync_interval = 100  # Sync every N writes for safety
        self._open_cache()

    def _open_cache(self):
        """Open the shelve cache with error handling."""
        try:
            # Use writeback=False to avoid keeping all data in memory and write in batch with _sync_interval
            self._cache = shelve.open(self.cache_path, writeback=False)
            logging.info(f"Successfully opened {self.cache_name} cache with {len(self._cache)} existing entries")
        except Exception as e:
            logging.error(f"Failed to open {self.cache_name} cache: {e}")
            raise

    def __contains__(self, key):
        with self._lock:
            try:
                return key in self._cache
            except Exception as e:
                logging.warning(f"Error checking key in {self.cache_name} cache: {e}")
                return False

    def __getitem__(self, key):
        with self._lock:
            try:
                return self._cache[key]
            except Exception as e:
                logging.warning(f"Error getting key from {self.cache_name} cache: {e}")
                raise

    def __setitem__(self, key, value):
        with self._lock:
            try:
                self._cache[key] = value
                self._write_count += 1
                # Only sync periodically instead of on every write
                if self._write_count >= self._sync_interval:
                    self._cache.sync()
                    self._write_count = 0
            except Exception as e:
                logging.warning(f"Error setting key in {self.cache_name} cache: {e}")
                raise

    def __delitem__(self, key):
        with self._lock:
            try:
                del self._cache[key]
                # Sync after delete since it's rare
                self._cache.sync()
            except Exception as e:
                logging.warning(f"Error deleting key from {self.cache_name} cache: {e}")
                raise

    def sync(self):
        """Explicitly sync the cache to disk."""
        with self._lock:
            try:
                if self._cache:
                    self._cache.sync()
                    logging.debug(f"Synced {self.cache_name} cache to disk")
            except Exception as e:
                logging.warning(f"Error syncing {self.cache_name} cache: {e}")

    def close(self):
        """Close the cache, ensuring all data is written."""
        with self._lock:
            try:
                if self._cache:
                    # Final sync before closing
                    self._cache.sync()
                    self._cache.close()
                    logging.info(f"Closed {self.cache_name} cache")
            except Exception as e:
                logging.warning(f"Error closing {self.cache_name} cache: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __len__(self):
        """Return the number of items in the cache."""
        with self._lock:
            try:
                return len(self._cache)
            except Exception as e:
                logging.warning(f"Error getting length of {self.cache_name} cache: {e}")
                return 0

    def items(self):
        """Return an iterator over (key, value) pairs."""
        with self._lock:
            try:
                return self._cache.items()
            except Exception as e:
                logging.warning(f"Error getting items from {self.cache_name} cache: {e}")
                return iter([])

    def keys(self):
        """Return an iterator over keys."""
        with self._lock:
            try:
                return self._cache.keys()
            except Exception as e:
                logging.warning(f"Error getting keys from {self.cache_name} cache: {e}")
                return iter([])

    def values(self):
        """Return an iterator over values."""
        with self._lock:
            try:
                return self._cache.values()
            except Exception as e:
                logging.warning(f"Error getting values from {self.cache_name} cache: {e}")
                return iter([])
