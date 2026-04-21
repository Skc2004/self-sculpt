"""
In-memory prediction cache with optional Redis backend.

Falls back gracefully to LRU cache if Redis is unavailable.
"""

import hashlib
import json
from functools import lru_cache

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class PredictionCache:
    """Caching layer for model predictions.

    Uses Redis if available, otherwise falls back to in-memory LRU cache.
    Cache keys are SHA-256 hashes of the input pixel data.

    Args:
        redis_url: Redis connection URL (e.g., 'redis://localhost:6379').
        max_memory_items: Max items in the in-memory fallback cache.
        ttl: Time-to-live in seconds for Redis entries.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        max_memory_items: int = 1000,
        ttl: int = 3600,
    ):
        self.ttl = ttl
        self.redis_client = None

        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                print("  Cache: Connected to Redis")
            except (redis.ConnectionError, redis.RedisError):
                self.redis_client = None
                print("  Cache: Redis unavailable, using in-memory cache")
        else:
            print("  Cache: Redis package not installed, using in-memory cache")

        # In-memory fallback
        self._memory_cache = {}
        self._max_items = max_memory_items

    @staticmethod
    def _make_key(pixels: list) -> str:
        """Create a cache key from pixel data."""
        data = json.dumps(pixels, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def get(self, pixels: list) -> dict | None:
        """Look up a cached prediction.

        Args:
            pixels: Input pixel data (3D list).

        Returns:
            Cached prediction dict, or None if not found.
        """
        key = self._make_key(pixels)

        # Try Redis first
        if self.redis_client:
            try:
                data = self.redis_client.get(f"pred:{key}")
                if data:
                    return json.loads(data)
            except Exception:
                pass

        # Fallback to memory
        return self._memory_cache.get(key)

    def set(self, pixels: list, prediction: dict):
        """Cache a prediction result.

        Args:
            pixels: Input pixel data (3D list).
            prediction: Prediction dict to cache.
        """
        key = self._make_key(pixels)

        # Try Redis
        if self.redis_client:
            try:
                self.redis_client.setex(
                    f"pred:{key}",
                    self.ttl,
                    json.dumps(prediction),
                )
                return
            except Exception:
                pass

        # Fallback to memory (simple eviction: clear half when full)
        if len(self._memory_cache) >= self._max_items:
            keys = list(self._memory_cache.keys())
            for k in keys[: len(keys) // 2]:
                del self._memory_cache[k]

        self._memory_cache[key] = prediction
