"""
metrics.py
----------
Simple in-memory metrics collector for monitoring the RAG API.

Tracks:
  - Total requests per endpoint
  - Average/p95 latency per endpoint
  - Total queries and uploads
  - Error count and recent errors
  - Uptime

Production alternative: swap this for Prometheus client or DataDog.
This is intentionally dependency-free for easy local setup.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class MetricsCollector:
    start_time: float = field(default_factory=time.time)
    request_counts: dict = field(default_factory=lambda: defaultdict(int))
    request_latencies: dict = field(default_factory=lambda: defaultdict(list))
    query_count: int = 0
    upload_count: int = 0
    total_chunks_indexed: int = 0
    error_count: int = 0
    recent_errors: list = field(default_factory=list)
    recent_queries: list = field(default_factory=list)

    def record_request(self, path: str, latency_ms: float):
        self.request_counts[path] += 1
        latencies = self.request_latencies[path]
        latencies.append(latency_ms)
        # Keep only last 1000 latencies per path to avoid memory growth
        if len(latencies) > 1000:
            self.request_latencies[path] = latencies[-500:]

    def record_query(self, question: str):
        self.query_count += 1
        self.recent_queries.append(question)
        if len(self.recent_queries) > 50:
            self.recent_queries = self.recent_queries[-50:]

    def record_upload(self, filename: str, chunks: int):
        self.upload_count += 1
        self.total_chunks_indexed += chunks

    def record_error(self, endpoint: str, error: str):
        self.error_count += 1
        self.recent_errors.append({"endpoint": endpoint, "error": error, "time": time.time()})
        if len(self.recent_errors) > 20:
            self.recent_errors = self.recent_errors[-20:]

    def _percentile(self, values: list, p: float) -> float:
        if not values:
            return 0
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * p / 100)
        return round(sorted_vals[min(idx, len(sorted_vals) - 1)], 1)

    def summary(self) -> dict:
        uptime_seconds = time.time() - self.start_time
        hours = int(uptime_seconds // 3600)
        minutes = int((uptime_seconds % 3600) // 60)

        endpoint_stats = {}
        for path, count in self.request_counts.items():
            latencies = self.request_latencies.get(path, [])
            endpoint_stats[path] = {
                "count": count,
                "avg_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
                "p95_ms": self._percentile(latencies, 95),
            }

        return {
            "uptime": f"{hours}h {minutes}m",
            "total_requests": sum(self.request_counts.values()),
            "total_queries": self.query_count,
            "total_uploads": self.upload_count,
            "total_chunks_indexed": self.total_chunks_indexed,
            "error_count": self.error_count,
            "endpoints": endpoint_stats,
            "recent_errors": self.recent_errors[-5:],
        }


# Singleton instance used across the app
metrics = MetricsCollector()
