"""Health reporting and circuit breaker pattern for ingestion sources."""
import os
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class SourceHealth:
    """Health metrics for a single source."""
    source_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    consecutive_failures: int = 0
    last_request_time: Optional[float] = None
    last_failure_time: Optional[float] = None
    last_failure_reason: Optional[str] = None
    last_status_code: Optional[int] = None
    circuit_state: CircuitState = CircuitState.CLOSED
    circuit_opened_at: Optional[float] = None
    avg_response_time_ms: float = 0.0
    _lock: Lock = field(default_factory=Lock, repr=False)

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 to 1.0)."""
        if self.total_requests == 0:
            return 1.0  # No requests = assume healthy
        return self.successful_requests / self.total_requests

    @property
    def is_healthy(self) -> bool:
        """Check if source is healthy (circuit closed and good success rate)."""
        if self.circuit_state == CircuitState.OPEN:
            return False
        # Consider unhealthy if success rate < 50% after 5+ requests
        if self.total_requests >= 5 and self.success_rate < 0.5:
            return False
        return True

    def record_request(self, success: bool, response_time_ms: float = 0,
                       error_reason: Optional[str] = None,
                       status_code: Optional[int] = None):
        """Record a request outcome."""
        with self._lock:
            self.total_requests += 1
            self.last_request_time = time.time()

            if success:
                self.successful_requests += 1
                self.consecutive_failures = 0
                # If half-open and success, close circuit
                if self.circuit_state == CircuitState.HALF_OPEN:
                    self.circuit_state = CircuitState.CLOSED
                    self.circuit_opened_at = None
                    logger.info(f"Circuit closed for {self.source_name} (recovered)")
            else:
                self.failed_requests += 1
                self.consecutive_failures += 1
                self.last_failure_time = time.time()
                self.last_failure_reason = error_reason
                self.last_status_code = status_code

            # Update average response time
            if self.total_requests == 1:
                self.avg_response_time_ms = response_time_ms
            else:
                self.avg_response_time_ms = (
                    (self.avg_response_time_ms * (self.total_requests - 1) + response_time_ms)
                    / self.total_requests
                )

            # Check if we should open circuit
            if self.consecutive_failures >= 5 and self.circuit_state == CircuitState.CLOSED:
                self._open_circuit()

    def _open_circuit(self):
        """Open the circuit breaker."""
        self.circuit_state = CircuitState.OPEN
        self.circuit_opened_at = time.time()
        logger.warning(
            f"Circuit opened for {self.source_name} after {self.consecutive_failures} failures. "
            f"Last error: {self.last_failure_reason}"
        )

    def can_attempt(self) -> bool:
        """Check if a request can be attempted (circuit allows)."""
        with self._lock:
            if self.circuit_state == CircuitState.CLOSED:
                return True
            if self.circuit_state == CircuitState.OPEN:
                # Try half-open after 60 seconds
                if self.circuit_opened_at and (time.time() - self.circuit_opened_at) > 60:
                    self.circuit_state = CircuitState.HALF_OPEN
                    logger.info(f"Circuit half-open for {self.source_name} (testing recovery)")
                    return True
                return False
            return True  # HALF_OPEN allows one request

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_name": self.source_name,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round(self.success_rate, 3),
            "consecutive_failures": self.consecutive_failures,
            "circuit_state": self.circuit_state.value,
            "is_healthy": self.is_healthy,
            "last_failure_reason": self.last_failure_reason,
            "last_status_code": self.last_status_code,
            "avg_response_time_ms": round(self.avg_response_time_ms, 1),
            "last_request_time": datetime.fromtimestamp(self.last_request_time).isoformat()
            if self.last_request_time else None,
            "last_failure_time": datetime.fromtimestamp(self.last_failure_time).isoformat()
            if self.last_failure_time else None,
        }


class SourceHealthTracker:
    """Tracks health for all sources."""

    # Source reliability classification
    ROBUST_SOURCES = {
        "openalex", "arxiv", "europepmc", "semanticscholar",
        "crossref", "unpaywall"
    }

    # These sources are known to be unstable/problematic
    UNSTABLE_SOURCES = {
        "oatd", "openaire_oai", "scielo", "zenodo",
        "openaire", "eric", "hal", "springeropen"
    }

    # Placeholder sources that should be disabled
    PLACEHOLDER_SOURCES = {
        "arc_sieve", "arc_sieve"
    }

    def __init__(self):
        self._health: Dict[str, SourceHealth] = {}
        self._lock = Lock()
        self._load_env_flags()

    def _load_env_flags(self):
        """Load environment flags for source enablement."""
        self._disabled_sources = set()
        self._unstable_enabled = os.getenv("ENABLE_UNSTABLE_SOURCES", "").lower() == "true"
        self._placeholders_enabled = os.getenv("ENABLE_PLACEHOLDER_SOURCES", "").lower() == "true"

        # Always disable ArcSieve unless explicitly enabled
        if not self._placeholders_enabled:
            self._disabled_sources.add("arc_sieve")
            self._disabled_sources.add("arc_sieve")

        # Check individual source flags
        for source in self.UNSTABLE_SOURCES:
            env_var = f"ENABLE_{source.upper()}"
            if os.getenv(env_var, "").lower() == "false":
                self._disabled_sources.add(source)
            elif source in self.UNSTABLE_SOURCES and not self._unstable_enabled:
                # Unstable sources disabled by default unless env var explicitly enables
                pass  # We'll handle this in is_source_enabled

    def is_source_enabled(self, source_name: str) -> bool:
        """Check if a source is enabled based on env flags."""
        normalized = source_name.lower().replace("-", "_")

        # Check explicitly disabled
        if normalized in self._disabled_sources:
            return False

        # Placeholder sources require ENABLE_PLACEHOLDER_SOURCES=true
        if normalized in self.PLACEHOLDER_SOURCES and not self._placeholders_enabled:
            return False

        # Unstable sources require ENABLE_UNSTABLE_SOURCES=true (unless explicitly enabled individually)
        env_var = f"ENABLE_{normalized.upper()}"
        if normalized in self.UNSTABLE_SOURCES:
            if os.getenv(env_var, "").lower() == "true":
                return True
            return self._unstable_enabled

        return True

    def get_health(self, source_name: str) -> SourceHealth:
        """Get or create health tracker for a source."""
        with self._lock:
            if source_name not in self._health:
                self._health[source_name] = SourceHealth(source_name=source_name)
            return self._health[source_name]

    def record_request(self, source_name: str, success: bool,
                     response_time_ms: float = 0,
                     error_reason: Optional[str] = None,
                     status_code: Optional[int] = None):
        """Record a request outcome for a source."""
        health = self.get_health(source_name)
        health.record_request(success, response_time_ms, error_reason, status_code)

    def can_use_source(self, source_name: str) -> bool:
        """Check if source is enabled and circuit allows requests."""
        if not self.is_source_enabled(source_name):
            return False
        health = self.get_health(source_name)
        return health.can_attempt()

    def get_health_report(self) -> Dict[str, Any]:
        """Generate health report for all sources."""
        with self._lock:
            return {
                "timestamp": datetime.now().isoformat(),
                "sources": {
                    name: health.to_dict()
                    for name, health in self._health.items()
                },
                "summary": {
                    "total_sources": len(self._health),
                    "healthy_sources": sum(1 for h in self._health.values() if h.is_healthy),
                    "open_circuits": sum(1 for h in self._health.values()
                                       if h.circuit_state == CircuitState.OPEN),
                }
            }

    def print_health_report(self):
        """Print formatted health report to logs."""
        report = self.get_health_report()
        logger.info("=" * 80)
        logger.info("SOURCE HEALTH REPORT")
        logger.info("=" * 80)

        for name, health in sorted(report["sources"].items()):
            enabled = "[ok]" if self.is_source_enabled(name) else "[err]"
            circuit = health["circuit_state"].upper()
            if health["circuit_state"] == "open":
                circuit_icon = "🔴"
            elif health["circuit_state"] == "half_open":
                circuit_icon = "🟡"
            else:
                circuit_icon = "🟢"

            logger.info(
                f"{enabled} {circuit_icon} {name:20s} | "
                f"Success: {health['success_rate']:.1%} ({health['successful_requests']}/{health['total_requests']}) | "
                f"Circuit: {circuit:10s} | "
                f"Avg: {health['avg_response_time_ms']:.0f}ms"
            )
            if health["last_failure_reason"]:
                logger.info(
                    f"    Last error: {health['last_failure_reason']} "
                    f"(HTTP {health['last_status_code']})"
                )

        logger.info("-" * 80)
        summary = report["summary"]
        logger.info(f"Summary: {summary['healthy_sources']}/{summary['total_sources']} healthy, "
                   f"{summary['open_circuits']} open circuits")
        logger.info("=" * 80)

    def get_enabled_sources(self) -> List[str]:
        """Get list of currently enabled sources."""
        all_sources = set(self.ROBUST_SOURCES) | set(self.UNSTABLE_SOURCES) | self.PLACEHOLDER_SOURCES
        return [s for s in all_sources if self.is_source_enabled(s)]


# Global health tracker instance
_health_tracker: Optional[SourceHealthTracker] = None


def get_health_tracker() -> SourceHealthTracker:
    """Get or create the global health tracker."""
    global _health_tracker
    if _health_tracker is None:
        _health_tracker = SourceHealthTracker()
    return _health_tracker


def reset_health_tracker():
    """Reset the global health tracker (useful for testing)."""
    global _health_tracker
    _health_tracker = None
