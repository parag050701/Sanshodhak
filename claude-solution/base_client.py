"""Base client with rate limiting, retries, circuit breaker, and accurate error handling."""
import time
import logging
import requests
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class FailureReason(str, Enum):
    TIMEOUT = "timeout"
    HTTP_4XX = "http_client_error"
    HTTP_5XX = "http_server_error"
    RATE_LIMITED = "rate_limited"
    CONNECTION_ERROR = "connection_error"
    CIRCUIT_OPEN = "circuit_breaker_open"
    NOT_IMPLEMENTED = "not_implemented"
    UNKNOWN = "unknown"


@dataclass
class SourceHealth:
    """Tracks per-source health for circuit breaking."""
    source: str
    consecutive_failures: int = 0
    total_requests: int = 0
    total_failures: int = 0
    last_failure_reason: Optional[FailureReason] = None
    last_failure_time: Optional[datetime] = None
    circuit_open: bool = False
    circuit_open_until: Optional[datetime] = None

    # Thresholds
    failure_threshold: int = 5          # consecutive failures before opening circuit
    recovery_timeout_sec: int = 120     # how long circuit stays open

    def record_success(self):
        self.total_requests += 1
        self.consecutive_failures = 0
        if self.circuit_open and datetime.now() >= (self.circuit_open_until or datetime.min):
            self.circuit_open = False
            self.circuit_open_until = None

    def record_failure(self, reason: FailureReason):
        self.total_requests += 1
        self.total_failures += 1
        self.consecutive_failures += 1
        self.last_failure_reason = reason
        self.last_failure_time = datetime.now()
        if self.consecutive_failures >= self.failure_threshold:
            self.circuit_open = True
            self.circuit_open_until = datetime(
                *([datetime.now().year, datetime.now().month, datetime.now().day,
                   datetime.now().hour, datetime.now().minute, datetime.now().second])
            )
            # Simpler: just use time offset
            import time as _t
            self._open_epoch = _t.time() + self.recovery_timeout_sec

    def is_open(self) -> bool:
        if not self.circuit_open:
            return False
        import time as _t
        if _t.time() >= getattr(self, '_open_epoch', 0):
            # Half-open: allow one probe
            self.circuit_open = False
            return False
        return True

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.total_failures) / self.total_requests

    def summary(self) -> dict:
        return {
            "source": self.source,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "success_rate": f"{self.success_rate:.1%}",
            "consecutive_failures": self.consecutive_failures,
            "last_failure_reason": self.last_failure_reason,
            "circuit_open": self.circuit_open,
        }


# --- APIError replaces silent None returns ---

class APIError(Exception):
    """Carries the real HTTP status / reason so callers can log accurately."""
    def __init__(self, reason: FailureReason, status_code: Optional[int] = None,
                 message: str = ""):
        self.reason = reason
        self.status_code = status_code
        self.message = message
        super().__init__(f"[{reason}] HTTP {status_code}: {message}")


class BaseAPIClient(ABC):
    """
    Abstract base class for all API clients.

    Provides:
    - Rate limiting
    - Retry logic with exponential backoff
    - Session management
    - Accurate error classification (no false "timeout" labels)
    - Per-source circuit breaker
    - Health reporting
    """

    def __init__(
        self,
        base_url: str,
        rate_limit_delay: float = 1.0,
        max_retries: int = 3,
        timeout: int = 30,
        headers: Optional[Dict[str, str]] = None,
        source_name: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip('/')
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.timeout = timeout
        self._source_name = source_name or self.__class__.__name__.lower()

        self.session = requests.Session()
        default_headers = {
            'User-Agent': 'SanshodhakBot/1.0 (Research Paper Intelligence; mailto:contact@sanshodhak.edu)',
            'Accept': 'application/json',
        }
        if headers:
            default_headers.update(headers)
        self.session.headers.update(default_headers)

        self.last_request_time = datetime.min
        self.health = SourceHealth(source=self._source_name)

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _rate_limit(self):
        elapsed = (datetime.now() - self.last_request_time).total_seconds()
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = datetime.now()

    # ------------------------------------------------------------------
    # Core request method — returns (Response, None) or (None, APIError)
    # ------------------------------------------------------------------

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[Optional[requests.Response], Optional[APIError]]:
        """
        Make HTTP request with retries and accurate error classification.

        Returns:
            (response, None) on success
            (None, APIError) on failure — APIError.reason tells you *why*
        """
        if self.health.is_open():
            err = APIError(FailureReason.CIRCUIT_OPEN,
                           message=f"Circuit breaker open for {self._source_name}")
            logger.warning(str(err))
            return None, err

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        last_error: Optional[APIError] = None

        for attempt in range(self.max_retries):
            try:
                self._rate_limit()

                response = self.session.request(
                    method=method,
                    url=url,
                    params=params,
                    data=data,
                    json=json_data,
                    headers=headers,
                    timeout=self.timeout,
                    allow_redirects=True,
                )

                # --- Rate limited ---
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"[{self._source_name}] 429 Rate limited. Waiting {retry_after}s...")
                    time.sleep(retry_after)
                    last_error = APIError(FailureReason.RATE_LIMITED, 429,
                                          f"Retry-After={retry_after}s")
                    continue

                # --- Success ---
                if response.status_code < 400:
                    self.health.record_success()
                    return response, None

                # --- Client error (4xx) — don't retry, report accurately ---
                if 400 <= response.status_code < 500:
                    body_snippet = response.text[:200] if response.text else ""
                    err = APIError(FailureReason.HTTP_4XX, response.status_code,
                                   f"{url} | {body_snippet}")
                    logger.warning(f"[{self._source_name}] Client error: {err}")
                    self.health.record_failure(FailureReason.HTTP_4XX)
                    return None, err

                # --- Server error (5xx) — retry with backoff ---
                wait_time = 2 ** attempt
                last_error = APIError(FailureReason.HTTP_5XX, response.status_code,
                                      f"attempt {attempt+1}/{self.max_retries}")
                if attempt < self.max_retries - 1:
                    logger.warning(f"[{self._source_name}] {response.status_code} server error. "
                                   f"Retry in {wait_time}s...")
                    time.sleep(wait_time)
                    continue

            except requests.Timeout:
                wait_time = 2 ** attempt
                last_error = APIError(FailureReason.TIMEOUT, message=f"{url}")
                if attempt < self.max_retries - 1:
                    logger.warning(f"[{self._source_name}] Timeout. Retry {attempt+1}/{self.max_retries} in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                logger.error(f"[{self._source_name}] Timed out after {self.max_retries} attempts: {url}")

            except requests.ConnectionError as e:
                wait_time = 2 ** attempt
                last_error = APIError(FailureReason.CONNECTION_ERROR, message=str(e))
                if attempt < self.max_retries - 1:
                    logger.warning(f"[{self._source_name}] Connection error. Retry {attempt+1}/{self.max_retries}...")
                    time.sleep(wait_time)
                    continue
                logger.error(f"[{self._source_name}] Connection failed: {e}")

            except requests.RequestException as e:
                last_error = APIError(FailureReason.UNKNOWN, message=str(e))
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                logger.error(f"[{self._source_name}] Request failed: {e}")

        if last_error:
            self.health.record_failure(last_error.reason)
        return None, last_error or APIError(FailureReason.UNKNOWN)

    # ------------------------------------------------------------------
    # Convenience methods — return (Response | None) for backward compat
    # but log the real error reason
    # ------------------------------------------------------------------

    def get(self, endpoint: str, **kwargs) -> Optional[requests.Response]:
        resp, err = self._make_request('GET', endpoint, **kwargs)
        return resp

    def post(self, endpoint: str, **kwargs) -> Optional[requests.Response]:
        resp, err = self._make_request('POST', endpoint, **kwargs)
        return resp

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_health(self) -> dict:
        return self.health.summary()

    @abstractmethod
    def search(self, query: str, limit: int = 10, **kwargs) -> Any:
        pass

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
