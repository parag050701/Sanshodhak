"""Base client with rate limiting, retries, and error handling."""
import time
import logging
import os
import requests
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RequestResult:
    """Result of an HTTP request with detailed error information."""
    response: Optional[requests.Response] = None
    error_type: Optional[str] = None  # 'timeout', '4xx', '5xx', 'rate_limit', 'connection', 'unknown'
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    attempts: int = 0

    @property
    def success(self) -> bool:
        return self.response is not None and self.status_code is not None and self.status_code < 400

    def to_error_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for error reporting."""
        return {
            'error_type': self.error_type,
            'status_code': self.status_code,
            'error_message': self.error_message,
            'attempts': self.attempts
        }


class BaseAPIClient(ABC):
    """
    Abstract base class for all API clients.
    
    Provides:
    - Rate limiting
    - Retry logic with exponential backoff
    - Session management
    - Error handling
    """
    
    def __init__(
        self,
        base_url: str,
        rate_limit_delay: float = 1.0,
        max_retries: int = 3,
        timeout: int = 30,
        headers: Optional[Dict[str, str]] = None
    ):
        self.base_url = base_url.rstrip('/')
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_retry_after_s = int(os.getenv("MAX_RETRY_AFTER_S", "8"))
        self.fail_fast_on_429 = os.getenv("FAIL_FAST_ON_429", "true").lower() == "true"
        
        # Session with default headers
        self.session = requests.Session()
        default_headers = {
            'User-Agent': 'SanshodhakBot/1.0 (Research Paper Intelligence; mailto:contact@sanshodhak.edu)',
            'Accept': 'application/json',
        }
        if headers:
            default_headers.update(headers)
        self.session.headers.update(default_headers)
        
        # Rate limiting
        self.last_request_time = datetime.min
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = (datetime.now() - self.last_request_time).total_seconds()
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = datetime.now()
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> RequestResult:
        """
        Make HTTP request with retries and detailed error reporting.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (relative to base_url)
            params: URL parameters
            data: Form data
            json_data: JSON body
            headers: Additional headers

        Returns:
            RequestResult with response and detailed error information
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        result = RequestResult()

        for attempt in range(self.max_retries):
            result.attempts = attempt + 1
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
                    allow_redirects=True
                )

                result.response = response
                result.status_code = response.status_code

                # Handle rate limiting (429)
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    wait_s = min(retry_after, self.max_retry_after_s)
                    logger.warning(
                        f"Rate limited (429). Retry-After={retry_after}s, capped wait={wait_s}s"
                    )
                    result.error_type = 'rate_limit'
                    result.error_message = f"Rate limited, retry-after={retry_after}s, waited={wait_s}s"
                    if self.fail_fast_on_429:
                        return result
                    time.sleep(wait_s)
                    continue

                # Success
                if response.status_code < 400:
                    return result

                # Client error (4xx) - don't retry
                if 400 <= response.status_code < 500:
                    result.error_type = '4xx'
                    result.error_message = f"Client error {response.status_code}: {url}"
                    logger.warning(f"Client error {response.status_code}: {url}")
                    return result

                # Server error (5xx) - retry
                result.error_type = '5xx'
                result.error_message = f"Server error {response.status_code}"
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Server error {response.status_code}. Retry {attempt+1}/{self.max_retries} in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                return result

            except requests.Timeout as e:
                result.error_type = 'timeout'
                result.error_message = f"Request timeout after {self.timeout}s"
                if attempt < self.max_retries - 1:
                    logger.warning(f"Timeout. Retry {attempt+1}/{self.max_retries}...")
                    time.sleep(2 ** attempt)
                    continue
                logger.error(f"Request timed out after {self.max_retries} attempts: {url}")
                return result

            except requests.ConnectionError as e:
                result.error_type = 'connection'
                result.error_message = f"Connection error: {str(e)}"
                if attempt < self.max_retries - 1:
                    logger.warning(f"Connection error. Retry {attempt+1}/{self.max_retries}...")
                    time.sleep(2 ** attempt)
                    continue
                logger.error(f"Connection failed after {self.max_retries} attempts: {url}")
                return result

            except requests.RequestException as e:
                result.error_type = 'unknown'
                result.error_message = f"Request failed: {str(e)}"
                if attempt < self.max_retries - 1:
                    logger.warning(f"Request error: {e}. Retry {attempt+1}/{self.max_retries}...")
                    time.sleep(2 ** attempt)
                    continue
                logger.error(f"Request failed after {self.max_retries} attempts: {e}")
                return result

        return result
    
    def get(self, endpoint: str, **kwargs) -> Optional[requests.Response]:
        """Make GET request - backward compatible, returns response or None."""
        result = self._make_request('GET', endpoint, **kwargs)
        return result.response if result.success else None

    def post(self, endpoint: str, **kwargs) -> Optional[requests.Response]:
        """Make POST request - backward compatible, returns response or None."""
        result = self._make_request('POST', endpoint, **kwargs)
        return result.response if result.success else None

    def get_with_error(self, endpoint: str, **kwargs) -> RequestResult:
        """Make GET request and return full RequestResult with error details."""
        return self._make_request('GET', endpoint, **kwargs)

    def post_with_error(self, endpoint: str, **kwargs) -> RequestResult:
        """Make POST request and return full RequestResult with error details."""
        return self._make_request('POST', endpoint, **kwargs)
    
    @abstractmethod
    def search(self, query: str, limit: int = 10, **kwargs) -> Any:
        """
        Search for papers (must be implemented by subclasses).
        
        Args:
            query: Search query
            limit: Maximum number of results
            **kwargs: Additional search parameters
            
        Returns:
            List of PaperMetadata objects
        """
        pass
    
    def close(self):
        """Close the session."""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
