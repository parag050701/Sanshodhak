"""Base client with rate limiting, retries, and error handling."""
import time
import logging
import requests
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


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
    ) -> Optional[requests.Response]:
        """
        Make HTTP request with retries and error handling.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (relative to base_url)
            params: URL parameters
            data: Form data
            json_data: JSON body
            headers: Additional headers
            
        Returns:
            Response object or None if all retries failed
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
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
                    allow_redirects=True
                )
                
                # Handle rate limiting (429)
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited. Waiting {retry_after}s...")
                    time.sleep(retry_after)
                    continue
                
                # Success
                if response.status_code < 400:
                    return response
                
                # Client error (4xx) - don't retry
                if 400 <= response.status_code < 500:
                    logger.warning(f"Client error {response.status_code}: {url}")
                    return None
                
                # Server error (5xx) - retry
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Server error {response.status_code}. Retry {attempt+1}/{self.max_retries} in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                return None
                
            except requests.Timeout:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Timeout. Retry {attempt+1}/{self.max_retries}...")
                    time.sleep(2 ** attempt)
                    continue
                logger.error(f"Request timed out after {self.max_retries} attempts: {url}")
                return None
                
            except requests.RequestException as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Request error: {e}. Retry {attempt+1}/{self.max_retries}...")
                    time.sleep(2 ** attempt)
                    continue
                logger.error(f"Request failed after {self.max_retries} attempts: {e}")
                return None
        
        return None
    
    def get(self, endpoint: str, **kwargs) -> Optional[requests.Response]:
        """Make GET request."""
        return self._make_request('GET', endpoint, **kwargs)
    
    def post(self, endpoint: str, **kwargs) -> Optional[requests.Response]:
        """Make POST request."""
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
