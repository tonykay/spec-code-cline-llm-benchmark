"""
Base class for API endpoints.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any


class BaseEndpoint(ABC):
    """
    Abstract base class for all API endpoints.
    """

    def __init__(
        self,
        endpoint_url: str,
        model_name: str,
        api_key: Optional[str] = None,
        timeout: int = 120,
    ):
        """
        Initialize the endpoint.

        Args:
            endpoint_url: The URL of the API endpoint.
            model_name: The name of the model to use.
            api_key: Optional API key for authentication.
            timeout: Request timeout in seconds.
        """
        self.endpoint_url = endpoint_url
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout

    @abstractmethod
    def complete(
        self,
        prompt: str,
        token_limit: Optional[int] = None,
        metrics: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Send a completion request to the API.

        Args:
            prompt: The prompt to send.
            token_limit: Optional maximum number of tokens to generate.
            metrics: Optional metrics collector to record streaming metrics.

        Returns:
            A dictionary containing the response data.
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get a descriptive name for this endpoint.

        Returns:
            A string describing the endpoint and model.
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.

        Args:
            text: The text to count tokens for.

        Returns:
            The number of tokens in the text.
        """
        pass
