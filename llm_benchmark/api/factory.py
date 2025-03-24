"""
Factory for creating API endpoint instances.
"""

import logging
from typing import Optional
from urllib.parse import urlparse

from llm_benchmark.api.base import BaseEndpoint
from llm_benchmark.api.openai import OpenAIEndpoint
from llm_benchmark.api.local import LocalEndpoint
from llm_benchmark.api.custom import CustomEndpoint


def create_endpoint(
    endpoint_url: str,
    model_name: str,
    api_key: Optional[str] = None,
    timeout: int = 120,
) -> BaseEndpoint:
    """
    Create an appropriate endpoint instance based on the URL.

    Args:
        endpoint_url: The URL of the API endpoint.
        model_name: The name of the model to use.
        api_key: Optional API key for authentication.
        timeout: Request timeout in seconds.

    Returns:
        An instance of a BaseEndpoint subclass.

    Raises:
        ValueError: If the endpoint URL is invalid or unsupported.
    """
    logger = logging.getLogger("llm_benchmark.api.factory")
    parsed_url = urlparse(endpoint_url)
    
    # OpenAI endpoint
    if "openai.com" in parsed_url.netloc:
        logger.debug(f"Creating OpenAI endpoint for {model_name}")
        return OpenAIEndpoint(endpoint_url, model_name, api_key, timeout)
    
    # Local endpoints
    if parsed_url.netloc in ("localhost", "127.0.0.1") or parsed_url.netloc.startswith("localhost:"):
        # Check for common local endpoints

        # Ollama typically runs on port 11434
        if "11434" in parsed_url.netloc or "ollama" in parsed_url.path:
            logger.debug(f"Creating Local endpoint (Ollama) for {model_name}")
            return LocalEndpoint(endpoint_url, model_name, api_key, timeout, provider="ollama")
        
        # vLLM doesn't have a standard port, but we can check path
        if "generate" in parsed_url.path or "vllm" in parsed_url.path:
            logger.debug(f"Creating Local endpoint (vLLM) for {model_name}")
            return LocalEndpoint(endpoint_url, model_name, api_key, timeout, provider="vllm")
        
        # Default to generic local endpoint
        logger.debug(f"Creating generic Local endpoint for {model_name}")
        return LocalEndpoint(endpoint_url, model_name, api_key, timeout)
    
    # For any other endpoint, use the custom implementation
    logger.debug(f"Creating Custom endpoint for {model_name} at {endpoint_url}")
    return CustomEndpoint(endpoint_url, model_name, api_key, timeout)
