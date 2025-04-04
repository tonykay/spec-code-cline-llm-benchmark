"""
Custom API endpoint implementation.
"""

import json
import logging
import time
import re
from typing import Dict, Optional, Any, List

import requests

from llm_benchmark.api.base import BaseEndpoint
from llm_benchmark.metrics.collector import MetricsCollector


class CustomEndpoint(BaseEndpoint):
    """
    Implementation of BaseEndpoint for custom API endpoints.
    This is a flexible implementation that tries to adapt to different API formats.
    """

    def __init__(
        self,
        endpoint_url: str,
        model_name: str,
        api_key: Optional[str] = None,
        timeout: int = 120,
    ):
        """
        Initialize the custom endpoint.

        Args:
            endpoint_url: The URL of the API endpoint.
            model_name: The name of the model to use.
            api_key: Optional API key for authentication.
            timeout: Request timeout in seconds.
        """
        super().__init__(endpoint_url, model_name, api_key, timeout)
        self.logger = logging.getLogger("llm_benchmark.api.custom")

    def get_name(self) -> str:
        """
        Get a descriptive name for this endpoint.

        Returns:
            A string describing the endpoint and model.
        """
        return f"Custom API ({self.model_name} @ {self.endpoint_url})"

    def complete(
        self,
        prompt: str,
        token_limit: Optional[int] = None,
        metrics: Optional[MetricsCollector] = None,
    ) -> Dict[str, Any]:
        """
        Send a completion request to the custom API.

        Args:
            prompt: The prompt to send.
            token_limit: Maximum number of tokens to generate.
            metrics: Optional metrics collector to record streaming metrics.

        Returns:
            A dictionary containing the response data.

        Raises:
            ValueError: If the model name is invalid.
            requests.RequestException: If the API request fails.
        """
        # Count input tokens (estimated)
        input_tokens = self.count_tokens(prompt)
        if metrics:
            metrics.record_input_tokens(input_tokens)

        # Try different API formats
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            # Try common auth header formats
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Try OpenAI-compatible format first
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }

        if token_limit is not None:
            data["max_tokens"] = token_limit

        self.logger.debug(f"Sending request to custom endpoint: {self.endpoint_url}")

        try:
            response = requests.post(
                self.endpoint_url,
                headers=headers,
                json=data,
                timeout=self.timeout,
                stream=True,
            )

            # If OpenAI-compatible chat format fails, try raw completion format
            if 400 <= response.status_code < 500:
                self.logger.debug("Chat format failed, trying completion format")
                data = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": True,
                }
                if token_limit is not None:
                    data["max_tokens"] = token_limit

                response = requests.post(
                    self.endpoint_url,
                    headers=headers,
                    json=data,
                    timeout=self.timeout,
                    stream=True,
                )

            # If both formats fail, try with a different auth format
            if 400 <= response.status_code < 500:
                self.logger.debug("Trying different auth header format")
                headers["Authorization"] = f"Key {self.api_key}"
                response = requests.post(
                    self.endpoint_url,
                    headers=headers,
                    json=data,
                    timeout=self.timeout,
                    stream=True,
                )

            response.raise_for_status()

            # Process streaming response - handle various formats
            full_text = ""
            first_token_received = False
            start_time = time.time()

            for line in response.iter_lines():
                if not line:
                    continue

                # Handle SSE format
                if line.startswith(b"data: "):
                    line = line[6:]  # Remove 'data: ' prefix
                    if line.strip() == b"[DONE]":
                        break

                try:
                    # Try to parse as JSON
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        # If not JSON, treat as raw text
                        token_text = line.decode("utf-8")
                        chunk = None

                    # Extract text from JSON if available
                    token_text = ""
                    if chunk:
                        if "choices" in chunk:
                            choices = chunk["choices"]
                            if choices:
                                if "delta" in choices[0]:
                                    delta = choices[0]["delta"]
                                    token_text = delta.get("content", "")
                                elif "text" in choices[0]:
                                    token_text = choices[0]["text"]
                        elif "response" in chunk:
                            token_text = chunk["response"]
                        elif "text" in chunk:
                            token_text = chunk["text"]
                        elif "content" in chunk:
                            token_text = chunk["content"]
                        elif "generated_text" in chunk:
                            token_text = chunk["generated_text"]

                    # Record first token time
                    if not first_token_received and token_text.strip():
                        first_token_received = True
                        ttft = (time.time() - start_time) * 1000  # ms
                        if metrics:
                            metrics.record_ttft(ttft)

                    # Append to full text
                    full_text += token_text

                    # Update token count in real-time
                    if metrics:
                        current_tokens = self.count_tokens(full_text)
                        metrics.update_output_tokens(current_tokens)

                except Exception as e:
                    self.logger.warning(f"Error processing response chunk: {e}")

            # Calculate final token count
            output_tokens = self.count_tokens(full_text)

            return {
                "text": full_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_estimate": None  # Custom models don't have default cost estimates
            }

        except requests.RequestException as e:
            self.logger.error(f"Request to custom API failed: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.
        For custom endpoints, this is an approximation.

        Args:
            text: The text to count tokens for.

        Returns:
            The estimated number of tokens in the text.
        """
        if not text:
            return 0

        # Simple estimation based on word count and punctuation
        # This is a rough approximation - real tokenizers are model-specific
        words = re.findall(r'\w+', text)
        punctuation = re.findall(r'[^\w\s]', text)

        # Roughly estimate tokens - words + punctuation + some overhead for splits
        token_estimate = len(words) + len(punctuation) + (len(words) // 6)
        return token_estimate
