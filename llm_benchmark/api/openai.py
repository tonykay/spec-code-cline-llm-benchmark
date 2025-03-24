"""
OpenAI API endpoint implementation.
"""

import json
import logging
import time
from typing import Dict, Optional, Any, List, Iterator

import requests
import tiktoken

from llm_benchmark.api.base import BaseEndpoint
from llm_benchmark.metrics.collector import MetricsCollector


class OpenAIEndpoint(BaseEndpoint):
    """
    Implementation of BaseEndpoint for the OpenAI API.
    """

    def __init__(
        self,
        endpoint_url: str,
        model_name: str,
        api_key: Optional[str] = None,
        timeout: int = 120,
    ):
        """
        Initialize the OpenAI endpoint.

        Args:
            endpoint_url: The URL of the OpenAI API endpoint.
            model_name: The name of the model to use (e.g., gpt-4o).
            api_key: The OpenAI API key for authentication.
            timeout: Request timeout in seconds.
        """
        super().__init__(endpoint_url, model_name, api_key, timeout)
        self.logger = logging.getLogger("llm_benchmark.api.openai")

        if not api_key:
            self.logger.warning(
                "No API key provided for OpenAI API. Requests may fail."
            )

        # Ensure endpoint URL ends with '/v1' for OpenAI API
        if not endpoint_url.endswith("/v1"):
            if endpoint_url.endswith("/"):
                self.endpoint_url = f"{endpoint_url}v1"
            else:
                self.endpoint_url = f"{endpoint_url}/v1"

        # Initialize tokenizer based on model
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self.logger.warning(
                f"No specific tokenizer found for {model_name}, using cl100k_base instead"
            )
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def get_name(self) -> str:
        """
        Get a descriptive name for this endpoint.

        Returns:
            A string describing the endpoint and model.
        """
        return f"OpenAI {self.model_name}"

    def complete(
        self,
        prompt: str,
        token_limit: Optional[int] = None,
        metrics: Optional[MetricsCollector] = None,
    ) -> Dict[str, Any]:
        """
        Send a completion request to the OpenAI API.

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
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # Configure API endpoint based on the model
        if self.model_name.startswith(("gpt-3.5", "gpt-4")):
            endpoint = f"{self.endpoint_url}/chat/completions"
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,  # Always stream to measure TTFT
            }
        else:
            # Fallback to completions for other models
            endpoint = f"{self.endpoint_url}/completions"
            data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": True,  # Always stream to measure TTFT
            }

        # Add token limit if provided
        if token_limit is not None:
            if token_limit <= 0:
                raise ValueError("Token limit must be positive")
            data["max_tokens"] = token_limit

        self.logger.debug(f"Sending request to {endpoint}")
        
        # Count input tokens
        input_tokens = self.count_tokens(prompt)
        if metrics:
            metrics.record_input_tokens(input_tokens)

        # Send streaming request
        try:
            response = requests.post(
                endpoint, 
                headers=headers, 
                json=data, 
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()
            
            # Process streaming response
            full_text = ""
            first_token_received = False
            start_time = time.time()
            
            for line in response.iter_lines():
                if not line:
                    continue
                    
                # Check if this is SSE data
                if line.startswith(b"data: "):
                    line = line[6:]  # Remove 'data: ' prefix
                    
                    # Check for the end of the stream
                    if line.strip() == b"[DONE]":
                        break
                    
                    try:
                        json_data = json.loads(line)
                        
                        # Get the token from the appropriate field based on endpoint
                        if "chat/completions" in endpoint:
                            choices = json_data.get("choices", [])
                            if choices and "delta" in choices[0]:
                                delta = choices[0]["delta"]
                                token_text = delta.get("content", "")
                            else:
                                continue
                        else:
                            choices = json_data.get("choices", [])
                            if choices and "text" in choices[0]:
                                token_text = choices[0]["text"]
                            else:
                                continue
                        
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
                            
                    except json.JSONDecodeError:
                        self.logger.warning(f"Failed to parse JSON from line: {line}")
            
            # Calculate final token count
            output_tokens = self.count_tokens(full_text)
            
            # Record cost estimate if we have a model that supports it
            cost_estimate = self._calculate_cost_estimate(input_tokens, output_tokens)
            if metrics and cost_estimate:
                metrics.record_cost(cost_estimate)
            
            return {
                "text": full_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_estimate": cost_estimate
            }
            
        except requests.RequestException as e:
            self.logger.error(f"Request to OpenAI API failed: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.

        Args:
            text: The text to count tokens for.

        Returns:
            The number of tokens in the text.
        """
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def _calculate_cost_estimate(self, input_tokens: int, output_tokens: int) -> Optional[float]:
        """
        Calculate the estimated cost for the API call based on token usage.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Estimated cost in USD or None if the model isn't recognized.
        """
        # Pricing as of April 2024 - may need updating over time
        prices = {
            "gpt-4o": {"input": 5.0, "output": 15.0},  # $5/$15 per million tokens
            "gpt-4": {"input": 30.0, "output": 60.0},  # $30/$60 per million tokens
            "gpt-4-turbo": {"input": 10.0, "output": 30.0},  # $10/$30 per million tokens
            "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},  # $0.50/$1.50 per million tokens
        }
        
        # Find matching price
        price = None
        for model_prefix, price_data in prices.items():
            if self.model_name.startswith(model_prefix):
                price = price_data
                break
        
        if not price:
            return None
            
        # Calculate cost (price is per million tokens)
        input_cost = (input_tokens / 1_000_000) * price["input"]
        output_cost = (output_tokens / 1_000_000) * price["output"]
        total_cost = input_cost + output_cost
        
        return round(total_cost, 6)  # Return cost with 6 decimal places
