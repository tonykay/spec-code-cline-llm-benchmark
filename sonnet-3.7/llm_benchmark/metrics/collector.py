"""
Metrics collection and analysis module for LLM benchmarking.
"""

import time
from typing import Dict, Any, Optional


class MetricsCollector:
    """
    Collects and stores metrics for LLM benchmark runs.
    """

    def __init__(self):
        """Initialize the metrics collector."""
        self.metrics = {
            "ttft_ms": None,  # Time to First Token in milliseconds
            "total_time_s": None,  # Total generation time in seconds
            "tokens_per_second": None,  # Rate of token generation
            "input_tokens": None,  # Number of tokens in the input prompt
            "output_tokens": 0,  # Number of tokens in the generated response
            "total_tokens": None,  # Sum of input and output tokens
            "cost_estimate": None,  # Estimated cost of the API call
        }
        self.start_time = time.time()

    def record_ttft(self, ttft_ms: float) -> None:
        """
        Record the time to first token.

        Args:
            ttft_ms: Time to first token in milliseconds.
        """
        self.metrics["ttft_ms"] = ttft_ms

    def record_input_tokens(self, count: int) -> None:
        """
        Record the number of input tokens.

        Args:
            count: Number of input tokens.
        """
        self.metrics["input_tokens"] = count

    def update_output_tokens(self, count: int) -> None:
        """
        Update the output token count.

        Args:
            count: Current number of output tokens.
        """
        self.metrics["output_tokens"] = count

    def record_cost(self, cost: float) -> None:
        """
        Record the estimated cost of the API call.

        Args:
            cost: Estimated cost in USD.
        """
        self.metrics["cost_estimate"] = cost

    def record_completion(self, response: Dict[str, Any], total_time: float) -> None:
        """
        Record metrics from a completed response.

        Args:
            response: The response dictionary from the API.
            total_time: Total time taken for the response in seconds.
        """
        self.metrics["total_time_s"] = total_time
        
        # In case output_tokens wasn't streamed, get it from the response
        if "output_tokens" in response:
            self.metrics["output_tokens"] = response["output_tokens"]
            
        # Update cost if available
        if "cost_estimate" in response and response["cost_estimate"] is not None:
            self.metrics["cost_estimate"] = response["cost_estimate"]
            
        # Calculate derived metrics
        if self.metrics["input_tokens"] is not None:
            self.metrics["total_tokens"] = self.metrics["input_tokens"] + self.metrics["output_tokens"]
            
        if total_time > 0 and self.metrics["output_tokens"] > 0:
            self.metrics["tokens_per_second"] = self.metrics["output_tokens"] / total_time

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get the collected metrics.

        Returns:
            A dictionary of all collected metrics.
        """
        # Ensure values are properly formatted
        formatted_metrics = {}
        for key, value in self.metrics.items():
            if key in ["ttft_ms", "total_time_s", "tokens_per_second", "cost_estimate"] and value is not None:
                formatted_metrics[key] = round(value, 2)
            else:
                formatted_metrics[key] = value
                
        return formatted_metrics
