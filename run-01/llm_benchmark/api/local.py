"""
Local API endpoint implementation for benchmarking local LLM servers.
"""

import json
import logging
import time
import re
from typing import Dict, Optional, Any, List

import requests

from llm_benchmark.api.base import BaseEndpoint
from llm_benchmark.metrics.collector import MetricsCollector


class LocalEndpoint(BaseEndpoint):
    """
    Implementation of BaseEndpoint for local API servers (e.g., Ollama, vLLM).
    """

    def __init__(
        self,
        endpoint_url: str,
        model_name: str,
        api_key: Optional[str] = None,
        timeout: int = 120,
        provider: str = "generic",
    ):
        """
        Initialize the local endpoint.

        Args:
            endpoint_url: The URL of the local API endpoint.
            model_name: The name of the model to use.
            api_key: Optional API key (usually not required for local endpoints).
            timeout: Request timeout in seconds.
            provider: The provider name (e.g., "ollama", "vllm", "generic").
        """
        super().__init__(endpoint_url, model_name, api_key, timeout)
        self.logger = logging.getLogger("llm_benchmark.api.local")
        self.provider = provider.lower()

        # Ensure endpoint URL is properly formatted based on provider
        if self.provider == "ollama":
            # For Ollama, the API endpoint should be /api/generate
            # Make sure we don't have extra paths like /v1 that might cause issues
            if "api/generate" not in endpoint_url:
                # Strip any existing path like /v1
                base_url = endpoint_url.rstrip("/")
                if "/v1" in base_url:
                    base_url = base_url.split("/v1")[0]
                
                # Append the correct path
                if base_url.endswith("/"):
                    self.endpoint_url = f"{base_url}api/generate"
                else:
                    self.endpoint_url = f"{base_url}/api/generate"
            else:
                self.endpoint_url = endpoint_url
        
        self.logger.debug(f"Initialized {self.provider} endpoint at {self.endpoint_url}")

    def get_name(self) -> str:
        """
        Get a descriptive name for this endpoint.

        Returns:
            A string describing the endpoint and model.
        """
        return f"Local {self.provider.capitalize()} ({self.model_name})"

    def complete(
        self,
        prompt: str,
        token_limit: Optional[int] = None,
        metrics: Optional[MetricsCollector] = None,
    ) -> Dict[str, Any]:
        """
        Send a completion request to the local API.

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
        # Record input tokens (estimated)
        self.input_tokens = self.count_tokens(prompt)
        if metrics:
            metrics.record_input_tokens(self.input_tokens)

        # Create request based on provider
        if self.provider == "ollama":
            return self._complete_ollama(prompt, token_limit, metrics)
        elif self.provider == "vllm":
            return self._complete_vllm(prompt, token_limit, metrics)
        else:
            return self._complete_generic(prompt, token_limit, metrics)

    def _complete_ollama(
        self,
        prompt: str,
        token_limit: Optional[int] = None,
        metrics: Optional[MetricsCollector] = None,
    ) -> Dict[str, Any]:
        """
        Send a completion request to an Ollama endpoint.
        """
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
        }
        
        if token_limit is not None:
            data["max_length"] = token_limit
        
        self.logger.debug(f"Sending request to Ollama: {self.endpoint_url}")
        
        try:
            response = requests.post(
                self.endpoint_url,
                headers=headers,
                json=data,
                timeout=self.timeout,
                stream=True,
            )
            response.raise_for_status()
            
            # Process streaming response
            full_text = ""
            first_token_received = False
            start_time = time.time()
            
            for line in response.iter_lines():
                if not line:
                    continue
                
                try:
                    chunk = json.loads(line)
                    
                    # Get token from response
                    if "response" in chunk:
                        token_text = chunk["response"]
                        
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
                    
                    # Check for done
                    if chunk.get("done", False):
                        break
                        
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to parse JSON from line: {line}")
            
            # Calculate final token count
            output_tokens = self.count_tokens(full_text)
            
            return {
                "text": full_text,
                "input_tokens": self.input_tokens,
                "output_tokens": output_tokens,
                "cost_estimate": None  # Local models don't have cost
            }
            
        except requests.RequestException as e:
            self.logger.error(f"Request to Ollama API failed: {e}")
            raise

    def _complete_vllm(
        self,
        prompt: str,
        token_limit: Optional[int] = None,
        metrics: Optional[MetricsCollector] = None,
    ) -> Dict[str, Any]:
        """
        Send a completion request to a vLLM endpoint.
        """
        headers = {"Content-Type": "application/json"}
        
        # vLLM can use OpenAI-compatible endpoint
        if "/v1" in self.endpoint_url:
            # OpenAI-compatible endpoint
            endpoint = f"{self.endpoint_url}/chat/completions"
            data = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            }
            if token_limit is not None:
                data["max_tokens"] = token_limit
        else:
            # Native vLLM endpoint
            endpoint = self.endpoint_url
            data = {
                "prompt": prompt,
                "model": self.model_name,
                "stream": True,
            }
            if token_limit is not None:
                data["max_tokens"] = token_limit
        
        self.logger.debug(f"Sending request to vLLM: {endpoint}")
        
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=data,
                timeout=self.timeout,
                stream=True,
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
                    chunk = json.loads(line)
                    
                    # Extract text based on response format
                    token_text = ""
                    if "choices" in chunk:
                        choices = chunk["choices"]
                        if choices and "delta" in choices[0]:
                            delta = choices[0]["delta"]
                            token_text = delta.get("content", "")
                        elif choices and "text" in choices[0]:
                            token_text = choices[0]["text"]
                    elif "text" in chunk:
                        token_text = chunk["text"]
                    
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
            
            return {
                "text": full_text,
                "input_tokens": self.input_tokens,
                "output_tokens": output_tokens,
                "cost_estimate": None  # Local models don't have cost
            }
            
        except requests.RequestException as e:
            self.logger.error(f"Request to vLLM API failed: {e}")
            raise

    def _complete_generic(
        self,
        prompt: str,
        token_limit: Optional[int] = None,
        metrics: Optional[MetricsCollector] = None,
    ) -> Dict[str, Any]:
        """
        Send a completion request to a generic LLM endpoint.
        This tries to handle various API formats.
        """
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Try to guess appropriate format
        # Start with OpenAI-compatible format as it's common
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
        }
        
        if token_limit is not None:
            data["max_tokens"] = token_limit
        
        self.logger.debug(f"Sending request to generic endpoint: {self.endpoint_url}")
        
        try:
            response = requests.post(
                self.endpoint_url,
                headers=headers,
                json=data,
                timeout=self.timeout,
                stream=True,
            )
            
            # If we get a 4xx error, try other formats
            if 400 <= response.status_code < 500:
                self.logger.debug("Initial request format failed, trying other formats")
                # Try chat format
                data = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
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
                "input_tokens": self.input_tokens,
                "output_tokens": output_tokens,
                "cost_estimate": None  # Generic models don't have cost
            }
            
        except requests.RequestException as e:
            self.logger.error(f"Request to generic API failed: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.
        For local models, this is an approximation.

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
