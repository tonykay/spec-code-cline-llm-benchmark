#!/usr/bin/env python3
"""
Command-line interface test for LLM benchmarking utility.
This script tests the CLI without making actual API calls.
"""

import argparse
import sys
from unittest.mock import patch

# Patch sys.argv to simulate command-line arguments
test_args = [
    "llm_benchmark.py",
    "--endpoint", "https://api.example.com/v1",
    "--model", "test-model",
    "--prompt-file", "prompts.yaml",
    "--output", "text",
    "--verbose"
]

# Mock the API response to avoid actual API calls
mock_response = {
    "text": "This is a mock response from the API.",
    "input_tokens": 10,
    "output_tokens": 8,
    "cost_estimate": 0.0001
}

def main():
    """Run CLI test with mocked components."""
    print("Testing LLM Benchmark CLI...")
    
    # Capture the original argv
    original_argv = sys.argv.copy()
    
    try:
        # Override argv with test arguments
        sys.argv = test_args
        
        # Import the modules
        from llm_benchmark.cli.parser import parse_args
        
        # Test argument parsing
        args = parse_args()
        
        print(f"\n✅ CLI argument parsing works!")
        print(f"Parsed arguments: {args}")
        
        # Test that the required modules can be imported
        from llm_benchmark.cli.runner import run_benchmark
        from llm_benchmark.api.factory import create_endpoint
        from llm_benchmark.metrics.collector import MetricsCollector
        from llm_benchmark.utils.output import format_results
        
        print(f"\n✅ All modules imported successfully!")
        
        # Use patches to avoid actual API calls
        with patch('llm_benchmark.api.factory.create_endpoint') as mock_factory, \
             patch('llm_benchmark.cli.runner.load_prompts') as mock_load_prompts:
            
            # Configure mocks
            from llm_benchmark.api.base import BaseEndpoint
            # Create a mock endpoint class with required implementation
            class MockEndpoint(BaseEndpoint):
                def complete(self, prompt, token_limit=None, metrics=None):
                    return mock_response
                    
                def get_name(self):
                    return f"Mock {self.model_name}"
                    
                def count_tokens(self, text):
                    return len(text.split())
            
            # Instantiate with required parameters
            mock_endpoint = MockEndpoint(
                endpoint_url="https://api.example.com/v1",
                model_name="test-model",
                api_key="fake-key",
                timeout=10
            )
            mock_factory.return_value = mock_endpoint
            mock_load_prompts.return_value = [
                {"name": "Test prompt", "text": "This is a test prompt"}
            ]
            
            print("\n✅ Mock endpoint configured")
            
            # Test the main entry point (would make API calls in a real run)
            print("\n⚠️ Note: This is only testing CLI functionality, not making actual API calls")
            print("To run with a real API endpoint, use: python llm_benchmark.py --endpoint <URL> --model <MODEL>")
            
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during CLI test: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Restore original argv
        sys.argv = original_argv

if __name__ == "__main__":
    sys.exit(main())
