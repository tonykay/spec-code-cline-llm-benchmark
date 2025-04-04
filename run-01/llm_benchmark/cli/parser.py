"""
Command-line argument parser for the LLM benchmark utility.
"""

import argparse
import os
from typing import Any


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the LLM benchmark utility.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark the performance of LLM models across different API endpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--endpoint",
        required=True,
        help="URL of the API endpoint to benchmark",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Name of the model to use (e.g., gpt-4o, llama3)",
    )

    parser.add_argument(
        "--prompt-file",
        default="prompts.yaml",
        help="Path to YAML file containing prompts",
    )

    parser.add_argument(
        "--token-limit",
        type=int,
        default=None,
        help="Maximum number of tokens in the response",
    )

    parser.add_argument(
        "--output",
        choices=["text", "json", "csv"],
        default="text",
        help="Output format for benchmark results",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds",
    )

    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY", None),
        help="API key for services requiring authentication (default: from env)",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of times to run each prompt for averaging",
    )

    return parser.parse_args()
