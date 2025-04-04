#!/usr/bin/env python3
"""
LLM Benchmarking Utility

A command-line tool to benchmark the performance of various LLM models
across different API endpoints.
"""

import sys
from llm_benchmark.cli.parser import parse_args
from llm_benchmark.cli.runner import run_benchmark


def main():
    """Main entry point for the LLM benchmarking utility."""
    try:
        args = parse_args()
        run_benchmark(args)
        return 0
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if getattr(args, "verbose", False):
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
