#!/usr/bin/env python3
"""
Test script for the LLM benchmarking utility.
This validates that the components can be imported properly.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all necessary modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        # Basic imports
        import yaml
        import requests
        import tqdm
        import rich
        
        logger.info("Base dependencies imported successfully")
        
        # Core package imports
        from llm_benchmark.cli.parser import parse_args
        from llm_benchmark.api.base import BaseEndpoint
        from llm_benchmark.metrics.collector import MetricsCollector
        from llm_benchmark.utils.timer import Timer
        from llm_benchmark.utils.output import format_results
        
        logger.info("LLM benchmark modules imported successfully")
        
        # Try to import optional dependencies
        try:
            import tiktoken
            logger.info("Optional dependency tiktoken imported successfully")
        except ImportError:
            logger.warning("Optional dependency tiktoken not found, OpenAI token counting will be affected")
            
        return True
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return False

def main():
    """Main test function."""
    logger.info("Starting LLM benchmark test...")
    
    if test_imports():
        logger.info("All imports successful!")
        print("\n✅ LLM benchmark package imports validated successfully\n")
        return 0
    else:
        logger.error("Import tests failed")
        print("\n❌ LLM benchmark package validation failed\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
