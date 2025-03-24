"""
Timer utility for benchmarking operations.
"""

import time
from typing import Optional


class Timer:
    """
    Context manager for timing operations.
    """

    def __init__(self):
        """Initialize timer with zero elapsed time."""
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: float = 0.0

    def __enter__(self):
        """Start timing when entering the context."""
        self.start_time = time.time()
        self.end_time = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing when exiting the context."""
        self.end_time = time.time()
        if self.start_time is not None:
            self.elapsed = self.end_time - self.start_time
        else:
            self.elapsed = 0.0

    def reset(self):
        """Reset the timer."""
        self.start_time = None
        self.end_time = None
        self.elapsed = 0.0

    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        self.end_time = None
        return self

    def stop(self):
        """Stop the timer and return elapsed time."""
        if self.start_time is None:
            raise ValueError("Timer was not started")
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        return self.elapsed
