from setuptools import setup, find_packages

setup(
    name="llm_benchmark",
    version="0.1.0",
    description="A utility for benchmarking LLM performance",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "requests>=2.28.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "rich>=13.0.0",
        "tiktoken>=0.5.0",
    ],
    entry_points={
        "console_scripts": [
            "llm-benchmark=llm_benchmark.py:main",
        ],
    },
    python_requires=">=3.12",
)
