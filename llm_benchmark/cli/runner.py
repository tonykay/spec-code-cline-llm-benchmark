"""
Benchmark runner implementation.
"""

import argparse
import logging
import sys
import time
import yaml
from typing import Dict, List, Optional

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from llm_benchmark.api.factory import create_endpoint
from llm_benchmark.metrics.collector import MetricsCollector
from llm_benchmark.utils.output import format_results
from llm_benchmark.utils.timer import Timer


def load_prompts(prompt_file: str) -> List[Dict]:
    """
    Load prompts from a YAML file.

    Args:
        prompt_file: Path to the YAML file containing prompts.

    Returns:
        List of prompt dictionaries.
    """
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict) or "prompts" not in data:
            raise ValueError(
                f"Invalid prompt file format. Expected a 'prompts' key with a list of prompts."
            )

        return data["prompts"]
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")


def run_benchmark(args: argparse.Namespace) -> None:
    """
    Run the benchmark with the provided arguments.

    Args:
        args: Command-line arguments.
    """
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("llm_benchmark")

    console = Console()
    console.print(
        f"[bold green]LLM Benchmark[/bold green]: {args.model} @ {args.endpoint}"
    )
    console.print("=" * 50)
    console.print()

    # Load prompts
    try:
        prompts = load_prompts(args.prompt_file)
        console.print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")
    except Exception as e:
        console.print(f"[bold red]Error loading prompts:[/bold red] {e}")
        sys.exit(1)

    # Create API endpoint
    try:
        endpoint = create_endpoint(
            endpoint_url=args.endpoint,
            model_name=args.model,
            api_key=args.api_key,
            timeout=args.timeout,
        )
        console.print(f"Using endpoint: {endpoint.get_name()}")
    except Exception as e:
        console.print(f"[bold red]Error creating endpoint:[/bold red] {e}")
        sys.exit(1)

    # Run benchmarks
    all_results = []
    for prompt_data in prompts:
        prompt_name = prompt_data.get("name", "Unnamed prompt")
        prompt_text = prompt_data.get("text")
        
        if not prompt_text:
            logger.warning(f"Skipping prompt '{prompt_name}' - missing text")
            continue
            
        # Get prompt-specific token limit or fall back to the global one
        token_limit = prompt_data.get("token_limit", args.token_limit)
        
        console.print(f"\n[bold]Running benchmark for:[/bold] {prompt_name}")
        
        # Run multiple times if requested
        prompt_results = []
        for run in tqdm(range(args.runs), desc="Runs"):
            try:
                metrics = MetricsCollector()
                
                # Start timing and send request
                with Timer() as timer:
                    response = endpoint.complete(
                        prompt=prompt_text,
                        token_limit=token_limit,
                        metrics=metrics,
                    )
                
                # Collect metrics
                total_time = timer.elapsed
                metrics.record_completion(response, total_time)
                
                result = metrics.get_metrics()
                result["run"] = run + 1
                prompt_results.append(result)
                
                if args.verbose:
                    logger.debug(f"Run {run+1} results: {result}")
                
            except Exception as e:
                console.print(f"[bold red]Error in run {run+1}:[/bold red] {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
        
        # Aggregate results from multiple runs
        if prompt_results:
            avg_result = {
                "prompt_name": prompt_name,
                "ttft_ms": sum(r["ttft_ms"] for r in prompt_results) / len(prompt_results),
                "total_time_s": sum(r["total_time_s"] for r in prompt_results) / len(prompt_results),
                "tokens_per_second": sum(r["tokens_per_second"] for r in prompt_results) / len(prompt_results),
                "input_tokens": prompt_results[0]["input_tokens"],  # Should be the same for all runs
                "output_tokens": sum(r["output_tokens"] for r in prompt_results) / len(prompt_results),
                "total_tokens": sum(r["total_tokens"] for r in prompt_results) / len(prompt_results),
                "cost_estimate": sum(r["cost_estimate"] for r in prompt_results) / len(prompt_results) if "cost_estimate" in prompt_results[0] else None,
                "runs": args.runs,
            }
            all_results.append(avg_result)
            
            # Display individual result
            format_results([avg_result], console, args.output, is_summary=False)
    
    # Display summary of all results
    if all_results:
        console.print("\n[bold]Summary Metrics (Average)[/bold]")
        console.print("-" * 50)
        
        # Calculate overall averages
        summary = {
            "prompt_name": "OVERALL AVERAGE",
            "ttft_ms": sum(r["ttft_ms"] for r in all_results) / len(all_results),
            "total_time_s": sum(r["total_time_s"] for r in all_results) / len(all_results),
            "tokens_per_second": sum(r["tokens_per_second"] for r in all_results) / len(all_results),
            "input_tokens": sum(r["input_tokens"] for r in all_results) / len(all_results),
            "output_tokens": sum(r["output_tokens"] for r in all_results) / len(all_results),
            "total_tokens": sum(r["total_tokens"] for r in all_results) / len(all_results),
            "cost_estimate": sum(r.get("cost_estimate", 0) for r in all_results) if all(r.get("cost_estimate") is not None for r in all_results) else None,
            "runs": sum(r["runs"] for r in all_results),
        }
        
        format_results([summary], console, args.output, is_summary=True)
    else:
        console.print("[bold yellow]No successful benchmark results to display[/bold yellow]")
