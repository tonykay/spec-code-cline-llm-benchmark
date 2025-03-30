"""
Output formatting utilities for benchmark results.
"""

import csv
import json
import os
import sys
from typing import Dict, List, Any, Optional, TextIO

from rich.console import Console
from rich.table import Table


def format_results(results: List[Dict[str, Any]], console: Console, format_type: str, is_summary: bool = False) -> None:
    """
    Format and display benchmark results.
    
    Args:
        results: List of benchmark result dictionaries.
        console: Rich console instance for output.
        format_type: Output format type ('text', 'json', or 'csv').
        is_summary: Whether this is a summary result.
    """
    if format_type == "text":
        _format_text(results, console, is_summary)
    elif format_type == "json":
        _format_json(results, console, is_summary)
    elif format_type == "csv":
        _format_csv(results, console, is_summary)
    else:
        console.print(f"[bold red]Unknown output format: {format_type}[/bold red]")
        

def _format_text(results: List[Dict[str, Any]], console: Console, is_summary: bool = False) -> None:
    """
    Format and display results as rich text table.
    
    Args:
        results: List of benchmark result dictionaries.
        console: Rich console instance for output.
        is_summary: Whether this is a summary result.
    """
    if not results:
        return
        
    table = Table(
        title="Summary Metrics" if is_summary else "Benchmark Results",
        show_header=True,
        header_style="bold",
    )
    
    # Add columns
    table.add_column("Metric", style="cyan")
    for result in results:
        prompt_name = result.get("prompt_name", "Unnamed")
        table.add_column(prompt_name, style="green" if is_summary else None)
    
    # Define metrics to display and their formatting
    metrics_to_display = [
        ("Time to First Token", "ttft_ms", "{} ms"),
        ("Total Generation Time", "total_time_s", "{} s"),
        ("Tokens Per Second", "tokens_per_second", "{}"),
        ("Input Tokens", "input_tokens", "{}"),
        ("Output Tokens", "output_tokens", "{}"),
        ("Total Tokens", "total_tokens", "{}"),
    ]
    
    # Add cost if available
    if any(r.get("cost_estimate") is not None for r in results):
        metrics_to_display.append(("Cost Estimate", "cost_estimate", "${:.4f}"))
    
    # Add rows
    for display_name, key, format_str in metrics_to_display:
        row = [display_name]
        for result in results:
            value = result.get(key)
            if value is None:
                formatted = "N/A"
            elif isinstance(value, float) and ":.4f" in format_str:
                formatted = format_str.format(value)
            else:
                formatted = format_str.format(value)
            row.append(formatted)
        table.add_row(*row)
    
    # Add number of runs if available and not summary
    if not is_summary and any("runs" in r for r in results):
        row = ["Number of Runs"]
        for result in results:
            row.append(str(result.get("runs", 1)))
        table.add_row(*row)
    
    console.print(table)


def _format_json(results: List[Dict[str, Any]], console: Console, is_summary: bool = False) -> None:
    """
    Format and display results as JSON.
    
    Args:
        results: List of benchmark result dictionaries.
        console: Rich console instance for output.
        is_summary: Whether this is a summary result.
    """
    output = {
        "type": "summary" if is_summary else "detail",
        "results": results,
    }
    
    console.print(json.dumps(output, indent=2))


def _format_csv(results: List[Dict[str, Any]], console: Console, is_summary: bool = False) -> None:
    """
    Format and display results as CSV.
    
    Args:
        results: List of benchmark result dictionaries.
        console: Rich console instance for output.
        is_summary: Whether this is a summary result.
    """
    if not results:
        return
        
    # Prepare field names (header row)
    # Start with standard metrics that should be in all results
    fieldnames = [
        "prompt_name", 
        "ttft_ms", 
        "total_time_s", 
        "tokens_per_second", 
        "input_tokens", 
        "output_tokens", 
        "total_tokens"
    ]
    
    # Add cost if available
    if any(r.get("cost_estimate") is not None for r in results):
        fieldnames.append("cost_estimate")
    
    # Add runs if available
    if any("runs" in r for r in results):
        fieldnames.append("runs")
    
    # Create CSV output
    output = []
    
    # Add header row
    output.append(",".join(fieldnames))
    
    # Add data rows
    for result in results:
        row = []
        for field in fieldnames:
            value = result.get(field, "")
            # Format the value for CSV
            if isinstance(value, str):
                # Escape quotes and wrap in quotes if contains comma
                if '"' in value or ',' in value:
                    # Double up any quotes and wrap the whole thing in quotes
                    value = '"' + value.replace('"', '""') + '"'
            elif isinstance(value, float):
                value = f"{value:.4f}" if field == "cost_estimate" else f"{value:.2f}"
            else:
                value = str(value)
            row.append(value)
        output.append(",".join(row))
    
    console.print("\n".join(output))


def save_results_to_file(
    results: List[Dict[str, Any]], 
    filepath: str, 
    format_type: str = "json",
    is_summary: bool = False
) -> None:
    """
    Save benchmark results to a file.
    
    Args:
        results: List of benchmark result dictionaries.
        filepath: Path to the output file.
        format_type: Output format type ('json' or 'csv').
        is_summary: Whether this is a summary result.
    """
    if not results:
        return
        
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    
    with open(filepath, "w", encoding="utf-8") as f:
        if format_type == "json":
            _save_json(results, f, is_summary)
        elif format_type == "csv":
            _save_csv(results, f)
        else:
            raise ValueError(f"Unsupported file format: {format_type}")


def _save_json(results: List[Dict[str, Any]], file: TextIO, is_summary: bool = False) -> None:
    """
    Save results as JSON.
    
    Args:
        results: List of benchmark result dictionaries.
        file: File object to write to.
        is_summary: Whether this is a summary result.
    """
    output = {
        "type": "summary" if is_summary else "detail",
        "results": results,
    }
    
    json.dump(output, file, indent=2)


def _save_csv(results: List[Dict[str, Any]], file: TextIO) -> None:
    """
    Save results as CSV.
    
    Args:
        results: List of benchmark result dictionaries.
        file: File object to write to.
    """
    if not results:
        return
        
    # Determine all possible fields from all results
    fields = set()
    for result in results:
        fields.update(result.keys())
    
    # Sort fields for consistent output
    fieldnames = sorted(fields)
    
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)
