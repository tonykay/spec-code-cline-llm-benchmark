# LLM Benchmarking Utility Specification

## Project Overview

Create a Python 3.12 command-line utility to benchmark the performance of various Large Language Models (LLMs) across different API endpoints. The tool should load prompts from a file, send them to specified model endpoints, and measure key performance metrics such as latency, throughput, and token generation speed.

## Core Requirements

1. Support for multiple API endpoints including OpenAI API, local endpoints (vLLM, Ollama), and custom REST endpoints
2. Ability to read prompts from a YAML file with optional token limits per prompt
3. Collection and reporting of key performance metrics
4. Command-line interface with appropriate arguments
5. Proper error handling and reporting
6. Support for batched benchmarking of multiple prompts
7. Clear, tabular output format for results

## Technical Requirements

- **Language**: Python 3.12+
- **Dependencies**: Required libraries should be minimal and well-maintained (requests, pyyaml, tqdm, rich for output formatting)
- **Architecture**: Modular design with separation of concerns (CLI parsing, API handling, metric collection, output formatting)

## Command-Line Arguments

The utility should support the following command-line arguments:

| Argument | Description | Required | Default |
|----------|-------------|----------|---------|
| `--endpoint` | URL of the API endpoint to benchmark | Yes | N/A |
| `--model` | Name of the model to use (e.g., `gpt-4o`, `llama3`) | Yes | N/A |
| `--prompt-file` | Path to YAML file containing prompts | No | `prompts.yaml` |
| `--token-limit` | Maximum number of tokens in the response | No | None (use model default) |
| `--output` | Output format (text, json, csv) | No | `text` |
| `--verbose` | Enable verbose logging | No | False |
| `--timeout` | Request timeout in seconds | No | 120 |
| `--api-key` | API key for services that require authentication | No | Read from environment |
| `--runs` | Number of times to run each prompt for averaging | No | 1 |

## Input Format (YAML)

The prompt file should follow this YAML structure:

```yaml
prompts:
  - name: "Simple question"
    text: "What is the capital of France?"
    token_limit: 100  # Optional per-prompt override

  - name: "Code generation"
    text: "Write a Python function to calculate Fibonacci numbers"
    token_limit: 500  # Optional per-prompt override
    
  # Additional prompts...
```

## Performance Metrics

The tool should measure and report the following metrics for each prompt:

1. **Time to First Token (TTFT)**: Time from request submission until the first token is received
2. **Total Generation Time**: Total time taken to generate the complete response
3. **Tokens Per Second (TPS)**: Rate of token generation (output tokens / generation time)
4. **Input Token Count**: Number of tokens in the input prompt
5. **Output Token Count**: Number of tokens in the generated response
6. **Total Tokens**: Sum of input and output tokens
7. **Cost Estimate** (if applicable): Estimated cost of the API call based on token usage and known pricing

## Implementation Details

### API Integration

Create a modular system for API endpoints with these core components:

1. **Base Endpoint Class**: Abstract class defining the interface for all endpoints
2. **OpenAI Endpoint**: Support for OpenAI API (gpt-3.5-turbo, gpt-4, gpt-4o, etc.)
3. **Local Endpoint**: Support for local API servers like vLLM, Ollama (with model specification)
4. **Custom Endpoint**: Support for custom REST endpoints with appropriate headers and authentication

### Token Counting

Implement token counting using the appropriate tokenizer for each model:
- OpenAI: Use `tiktoken` library
- Other models: Use model-specific tokenizers or estimate based on model information

### Stream Processing

Implement streaming response processing to accurately measure TTFT and TPS metrics:
- Track timestamp when first token arrives
- Count tokens as they arrive
- Calculate rolling TPS during generation

## Output Format

The tool should output results in a clear, tabular format for each prompt and an aggregated summary:

```
Benchmark Results for gpt-4o @ api.openai.com
=============================================

Prompt: "Simple question"
- Time to First Token: 150ms
- Total Generation Time: 1.2s
- Tokens Per Second: 42
- Input Tokens: 8
- Output Tokens: 50
- Total Tokens: 58
- Cost Estimate: $0.0029

[Additional prompts...]

Summary Metrics (Average)
------------------------
- Avg Time to First Token: 180ms
- Avg Tokens Per Second: 38.5
- Avg Response Time: 1.5s
- Total Cost: $0.0145
```

## Error Handling

The utility should handle various error conditions gracefully:

1. **Connection errors**: Report timeouts, DNS failures, etc.
2. **Authentication errors**: Clear message for API key issues
3. **Rate limiting**: Detect rate limits and provide clear error message
4. **Model unavailability**: Handle cases where the requested model is unavailable
5. **Malformed prompts**: Validate prompts before sending

## Example Usage

```bash
# Basic usage with OpenAI
python llm_benchmark.py --endpoint https://api.openai.com/v1 --model gpt-4o --api-key $OPENAI_API_KEY

# Local Ollama instance
python llm_benchmark.py --endpoint http://localhost:11434/api/generate --model llama3 --prompt-file custom_prompts.yaml

# Custom vLLM deployment with token limit
python llm_benchmark.py --endpoint http://localhost:8000/v1 --model mistral-7b --token-limit 2048 --verbose

# Multiple runs for statistical significance
python llm_benchmark.py --endpoint https://api.openai.com/v1 --model gpt-3.5-turbo --runs 5 --output json
```

## Additional Considerations

1. **Concurrent Benchmarking**: Optional support for running multiple prompts concurrently (with appropriate rate limiting)
2. **Visualization**: Consider adding simple visualization of results (bar charts, line graphs)
3. **Export**: Allow exporting results to CSV/JSON for further analysis
4. **Configuration File**: Support loading configuration from a file for complex benchmark setups

## Documentation

Generated code should include:
1. Docstrings for all functions and classes
2. README with installation and usage instructions
3. Example prompts file
4. Requirements.txt file