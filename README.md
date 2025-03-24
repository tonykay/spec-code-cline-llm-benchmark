# LLM Benchmark Utility

A Python command-line utility for benchmarking the performance of Large Language Models (LLMs) across different API endpoints.

## Features

- Benchmark different LLM providers (OpenAI, local endpoints like Ollama/vLLM, custom endpoints)
- Measure key performance metrics including:
  - Time to First Token (TTFT)
  - Total generation time
  - Tokens per second (TPS)
  - Token counts (input, output, total)
  - Cost estimates (where applicable)
- Support for batch testing with multiple prompts
- Multiple output formats (text, JSON, CSV)
- Easily extensible to support new LLM providers

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-benchmark.git
cd llm-benchmark

# Create a virtual environment (recommended)
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

The basic command structure is:

```bash
python llm_benchmark.py --endpoint <API_ENDPOINT> --model <MODEL_NAME> [options]
```

### Required Arguments

- `--endpoint`: URL of the API endpoint to benchmark
- `--model`: Name of the model to use (e.g., `gpt-4o`, `llama3`)

### Optional Arguments

- `--prompt-file`: Path to YAML file containing prompts (default: `prompts.yaml`)
- `--token-limit`: Maximum number of tokens in the response
- `--output`: Output format (`text`, `json`, `csv`) (default: `text`)
- `--verbose`: Enable verbose logging
- `--timeout`: Request timeout in seconds (default: 120)
- `--api-key`: API key for services that require authentication
- `--runs`: Number of times to run each prompt for averaging (default: 1)

## Examples

### OpenAI API

```bash
# Using OpenAI's API with GPT-4o
python llm_benchmark.py --endpoint https://api.openai.com/v1 --model gpt-4o --api-key $OPENAI_API_KEY

# Run each prompt 3 times for statistical significance
python llm_benchmark.py --endpoint https://api.openai.com/v1 --model gpt-3.5-turbo --runs 3
```

### Local Models

```bash
# Using Ollama with llama3
python llm_benchmark.py --endpoint http://localhost:11434/api/generate --model llama3

# Using vLLM with mistral-7b and token limit
python llm_benchmark.py --endpoint http://localhost:8000/v1 --model mistral-7b --token-limit 2048
```

### Custom Output Formats

```bash
# Output results as JSON
python llm_benchmark.py --endpoint https://api.openai.com/v1 --model gpt-4 --output json

# Output results as CSV
python llm_benchmark.py --endpoint https://api.openai.com/v1 --model gpt-4 --output csv
```

## Prompt File Format

The prompt file should follow this YAML structure:

```yaml
prompts:
  - name: "Simple question"
    text: "What is the capital of France?"
    token_limit: 100  # Optional per-prompt override

  - name: "Code generation"
    text: "Write a Python function to calculate Fibonacci numbers"
    token_limit: 500  # Optional per-prompt override
```

## Output Metrics

The tool measures and reports the following metrics:

1. **Time to First Token (TTFT)**: Time from request submission until the first token is received
2. **Total Generation Time**: Total time taken to generate the complete response
3. **Tokens Per Second (TPS)**: Rate of token generation (output tokens / generation time)
4. **Input Token Count**: Number of tokens in the input prompt
5. **Output Token Count**: Number of tokens in the generated response
6. **Total Tokens**: Sum of input and output tokens
7. **Cost Estimate** (if applicable): Estimated cost of the API call based on token usage and known pricing

## Extending

To add support for a new LLM provider:

1. Create a new endpoint class in `llm_benchmark/api/` that extends the `BaseEndpoint` abstract class
2. Implement the required methods (`complete`, `get_name`, `count_tokens`)
3. Add detection logic in the `create_endpoint` factory function

## Limitations

- Token counting for non-OpenAI models is approximate
- Cost estimation is only available for OpenAI models with known pricing
- Some provider-specific features may not be supported

## License

MIT
