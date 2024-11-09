# Simplified implemntation of OPRO (Optimization by PROmpting)

## Overview

This project implements a simplified version of Google DeepMind's OPRO (Optimization by PROmpting) framework as given in LLM as optimizers paper, specifically adapted for optimizing prompts for computer science questions from the MMLU dataset. 

## Research Background

### Original OPRO Paper
The original paper "Large Language Models as Optimizers" (Google DeepMind, 2024) introduces OPRO as a novel approach to using LLMs for optimization tasks. Key aspects include:

1. **Natural Language Optimization**: OPRO enables optimization through natural language descriptions rather than formal specifications.
2. **Meta-Prompt Structure**: Uses previous solutions and their scores to guide the optimization process.
3. **Exploration-Exploitation Balance**: Manages the trade-off between exploring new solutions and exploiting known good solutions.


## Implementation Details

### Core Components

1. **Configuration (OptimizationConfig)**
```python
max_steps: int = 150          # Maximum optimization steps
solutions_per_step: int = 8   # Solutions generated per step
max_history: int = 20        # Max number of previous solutions to keep
temperature: float = 1.0     # Temperature for generation
token_weight: float = 0.3    # Weight for token length in scoring
max_tokens: int =         # Maximum tokens allowed in prompt (variable)
```

2. **Scoring Mechanism**
The implementation uses a weighted scoring formula:
```python
combined_score = (1 - token_weight) * accuracy + token_weight * token_score
where token_score = 1 - (token_count / max_tokens)
```

This balances:
- Solution accuracy (70% weight by default)
- Token efficiency (30% weight by default)

3. **Key Classes**
- `TokenManager`: Handles token counting and limits
- `MMluDataHandler`: Manages MMLU dataset operations
- `Scorer`: Evaluates solutions using OpenAI API
- `OptimizerEngine`: Core optimization logic

### Architecture Flow

1. **Data Preparation**
   - Load MMLU computer science questions
   - Split into train/test sets
   - Sample questions for evaluation

2. **Optimization Process**
   - Generate meta-prompt using previous solutions
   - Create new candidate solutions
   - Evaluate solutions for accuracy and token efficiency
   - Update optimization history
   - Repeat until convergence or max steps

3. **Solution Evaluation**
   - Calculate accuracy using OpenAI API / Llama model through Groq
   - Count tokens using tiktoken
   - Compute combined score
   - Track best solutions

## Setup and Usage

### Prerequisites
```bash
pip install openai pandas numpy tiktoken tqdm
```

### Environment Variables
```bash
export OPENAI_API_KEY='your-api-key'
```

### Data Format
MMLU CSV file should contain:
- question: Question text
- A, B, C, D: Multiple choice options
- answer: Correct answer (A, B, C, or D)

### Basic Usage
```python
# Initialize configuration
config = OptimizationConfig()

# Setup data handler
data_handler = MMluDataHandler("path_to_mmlu_cs_data.csv")
data_handler.prepare_data()

# Initialize optimizer
optimizer = OptimizerEngine(config)

# Run optimization
results = optimizer.optimize(data_handler, config.max_steps)
```

## Customization

### Adjusting Optimization Priorities
Modify `token_weight` in `OptimizationConfig`:
- Higher values (>0.3) prioritize token efficiency
- Lower values (<0.3) prioritize accuracy

### Optimization Parameters
- Adjust `temperature` for exploration/exploitation balance
- Modify `solutions_per_step` for optimization stability
- Change `max_history` for memory management

## Results and Output

The optimization process produces:
1. Best found instruction
2. Accuracy metrics
3. Token efficiency metrics
4. Combined performance scores

Results are saved in JSON format with timestamp:
```json
{
    "steps": [...],
    "best_solution": {
        "instruction": "...",
        "accuracy": 0.85,
        "token_count": 45,
        "combined_score": 0.78
    },
    "best_score": 0.78
}
```

## Limitations and Considerations

1. **API Costs**: Uses OpenAI API calls for evaluation
2. **Rate Limits**: Consider API rate limiting in optimization process


## Future Work

Potential improvements:
1. Support for multiple LLM providers
2. Advanced token optimization strategies
3. Multi-objective optimization approaches
4. Benchmarking and evaluation of this implementation
5. Adding tokenizer for llama model (tiktoken tokeniser isn't compatible with llama models, compatible only to GPT based models.)

## References

1. Google DeepMind (2024). "Large Language Models as Optimizers"

## Feedback

Feel free to drop your feedbacks at hjawajiwar@gmail.com


