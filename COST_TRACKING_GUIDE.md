# LLM Cost and Token Tracking Guide

This guide explains how cost and token tracking works in GEPA.

## Overview

GEPA now automatically tracks LLM costs and token usage for all litellm calls made during optimization. The tracking covers:

1. **Task LM costs**: Costs from `adapter.evaluate()` calls (via `litellm.batch_completion()`)
2. **Reflection LM costs**: Costs from reflection model calls used for generating improved prompts
3. **Total costs**: Aggregate of all LLM costs across the optimization run

## How It Works

### Cost Tracking Components

1. **`CostTracker`** (`gepa/logging/cost_tracker.py`): Main class that tracks costs
   - Maintains separate cost metrics for task LM, reflection LM, and merge LM
   - Extracts cost and token usage from litellm response objects
   - Provides summary and logging functionality

2. **`TokenUsage`** and **`CostMetrics`** dataclasses: Store detailed metrics
   - `TokenUsage`: prompt tokens, completion tokens, total tokens
   - `CostMetrics`: aggregated costs, token counts, per-call costs

3. **`GEPAState.cost_tracker`**: Stores the CostTracker instance during optimization
4. **`GEPAResult.cost_summary`**: Contains cost summary in the final result

### Integration Points

**API Entry Point** (`api.py:optimize()`):
- Creates a `CostTracker` instance at the start
- Passes it to `DefaultAdapter` for task LM cost tracking
- Wraps the `reflection_lm` to track reflection costs
- Passes it to `GEPAEngine` for state management
- Logs the cost summary at the end of optimization

**Default Adapter** (`adapters/default_adapter/default_adapter.py`):
- Accepts optional `cost_tracker` parameter
- When `litellm.batch_completion()` is called, extracts cost from each response
- Calls `cost_tracker.add_task_lm_call()` for each response

**Reflection LM Wrapper** (`api.py`):
- Wraps the reflection_lm to capture responses
- Calls `cost_tracker.add_reflection_lm_call()` for tracking

## Usage

The cost tracking is automatic and requires no explicit action from users. However, you can access the costs in several ways:

### Option 1: After optimize() Returns

```python
from gepa import optimize

result = optimize(
    seed_candidate={"system_prompt": "You are a helpful assistant."},
    trainset=train_data,
    valset=val_data,
    task_lm="gpt-4",
    reflection_lm="gpt-4",
    max_metric_calls=100,
)

# Access cost summary from result
if result.cost_summary:
    print("Task LM Costs:")
    print(f"  Total Cost: ${result.cost_summary['task_lm']['total_cost']:.4f}")
    print(f"  Total Tokens: {result.cost_summary['task_lm']['total_tokens']}")
    print(f"  Calls: {result.cost_summary['task_lm']['call_count']}")

    print("\nReflection LM Costs:")
    print(f"  Total Cost: ${result.cost_summary['reflection_lm']['total_cost']:.4f}")
    print(f"  Total Tokens: {result.cost_summary['reflection_lm']['total_tokens']}")

    print("\nTotal Costs:")
    print(f"  Total Cost: ${result.cost_summary['total']['total_cost']:.4f}")
    print(f"  Total Tokens: {result.cost_summary['total']['total_tokens']}")
```

### Option 2: From Saved Result JSON

When using `run_dir`, the result is saved to `_gepa_result.json`:

```python
import json

with open("run_dir/_gepa_result.json", "r") as f:
    result_data = json.load(f)

cost_summary = result_data.get("cost_summary", {})
print(f"Total optimization cost: ${cost_summary['total']['total_cost']:.4f}")
```

### Option 3: Custom Logger

You can provide your own logger to capture cost logs:

```python
from gepa.logging.logger import StdOutLogger

class CostTrackingLogger(StdOutLogger):
    def log(self, message: str) -> None:
        # Custom logging logic
        print(f"[COST] {message}")

result = optimize(
    seed_candidate={"system_prompt": "..."},
    trainset=train_data,
    valset=val_data,
    task_lm="gpt-4",
    reflection_lm="gpt-4",
    max_metric_calls=100,
    logger=CostTrackingLogger(),  # Cost summary will be logged here
)
```

## Cost Data Structure

The `cost_summary` in the result has the following structure:

```python
{
    "task_lm": {
        "total_cost": 0.0,           # USD
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "call_count": 0,
        "per_call_costs": [0.0, ...] # List of individual call costs
    },
    "reflection_lm": {
        # Same structure as task_lm
    },
    "merge_lm": {
        # Same structure as task_lm (may be empty if no merges)
    },
    "total": {
        # Same structure as task_lm
    }
}
```

## How Costs Are Calculated

litellm automatically calculates costs using its pricing database based on:
- The model used
- Number of prompt tokens
- Number of completion tokens

The cost is extracted from the response via: `response._hidden_params["response_cost"]`

## Advanced Usage

### Custom Adapter with Cost Tracking

If you implement a custom adapter, pass the `cost_tracker` to it:

```python
from gepa import optimize
from gepa.adapters import GEPAAdapter
from gepa.logging.cost_tracker import CostTracker

class CustomAdapter(GEPAAdapter):
    def __init__(self, model, cost_tracker=None):
        self.model = model
        self.cost_tracker = cost_tracker

    def evaluate(self, batch, candidate, capture_traces=False):
        # Your evaluation logic
        # If using litellm:
        response = litellm.completion(model=self.model, messages=...)
        if self.cost_tracker:
            self.cost_tracker.add_task_lm_call(response)
        # ... rest of logic

# Pass your adapter to optimize()
result = optimize(
    seed_candidate={"system_prompt": "..."},
    trainset=train_data,
    valset=val_data,
    adapter=CustomAdapter(model="gpt-4"),
    reflection_lm="gpt-4",
    max_metric_calls=100,
)
```

### Manual Cost Tracking

You can also manually add costs if not using litellm:

```python
from gepa.logging.cost_tracker import CostTracker, TokenUsage

tracker = CostTracker()
usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
tracker.add_cost_metrics("task_lm", cost=0.05, usage=usage)

print(tracker.get_summary())
```

## Limitations

1. **Cost Calculation**: Costs depend on litellm's pricing database. Ensure you have the latest version of litellm.
2. **Custom Models**: If using custom models without litellm pricing, costs may be 0.0.
3. **Batch Operations**: Costs from `litellm.batch_completion()` are tracked per-call (per item in the batch).

## Troubleshooting

**Q: Why are costs showing as 0.0?**
A: The model might not be in litellm's pricing database, or the response object doesn't have cost information. Check that you're using a model that litellm supports.

**Q: How do I track costs for non-litellm models?**
A: Implement your own cost calculation and use `cost_tracker.add_cost_metrics()` manually.

**Q: Can I disable cost tracking?**
A: Cost tracking is lightweight and always enabled. The cost_tracker object is created but can be ignored in the result.

## Implementation Details

### Files Modified

1. **`gepa/logging/cost_tracker.py`** (new): Core cost tracking implementation
2. **`gepa/core/state.py`**: Added `cost_tracker` field to `GEPAState`
3. **`gepa/core/result.py`**: Added `cost_summary` field to `GEPAResult`
4. **`gepa/core/engine.py`**: Added `cost_tracker` parameter and state attachment
5. **`gepa/adapters/default_adapter/default_adapter.py`**: Added cost tracking in `evaluate()`
6. **`gepa/api.py`**: Added cost_tracker creation, reflection_lm wrapping, and cost logging

### How Cost Extraction Works

When litellm makes an API call, it stores the cost in `response._hidden_params["response_cost"]`. The `CostTracker` extracts this value using:

```python
def extract_cost_from_litellm_response(self, response) -> float:
    if hasattr(response, "_hidden_params"):
        cost = response._hidden_params.get("response_cost", 0.0)
        return float(cost) if cost is not None else 0.0
    return 0.0
```

Similarly, token usage is extracted from `response.usage.{prompt_tokens, completion_tokens, total_tokens}`.
