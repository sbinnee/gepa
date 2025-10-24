from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TokenUsage:
    """Tracks token usage for a single LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def aggregate(self) -> int:
        """Return total tokens used."""
        return self.total_tokens

    def __str__(self) -> str:
        return f"TokenUsage(prompt={self.prompt_tokens}, completion={self.completion_tokens}, total={self.total_tokens})"


@dataclass
class CostMetrics:
    """Tracks cost and token metrics for LLM calls."""
    total_cost: float = 0.0  # in USD
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0
    per_call_costs: list[float] = field(default_factory=list)

    def add_call(self, cost: float, usage: TokenUsage):
        """Add a single LLM call's metrics."""
        self.total_cost += cost
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens
        self.total_tokens += usage.total_tokens
        self.call_count += 1
        self.per_call_costs.append(cost)

    def merge(self, other: "CostMetrics") -> "CostMetrics":
        """Merge two CostMetrics objects."""
        merged = CostMetrics(
            total_cost=self.total_cost + other.total_cost,
            total_prompt_tokens=self.total_prompt_tokens + other.total_prompt_tokens,
            total_completion_tokens=self.total_completion_tokens + other.total_completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            call_count=self.call_count + other.call_count,
            per_call_costs=self.per_call_costs + other.per_call_costs,
        )
        return merged

    def __str__(self) -> str:
        return (f"CostMetrics(cost=${self.total_cost:.4f}, "
                f"tokens={self.total_tokens}, "
                f"prompt={self.total_prompt_tokens}, "
                f"completion={self.total_completion_tokens}, "
                f"calls={self.call_count})")

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "total_cost": self.total_cost,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
            "per_call_costs": self.per_call_costs,
        }


class CostTracker:
    """Tracks LLM costs across different phases of optimization."""

    def __init__(self):
        # Cost tracking by component/phase
        self.task_lm_costs = CostMetrics()  # Costs from adapter.evaluate() calls
        self.reflection_lm_costs = CostMetrics()  # Costs from reflection_lm calls
        self.merge_lm_costs = CostMetrics()  # Costs from merge proposer if applicable
        self.total_costs = CostMetrics()  # Aggregate of all costs
        self.model_name = None  # Track the model name for cost calculation fallback

    def extract_cost_from_litellm_response(self, response) -> float:
        """Extract cost from a litellm response object."""
        try:
            # Try _hidden_params first (for regular completion calls)
            if hasattr(response, "_hidden_params"):
                hidden_params = response._hidden_params
                if isinstance(hidden_params, dict):
                    cost = hidden_params.get("response_cost", 0.0)
                    if cost is not None:
                        return float(cost)

            # If _hidden_params doesn't work, try cost directly
            if hasattr(response, "cost"):
                return float(response.cost) if response.cost is not None else 0.0
        except (AttributeError, TypeError, KeyError, ValueError):
            pass
        return 0.0

    def extract_usage_from_litellm_response(self, response) -> TokenUsage:
        """Extract token usage from a litellm response object."""
        usage = TokenUsage()
        try:
            if hasattr(response, "usage"):
                usage_obj = response.usage
                if hasattr(usage_obj, "prompt_tokens"):
                    usage.prompt_tokens = usage_obj.prompt_tokens
                if hasattr(usage_obj, "completion_tokens"):
                    usage.completion_tokens = usage_obj.completion_tokens
                if hasattr(usage_obj, "total_tokens"):
                    usage.total_tokens = usage_obj.total_tokens
            # Fallback: calculate total_tokens if not present
            if usage.total_tokens == 0 and (usage.prompt_tokens > 0 or usage.completion_tokens > 0):
                usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        except (AttributeError, TypeError):
            pass
        return usage

    def _fallback_cost_from_usage(self, usage: TokenUsage, model: str = None) -> float:
        """Calculate cost from token usage using litellm's pricing if cost not available."""
        try:
            import litellm
            # Try to use litellm's cost calculation
            if hasattr(litellm, "completion_cost"):
                # This calculates cost based on token usage
                cost = litellm.completion_cost(
                    model=model or self.model_name,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens
                )
                return float(cost) if cost is not None else 0.0
        except Exception:
            pass
        return 0.0

    def add_task_lm_call(self, response, source: str = "adapter", model: str = None):
        """Track a task LM call (from adapter.evaluate())."""
        cost = self.extract_cost_from_litellm_response(response)
        usage = self.extract_usage_from_litellm_response(response)

        # If cost is 0 but we have token usage, try to calculate it
        if cost == 0.0 and (usage.prompt_tokens > 0 or usage.completion_tokens > 0):
            cost = self._fallback_cost_from_usage(usage, model)

        self.task_lm_costs.add_call(cost, usage)
        self.total_costs.add_call(cost, usage)

    def add_reflection_lm_call(self, response, model: str = None):
        """Track a reflection LM call."""
        cost = self.extract_cost_from_litellm_response(response)
        usage = self.extract_usage_from_litellm_response(response)

        # If cost is 0 but we have token usage, try to calculate it
        if cost == 0.0 and (usage.prompt_tokens > 0 or usage.completion_tokens > 0):
            cost = self._fallback_cost_from_usage(usage, model)

        self.reflection_lm_costs.add_call(cost, usage)
        self.total_costs.add_call(cost, usage)

    def add_merge_lm_call(self, response):
        """Track a merge LM call if applicable."""
        cost = self.extract_cost_from_litellm_response(response)
        usage = self.extract_usage_from_litellm_response(response)
        self.merge_lm_costs.add_call(cost, usage)
        self.total_costs.add_call(cost, usage)

    def add_cost_metrics(self, component: str, cost: float, usage: TokenUsage):
        """Manually add cost metrics for a component."""
        if component == "task_lm":
            self.task_lm_costs.add_call(cost, usage)
        elif component == "reflection_lm":
            self.reflection_lm_costs.add_call(cost, usage)
        elif component == "merge_lm":
            self.merge_lm_costs.add_call(cost, usage)
        self.total_costs.add_call(cost, usage)

    def get_summary(self) -> dict:
        """Get a summary of all costs."""
        return {
            "task_lm": self.task_lm_costs.to_dict(),
            "reflection_lm": self.reflection_lm_costs.to_dict(),
            "merge_lm": self.merge_lm_costs.to_dict(),
            "total": self.total_costs.to_dict(),
        }

    def log_summary(self, logger) -> None:
        """Log cost summary using the provided logger."""
        logger.log("\n" + "=" * 80)
        logger.log("LLM COST SUMMARY")
        logger.log("=" * 80)
        logger.log(f"Task LM (adapter evaluations):   {self.task_lm_costs}")
        logger.log(f"Reflection LM (proposals):       {self.reflection_lm_costs}")
        if self.merge_lm_costs.call_count > 0:
            logger.log(f"Merge LM:                       {self.merge_lm_costs}")
        logger.log(f"Total:                          {self.total_costs}")
        logger.log("=" * 80 + "\n")

    def __str__(self) -> str:
        return (f"CostTracker(task_lm={self.task_lm_costs}, "
                f"reflection_lm={self.reflection_lm_costs}, "
                f"total={self.total_costs})")
