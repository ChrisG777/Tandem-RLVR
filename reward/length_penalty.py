"""Token-cost adapter for the existing verl pilot correctness reward."""
import math
from typing import Any

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager


class LengthPenaltyRewardManager(NaiveRewardManager):
    """Apply R = correctness - coefficient * generated_tokens / response_budget.

    Reuse verl's decoding and the configured correctness verifier. Count actual
    unpadded response tokens, including EOS and the answer, never prompt tokens.
    Keep accuracy/format diagnostics unchanged and log the two reward components.
    The coefficient must be in [0, .25), preserving RuleTaker's .25 correctness
    increments across all allowed lengths. Invalid configuration or lengths fail.
    """

    def __init__(self, config: Any, tokenizer: Any, compute_score: Any, **kwargs: Any):
        self.coefficient = float(config.reward.length_penalty)
        self.budget = int(config.data.max_response_length)
        if not math.isfinite(self.coefficient) or not 0 <= self.coefficient < 0.25:
            raise ValueError("Length coefficient must be finite and in [0, .25)")
        if self.budget <= 0:
            raise ValueError("Response budget must be positive")
        super().__init__(config, tokenizer, compute_score, **kwargs)

    async def run_single(self, data: DataProto) -> dict[str, Any]:
        """Score one single-turn rollout; padding is excluded by attention_mask."""
        if len(data) != 1:
            raise ValueError("Length pilot requires exactly one single-turn response")
        item = data[0]
        width = item.batch["responses"].shape[-1]
        if width == 0:
            raise ValueError("Empty response tensor")
        length = int(item.batch["attention_mask"][-width:].sum().item())
        if not 0 < length <= self.budget:
            raise ValueError(f"Invalid response length {length} for budget {self.budget}")
        result = await super().run_single(data)
        correctness = float(result["reward_score"])
        penalty = self.coefficient * length / self.budget
        result["reward_score"] = correctness - penalty
        result["reward_extra_info"].update(
            score=result["reward_score"], correctness_reward=correctness,
            length_penalty=penalty, response_tokens=length,
        )
        return result
