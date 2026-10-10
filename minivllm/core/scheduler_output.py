"""What the scheduler asks the runner to compute in one step."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SchedulerOutput:
    """Work for one step.

    num_tokens maps seq_id -> number of tokens to compute this step.
    An empty dict means the scheduler has nothing to run.
    """

    num_tokens: dict[int, int] = field(default_factory=dict)
