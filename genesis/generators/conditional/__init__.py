"""Conditional generation API for synthetic data.

This package provides advanced conditional generation capabilities including:
- Condition-based filtering with operators (=, >, <, >=, <=, in, between)
- Efficient guided sampling to avoid rejection sampling overhead
- Upsampling for class imbalance correction
- Batch scenario generation
"""

from genesis.generators.conditional.conditions import (
    Condition,
    ConditionBuilder,
    ConditionSet,
    Operator,
)
from genesis.generators.conditional.samplers import (
    ConditionalSampler,
    GuidedConditionalSampler,
)
from genesis.generators.conditional.scenarios import (
    ScenarioGenerator,
    conditional_generate,
)
from genesis.generators.conditional.upsampling import Upsampler

__all__ = [
    # Conditions
    "Operator",
    "Condition",
    "ConditionSet",
    "ConditionBuilder",
    # Samplers
    "ConditionalSampler",
    "GuidedConditionalSampler",
    # Upsampling
    "Upsampler",
    # Scenarios
    "ScenarioGenerator",
    "conditional_generate",
]
