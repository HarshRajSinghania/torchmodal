"""
torchmodal.nn
~~~~~~~~~~~~~

Neural network modules for differentiable modal logic.

This subpackage provides ``nn.Module`` implementations for:

- **Operators**: Differentiable smooth aggregations (SmoothMin, SmoothMax, ConvPool)
- **Connectives**: Propositional logic (AND, OR, NOT, IMPLIES)
- **Modal**: Core modal neurons (Necessity □, Possibility ♢)
- **Accessibility**: Kripke accessibility relations (Fixed, Learnable, Metric,
  Attention)
"""

from torchmodal.nn.accessibility import (
    AttentionAccessibility,
    FixedAccessibility,
    LearnableAccessibility,
    MetricAccessibility,
    top_k_mask,
)
from torchmodal.nn.connectives import (
    Conjunction,
    Disjunction,
    Implication,
    Negation,
)
from torchmodal.nn.modal import Necessity, Possibility
from torchmodal.nn.operators import ConvPool, SmoothMax, SmoothMin, Softmax, Softmin

__all__ = [
    # Aggregation operators
    "SmoothMin",
    "SmoothMax",
    "ConvPool",
    # Legacy aliases
    "Softmin",
    "Softmax",
    # Propositional connectives
    "Negation",
    "Conjunction",
    "Disjunction",
    "Implication",
    # Modal operators
    "Necessity",
    "Possibility",
    # Accessibility relations
    "FixedAccessibility",
    "LearnableAccessibility",
    "MetricAccessibility",
    "AttentionAccessibility",
    "top_k_mask",
]
