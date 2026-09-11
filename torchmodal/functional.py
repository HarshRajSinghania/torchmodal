"""
torchmodal.functional
~~~~~~~~~~~~~~~~~~~~~

Functional API for differentiable modal logic operators.

Provides stateless functions for soft logic aggregations, propositional
connectives, and modal operators following the MLNN framework
(Sulc, 2026) with Kripke semantics.

All functions operate on tensors of truth bounds in [0, 1].

.. note::
   The aggregation operators are named ``smooth_min`` / ``smooth_max``
   (not ``softmin`` / ``softmax``) to avoid confusion with the standard
   probability-normalization ``torch.softmax``.  These operators are
   *log-sum-exp* aggregations that serve as sound *bounds* on the true
   min/max, not probability distributions.  Legacy aliases ``softmin``
   and ``softmax`` are provided for backward compatibility.
"""

from __future__ import annotations

import warnings

import torch
from torch import Tensor

__all__ = [
    # Differentiable aggregations
    "smooth_min",
    "smooth_max",
    "conv_pool",
    # Legacy aliases (deprecated)
    "softmin",
    "softmax",
    # Propositional connectives
    "negation",
    "conjunction",
    "disjunction",
    "implication",
    # Modal operators
    "necessity",
    "possibility",
    "until",
    # Contradiction
    "contradiction",
]

# ---------------------------------------------------------------------------
# Differentiable aggregations (Section 3.2.1 of the paper)
#
# Named smooth_min / smooth_max to avoid collision with the standard
# torch.softmax (probability normalization), which is used internally
# by conv_pool.  These are log-sum-exp aggregations providing sound
# bounds on true min/max.
# ---------------------------------------------------------------------------


def smooth_min(x: Tensor, tau: float = 0.1, dim: int = -1) -> Tensor:
    r"""Differentiable smooth minimum (log-sum-exp lower bound).

    .. math::
        \operatorname{smooth\_min}_\tau(\mathbf{x}) =
            -\tau \log \sum_i \exp(-x_i / \tau)

    This is a *sound lower bound* on :func:`torch.min`:
    ``smooth_min(x) <= min(x)`` for all ``x_i \in [0, 1]``.

    As :math:`\tau \to 0`, converges to :func:`torch.min`.

    .. note::
       Not to be confused with ``torch.softmax`` (probability normalization).
       This function computes a *scalar aggregation* via the log-sum-exp
       identity, not a probability distribution.

    Args:
        x: Input tensor of truth values in [0, 1].
        tau: Temperature controlling approximation sharpness. Default 0.1.
        dim: Dimension along which to aggregate. Default -1.

    Returns:
        Tensor with ``dim`` reduced.
    """
    return -tau * torch.logsumexp(-x / tau, dim=dim)


def smooth_max(x: Tensor, tau: float = 0.1, dim: int = -1) -> Tensor:
    r"""Differentiable smooth maximum (log-sum-exp upper bound).

    .. math::
        \operatorname{smooth\_max}_\tau(\mathbf{x}) =
            \tau \log \sum_i \exp(x_i / \tau)

    This is a *sound upper bound* on :func:`torch.max`:
    ``smooth_max(x) >= max(x)`` for all ``x_i \in [0, 1]``.

    As :math:`\tau \to 0`, converges to :func:`torch.max`.

    .. note::
       Not to be confused with ``torch.softmax`` (probability normalization).
       This function computes a *scalar aggregation* via the log-sum-exp
       identity, not a probability distribution.

    Args:
        x: Input tensor of truth values in [0, 1].
        tau: Temperature controlling approximation sharpness. Default 0.1.
        dim: Dimension along which to aggregate. Default -1.

    Returns:
        Tensor with ``dim`` reduced.
    """
    return tau * torch.logsumexp(x / tau, dim=dim)


# Legacy aliases --------------------------------------------------------

def softmin(x: Tensor, tau: float = 0.1, dim: int = -1) -> Tensor:
    """Deprecated alias for :func:`smooth_min`."""
    warnings.warn(
        "torchmodal.functional.softmin is deprecated, "
        "use smooth_min to avoid confusion with torch.softmax",
        DeprecationWarning,
        stacklevel=2,
    )
    return smooth_min(x, tau=tau, dim=dim)


def softmax(x: Tensor, tau: float = 0.1, dim: int = -1) -> Tensor:
    """Deprecated alias for :func:`smooth_max`."""
    warnings.warn(
        "torchmodal.functional.softmax is deprecated, "
        "use smooth_max to avoid confusion with torch.softmax",
        DeprecationWarning,
        stacklevel=2,
    )
    return smooth_max(x, tau=tau, dim=dim)


def conv_pool(
    x: Tensor, z: Tensor, tau: float = 0.1, dim: int = -1
) -> Tensor:
    r"""Convex pooling operator (attention-weighted average).

    Computes a convex combination of ``x`` using attention weights
    derived from ``z``:

    .. math::
        \operatorname{conv\_pool}_\tau(\mathbf{x}, \mathbf{z}) =
            \sum_i w_i\, x_i, \quad
            w_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}

    The weights ``w`` are a standard probability-normalized softmax
    (``torch.softmax``) applied to the *logits* ``z / tau``.

    **Bound properties** (for ``x_i \in [0, 1]``):

    - ``z = x``  → the largest values receive the highest weight,
      providing a differentiable *lower bound* on ``max(x)``.
    - ``z = -x`` → the smallest values receive the highest weight,
      providing a differentiable *upper bound* on ``min(x)``.

    These two modes are used in the Necessity (□) and Possibility (♢)
    operators to construct *sound* upper/lower bounds that complement
    the ``smooth_min`` / ``smooth_max`` bounds.

    Args:
        x: Values to pool, shape ``(..., N)``.
        z: Logits controlling the convex weights, same shape as ``x``.
            Use ``z = x`` for a lower bound on max, ``z = -x`` for an
            upper bound on min.
        tau: Temperature. Lower values sharpen the weighting toward the
            extreme element. Default 0.1.
        dim: Dimension along which to pool. Default -1.

    Returns:
        Tensor with ``dim`` reduced.
    """
    weights = torch.softmax(z / tau, dim=dim)
    return (weights * x).sum(dim=dim)


# ---------------------------------------------------------------------------
# Propositional connectives (weighted, real-valued logic)
# ---------------------------------------------------------------------------


def negation(x: Tensor) -> Tensor:
    r"""Fuzzy negation: :math:`\neg x = 1 - x`.

    Args:
        x: Truth values in [0, 1]. Can be bounds ``(L, U)`` — apply to each.

    Returns:
        Negated truth values.
    """
    return 1.0 - x


def conjunction(a: Tensor, b: Tensor) -> Tensor:
    r"""Łukasiewicz conjunction (fuzzy AND).

    .. math::
        a \wedge b = \max(0,\; a + b - 1)

    For bounds: ``L_{a∧b} = max(0, L_a + L_b - 1)``,
    ``U_{a∧b} = min(U_a, U_b)``.

    Args:
        a: First operand truth values in [0, 1].
        b: Second operand truth values in [0, 1].

    Returns:
        Conjunction truth values.
    """
    return torch.clamp(a + b - 1.0, min=0.0)


def disjunction(a: Tensor, b: Tensor) -> Tensor:
    r"""Łukasiewicz disjunction (fuzzy OR).

    .. math::
        a \vee b = \min(1,\; a + b)

    Args:
        a: First operand truth values in [0, 1].
        b: Second operand truth values in [0, 1].

    Returns:
        Disjunction truth values.
    """
    return torch.clamp(a + b, max=1.0)


def implication(a: Tensor, b: Tensor) -> Tensor:
    r"""Łukasiewicz implication.

    .. math::
        a \to b = \min(1,\; 1 - a + b)

    Equivalent to ``disjunction(negation(a), b)``.

    Args:
        a: Antecedent truth values in [0, 1].
        b: Consequent truth values in [0, 1].

    Returns:
        Implication truth values.
    """
    return torch.clamp(1.0 - a + b, max=1.0)


# ---------------------------------------------------------------------------
# Modal operators (Section 3.2.1)
# ---------------------------------------------------------------------------


def _select_terms(x: Tensor, top_k: int | None, largest: bool) -> Tensor:
    """Keep the ``top_k`` extreme aggregation terms of each row of ``x``.

    Top-k neighbourhoods must be selected on the quantity that is actually
    aggregated — ``(1 - A) + L`` for □, ``A + U - 1`` for ♢ — and per
    endpoint, never on ``A`` alone: selecting by ``A`` can drop the world
    whose ``L`` (or ``U``) carries the extremum, and the bound then over-
    (or under-) reports it, violating Theorem 1. Selecting on the terms
    themselves keeps the true extremum in the kept set, so the masked
    ``min`` / ``max`` is exact and the smooth aggregations stay within
    ``tau * log(top_k)`` of it. Only the kept terms are aggregated, so the
    result — and its gradient, which reaches exactly the selected entries
    of ``A`` — is independent of ``|W|``.

    Args:
        x: Aggregation terms, shape ``(|W|, |W|)`` (rows = source worlds).
        top_k: Number of terms to keep per row, or ``None`` for all of them.
            A ``top_k >= |W|`` also keeps all of them.
        largest: ``False`` keeps the smallest terms (□), ``True`` the largest
            (♢).

    Returns:
        ``x`` itself when nothing is dropped, else ``(|W|, top_k)``.
    """
    if top_k is None:
        return x
    if top_k < 1:
        raise ValueError(f"top_k must be a positive integer or None, got {top_k}")
    if top_k >= x.shape[1]:
        return x
    return torch.topk(x, top_k, dim=1, largest=largest).values


def necessity(
    prop_bounds: Tensor,
    accessibility: Tensor,
    tau: float = 0.1,
    top_k: int | None = None,
) -> Tensor:
    r"""Necessity (Box / □) operator — differentiable Kripke semantics.

    Computes truth bounds for □ϕ across all worlds using the weighted
    accessibility matrix. For each world *w*:

    .. math::
        L_{\Box\phi,w} = \operatorname{smooth\_min}_\tau \bigl\{
            (1 - \tilde{A}_{w,w'}) + L_{\phi,w'} \bigr\}_{w' \in W}

    .. math::
        U_{\Box\phi,w} = \operatorname{conv\_pool}_\tau \bigl(
            x_{w'}, \; -x_{w'} \bigr), \quad
            x_{w'} = (1 - \tilde{A}_{w,w'}) + U_{\phi,w'}

    The operator acts as a "weakest link" detector: if a world is highly
    accessible (Ã ≈ 1) but ϕ is false there, the score collapses.

    **Top-k aggregation.** With ``top_k=k`` each endpoint aggregates only
    the *k smallest* of its own implication terms — ``(1 - Ã) + L`` for
    the lower bound, ``(1 - Ã) + U`` for the upper — instead of the full
    row. Because the terms are selected on the aggregated quantity (not on
    ``Ã`` alone), the true minimum is always among the kept terms:
    ``L_□ <= min`` and ``U_□ >= min`` still hold (Theorem 1), the smooth
    lower bound is within ``tau * log(k)`` of the crisp minimum, the result
    does not depend on ``|W|``, and gradients reach exactly the ``k``
    selected entries of ``Ã`` per endpoint. The ``|W| x |W|`` term matrix
    is still formed; the aggregation itself is ``O(k * |W|)``.

    .. warning::
       Do **not** emulate ``top_k`` by zeroing entries of ``Ã`` before the
       call (the ``top_k=`` of the accessibility modules up to 0.2.0). Zeroed
       entries still enter the log-sum-exp with term ``1 + L`` and their
       summed mass drives the bounds to ``[0, 1]`` as ``|W|`` grows, and
       choosing neighbours by ``Ã`` alone is unsound.

    Args:
        prop_bounds: Truth bounds of shape ``(|W|, 2)`` where columns are
            ``[L, U]``, or ``(|W|,)`` for point-valued truth values (treated
            as both L and U).
        accessibility: Accessibility matrix of shape ``(|W|, |W|)``, values
            in [0, 1].
        tau: Temperature. Default 0.1.
        top_k: If set, aggregate only the ``top_k`` smallest implication
            terms per world and endpoint. ``None`` (default) aggregates the
            full row. Must be a positive integer.

    Returns:
        Tensor of shape ``(|W|, 2)`` or ``(|W|,)`` with necessity bounds.
    """
    point_valued = prop_bounds.dim() == 1
    if point_valued:
        prop_bounds = prop_bounds.unsqueeze(-1).expand(-1, 2)

    L_phi = prop_bounds[:, 0]  # (|W|,)
    U_phi = prop_bounds[:, 1]  # (|W|,)

    # (|W|, |W|): implication terms per source-target world pair
    impl_L = (1.0 - accessibility) + L_phi.unsqueeze(0)  # broadcast target
    impl_U = (1.0 - accessibility) + U_phi.unsqueeze(0)

    # Top-k: keep the k smallest terms of each endpoint (the true minimum is
    # always among them), so the aggregations below see only k terms.
    impl_L = _select_terms(impl_L, top_k, largest=False)
    impl_U = _select_terms(impl_U, top_k, largest=False)

    # Lower bound: smooth_min over target worlds (dim=1)
    L_box = smooth_min(impl_L, tau=tau, dim=1)

    # Upper bound: conv_pool with the negated implication as the logit (z = -x)
    U_box = conv_pool(impl_U, -impl_U, tau=tau, dim=1)

    result = torch.stack([L_box, U_box], dim=-1)
    result = torch.clamp(result, 0.0, 1.0)

    if point_valued:
        return result[:, 0]
    return result


def possibility(
    prop_bounds: Tensor,
    accessibility: Tensor,
    tau: float = 0.1,
    top_k: int | None = None,
) -> Tensor:
    r"""Possibility (Diamond / ♢) operator — differentiable Kripke semantics.

    Computes truth bounds for ♢ϕ across all worlds. For each world *w*:

    .. math::
        L_{\Diamond\phi,w} = \operatorname{conv\_pool}_\tau \bigl(
            x_{w'}, \; x_{w'} \bigr), \quad
            x_{w'} = \tilde{A}_{w,w'} + L_{\phi,w'} - 1

    .. math::
        U_{\Diamond\phi,w} = \operatorname{smooth\_max}_\tau \bigl\{
            \tilde{A}_{w,w'} + U_{\phi,w'} - 1 \bigr\}_{w' \in W}

    The operator acts as an "evidence scout": it activates if it finds any
    world that is both accessible and where ϕ is true.

    **Top-k aggregation.** With ``top_k=k`` each endpoint aggregates only
    the *k largest* of its own conjunction terms — ``Ã + L - 1`` for the
    lower bound, ``Ã + U - 1`` for the upper — so the true maximum is
    always among the kept terms: ``L_♢ <= max`` and ``U_♢ >= max`` still
    hold, the smooth upper bound is within ``tau * log(k)`` of the crisp
    maximum, nothing depends on ``|W|``, and gradients reach exactly the
    ``k`` selected entries of ``Ã`` per endpoint. See :func:`necessity`
    for why the selection must be made on the aggregated terms rather than
    on ``Ã`` alone.

    Args:
        prop_bounds: Truth bounds of shape ``(|W|, 2)`` or ``(|W|,)``.
        accessibility: Accessibility matrix ``(|W|, |W|)`` in [0, 1].
        tau: Temperature. Default 0.1.
        top_k: If set, aggregate only the ``top_k`` largest conjunction
            terms per world and endpoint. ``None`` (default) aggregates the
            full row. Must be a positive integer.

    Returns:
        Tensor of shape ``(|W|, 2)`` or ``(|W|,)`` with possibility bounds.
    """
    point_valued = prop_bounds.dim() == 1
    if point_valued:
        prop_bounds = prop_bounds.unsqueeze(-1).expand(-1, 2)

    L_phi = prop_bounds[:, 0]
    U_phi = prop_bounds[:, 1]

    # conjunction terms
    conj_L = accessibility + L_phi.unsqueeze(0) - 1.0
    conj_U = accessibility + U_phi.unsqueeze(0) - 1.0

    # Top-k: keep the k largest terms of each endpoint (the true maximum is
    # always among them).
    conj_L = _select_terms(conj_L, top_k, largest=True)
    conj_U = _select_terms(conj_U, top_k, largest=True)

    # Lower bound: conv_pool with the conjunction as both value and logit (z = x)
    L_dia = conv_pool(conj_L, conj_L, tau=tau, dim=1)

    # Upper bound: smooth_max (weighted existential)
    U_dia = smooth_max(conj_U, tau=tau, dim=1)

    result = torch.stack([L_dia, U_dia], dim=-1)
    result = torch.clamp(result, 0.0, 1.0)

    if point_valued:
        return result[:, 1]  # for point values return upper (existential)
    return result


def until(
    phi_bounds: Tensor,
    psi_bounds: Tensor,
    accessibility: Tensor,
    tau: float = 0.1,
) -> Tensor:
    r"""Until (U) operator — differentiable temporal semantics.

    Computes truth bounds for ``ϕ U ψ`` ("ϕ holds until ψ becomes true")
    over a forward-time accessibility structure.  For each time step *t*:

    .. math::
        (\phi\;\mathcal{U}\;\psi)_t = \bigvee_{t' \geq t}
            \Bigl(\psi_{t'} \;\wedge\; \bigwedge_{t \leq s < t'} \phi_s\Bigr)

    The implementation uses a backward dynamic-programming sweep that
    remains fully differentiable:

    .. math::
        U_t = \psi_t \;\lor\; (\phi_t \;\land\; U_{t+1})

    with ``U_T = ψ_T`` at the final time step.  All connectives use
    Łukasiewicz fuzzy logic (see :func:`conjunction`, :func:`disjunction`)
    so that the computation stays in [0, 1] and gradients flow smoothly.

    This closes the expressiveness gap with STLCG (Leung et al., 2023)
    which supports the Until operator for signal temporal logic.

    Args:
        phi_bounds: Truth bounds for ϕ, shape ``(T, 2)`` or ``(T,)``.
        psi_bounds: Truth bounds for ψ, shape ``(T, 2)`` or ``(T,)``.
        accessibility: Forward-time accessibility matrix ``(T, T)``.
            Only the temporal ordering matters; the matrix is used to
            determine the number of time steps. No aggregation over
            ``accessibility`` takes place, so there is no ``top_k``
            parameter here (see :func:`necessity` / :func:`possibility`).
        tau: Temperature (unused in the DP formulation, kept for API
            consistency). Default 0.1.

    Returns:
        Truth bounds for ``ϕ U ψ``, same shape as inputs.
    """
    point_valued = phi_bounds.dim() == 1
    if point_valued:
        phi_bounds = phi_bounds.unsqueeze(-1).expand(-1, 2)
        psi_bounds = psi_bounds.unsqueeze(-1).expand(-1, 2)

    T = phi_bounds.shape[0]

    L_phi, U_phi = phi_bounds[:, 0], phi_bounds[:, 1]
    L_psi, U_psi = psi_bounds[:, 0], psi_bounds[:, 1]

    # Build results as lists to avoid in-place ops (autograd-safe)
    L_list: list[Tensor] = [torch.tensor(0.0)] * T
    U_list: list[Tensor] = [torch.tensor(0.0)] * T

    # Base case: at the last step, Until reduces to ψ
    L_list[T - 1] = L_psi[T - 1]
    U_list[T - 1] = U_psi[T - 1]

    # Backward sweep: U_t = ψ_t ∨ (ϕ_t ∧ U_{t+1})
    for t in range(T - 2, -1, -1):
        # ϕ_t ∧ U_{t+1}  (Łukasiewicz conjunction)
        L_continue = torch.clamp(L_phi[t] + L_list[t + 1] - 1.0, min=0.0)
        U_continue = torch.min(U_phi[t], U_list[t + 1])

        # ψ_t ∨ (ϕ_t ∧ U_{t+1})  (Łukasiewicz disjunction)
        L_list[t] = torch.max(L_psi[t], L_continue)
        U_list[t] = torch.clamp(U_psi[t] + U_continue, max=1.0)

    result = torch.stack(
        [torch.stack(L_list), torch.stack(U_list)], dim=-1
    )

    if point_valued:
        return result[:, 0]
    return result


# ---------------------------------------------------------------------------
# Contradiction measure
# ---------------------------------------------------------------------------


def contradiction(bounds: Tensor, upper: Tensor | None = None) -> Tensor:
    r"""Compute contradiction loss from truth bounds.

    A contradiction arises when a lower bound exceeds an upper bound, an
    inconsistency no classical truth assignment can satisfy:

    .. math::
        \mathcal{L}_{\text{contra}} = \sum \max(0,\; L - U).

    Two equivalent call forms are accepted:

    - **Stacked** — ``contradiction(bounds)`` with ``bounds`` of shape
      ``(..., 2)`` holding ``[L, U]`` on the last dimension. This is the
      bound contradiction ``ReLU(L_phi - U_phi)``.
    - **Split** — ``contradiction(L, U)`` with the lower and upper sources
      passed as separate tensors of matching shape, for when they are
      computed apart. For example ``contradiction(box, dia)`` penalises a
      necessity that exceeds its possibility (the modal
      ``Box phi -> Diamond phi`` consistency requirement).

    The two forms agree by construction::

        contradiction(L, U) == contradiction(torch.stack([L, U], dim=-1))

    Args:
        bounds: Either a ``(..., 2)`` bound tensor (stacked form), or the
            lower-bound tensor (split form, when ``upper`` is provided).
        upper: Upper-bound tensor matching ``bounds``. If omitted,
            ``bounds`` is read as a stacked ``[L, U]`` pair.

    Returns:
        Scalar contradiction loss, summed over all elements.
    """
    if upper is None:
        L = bounds[..., 0]
        U = bounds[..., 1]
    else:
        L = bounds
        U = upper
    return torch.relu(L - U).sum()
