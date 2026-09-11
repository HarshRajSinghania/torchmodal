"""Regression tests for top-k aggregation in the modal operators.

Two defects of the masking up to 0.2.0 (``top_k=`` on the accessibility
modules, which zeroed all but the k largest entries of A before the operators
aggregated the full row):

1. **Selection by the wrong quantity.** □'s lower bound minimises
   ``(1 - A) + L``, which depends on ``L``, but neighbours were chosen by
   ``A`` alone — a world just below the k-th access value whose ``L`` carries
   the violation was dropped and necessity over-reported (Theorem 1 violated).
2. **Zeroing is not excluding.** A zeroed entry still entered the
   log-sum-exp with term ``1 + L`` (weight ``exp(-(1+L)/tau)``), so the bounds
   drifted to ``[0, 1]`` as ``|W|`` grew with k fixed.

The fix selects the k neighbours by the *aggregation argument*, per endpoint,
and aggregates only those (``top_k=`` on ``functional.necessity`` /
``possibility`` and the ``Necessity`` / ``Possibility`` modules).

``_legacy_*`` below reconstructs the old computation path — sparsify ``A`` with
``top_k_mask`` (still exported as a relation-sparsification utility), then
aggregate the full row — so the defects stay demonstrable.
"""

import math
import warnings

import pytest
import torch

import torchmodal
from torchmodal import functional as F
from torchmodal import nn
from torchmodal.inference import FormulaGraph, upward_downward
from torchmodal.nn import top_k_mask

TAU = 0.1


def _legacy_necessity(b, A, k, tau=TAU):
    """Pre-fix path: mask A by its k largest entries, aggregate the full row."""
    return F.necessity(b, top_k_mask(A, k), tau=tau)


def _legacy_possibility(b, A, k, tau=TAU):
    return F.possibility(b, top_k_mask(A, k), tau=tau)


def crisp_box(b, A):
    return torch.min((1 - A) + b[:, 0].unsqueeze(0), dim=1).values


def crisp_dia(b, A):
    return torch.max(A + b[:, 1].unsqueeze(0) - 1, dim=1).values


def random_model(n, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.rand(n, n, generator=g)
    L = torch.rand(n, generator=g) * 0.7
    U = L + torch.rand(n, generator=g) * (1 - L)
    return A, torch.stack([L, U], dim=1)


# -- defect 1: selection ---------------------------------------------------------


def test_counterexample_legacy_overreports_necessity():
    """A row = [0.90 0.88 0.86 0.84 0.83], L = [0.9 0.9 0.9 0.9 0.0]: the
    world with the smallest access carries the violation. Legacy top-4 keeps
    {0,1,2,3} and reports 0.86 against a crisp minimum of 0.17."""
    A = torch.zeros(5, 5)
    A[0] = torch.tensor([0.90, 0.88, 0.86, 0.84, 0.83])
    b = torch.tensor([0.9, 0.9, 0.9, 0.9, 0.0]).unsqueeze(1).expand(-1, 2)
    legacy = _legacy_necessity(b, A, 4)[0, 0]
    fixed = F.necessity(b, A, tau=TAU, top_k=4)[0, 0]
    crisp = crisp_box(b, A)[0]
    assert legacy > crisp + 0.5  # the defect: 0.86 vs 0.17
    assert fixed <= crisp + 1e-6  # sound
    assert fixed >= crisp - TAU * math.log(4) - 1e-6  # and within the τ log k gap


def test_counterexample_legacy_underreports_possibility():
    """Dual of the above for ♢: the violating world is the one where ϕ is
    true, and it sits just below the k-th access value."""
    A = torch.zeros(5, 5)
    A[0] = torch.tensor([0.90, 0.88, 0.86, 0.84, 0.83])
    b = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]).unsqueeze(1).expand(-1, 2)
    legacy = _legacy_possibility(b, A, 4)[0, 1]
    fixed = F.possibility(b, A, tau=TAU, top_k=4)[0, 1]
    crisp = crisp_dia(b, A)[0]  # 0.83
    assert legacy < crisp - 0.5  # the defect: U_◇ < true max
    assert fixed >= crisp - 1e-6  # sound
    assert fixed <= crisp + TAU * math.log(4) + 1e-6


@pytest.mark.parametrize("n,k", [(32, 4), (128, 8), (512, 8)])
def test_fixed_masked_bounds_are_sound_for_every_world(n, k):
    A, b = random_model(n, seed=n)
    box = F.necessity(b, A, tau=TAU, top_k=k)
    dia = F.possibility(b, A, tau=TAU, top_k=k)
    cb, cd = crisp_box(b, A), crisp_dia(b, A)
    # crisp upper of □ uses U; crisp lower of ◇ uses L
    cu = torch.min((1 - A) + b[:, 1].unsqueeze(0), dim=1).values
    cl = torch.max(A + b[:, 0].unsqueeze(0) - 1, dim=1).values
    assert torch.all(box[:, 0] <= cb + 1e-6)  # L_□ ≤ min
    assert torch.all(box[:, 1] >= cu.clamp(0, 1) - 1e-6)  # U_□ ≥ min
    assert torch.all(dia[:, 0] <= cl.clamp(0, 1) + 1e-6)  # L_◇ ≤ max
    assert torch.all(dia[:, 1] >= cd.clamp(0, 1) - 1e-6)  # U_◇ ≥ max
    # the log-sum-exp endpoints are within τ log k of the extremum, never worse
    assert torch.all(box[:, 0] >= (cb - TAU * math.log(k)).clamp(0) - 1e-6)
    assert torch.all(dia[:, 1] <= (cd + TAU * math.log(k)).clamp(max=1) + 1e-6)


@pytest.mark.parametrize("n,k", [(128, 8), (512, 8)])
def test_legacy_masking_is_unsound_somewhere(n, k):
    A, b = random_model(n, seed=n + 1)
    legacy = _legacy_necessity(b, A, k)[:, 0]
    assert torch.any(legacy > crisp_box(b, A) + 1e-4)


# -- defect 2: zeroing is not excluding -----------------------------------------


def _sparse_world(n):
    """World 0 sees 8 real neighbours; every other world is inaccessible (A = 0)
    and ϕ is false there — pure irrelevant mass that must not move the bounds."""
    A = torch.zeros(n, n)
    A[0, 1:9] = torch.linspace(0.95, 0.80, 8)
    b = torch.zeros(n, 2)
    b[1:9, 0], b[1:9, 1] = 0.5, 0.6
    return A, b


def test_legacy_bounds_depend_on_world_count():
    A16, b16 = _sparse_world(16)
    small = _legacy_necessity(b16, A16, 8)[0]
    A, b = _sparse_world(8192)
    large = _legacy_necessity(b, A, 8)[0]
    assert abs(float(large[0] - small[0])) > 0.2  # lower drifts toward 0
    assert abs(float(large[1] - small[1])) > 0.2  # upper drifts toward 1


def test_fixed_bounds_are_invariant_to_world_count():
    ref = None
    for n in (16, 256, 8192):
        A, b = _sparse_world(n)
        out = F.necessity(b, A, tau=TAU, top_k=8)[0]
        if ref is None:
            ref = out
        assert torch.allclose(out, ref, atol=1e-5), (n, out, ref)


def test_fixed_bounds_match_the_neighbourhood_alone():
    """With top_k=8 the bounds equal those of the 8-world sub-model."""
    A, b = _sparse_world(4096)
    out = F.necessity(b, A, tau=TAU, top_k=8)[0]
    ref = F.necessity(b[1:9], A[0:1, 1:9], tau=TAU)[0]
    assert torch.allclose(out, ref, atol=1e-6), (out, ref)


def test_fixed_equals_unmasked_when_k_covers_the_extremes():
    """top_k=None and top_k >= |W| reduce exactly to the unmasked operators."""
    A, b = random_model(24, seed=7)
    for k in (None, 24, 1000):
        assert torch.equal(
            F.necessity(b, A, tau=TAU, top_k=k), F.necessity(b, A, tau=TAU)
        )
        assert torch.equal(
            F.possibility(b, A, tau=TAU, top_k=k), F.possibility(b, A, tau=TAU)
        )


def test_point_valued_inputs_work_with_top_k():
    A, b = random_model(16, seed=11)
    p = b[:, 0]
    box = F.necessity(p, A, tau=TAU, top_k=4)
    dia = F.possibility(p, A, tau=TAU, top_k=4)
    assert box.shape == (16,) and dia.shape == (16,)
    assert torch.all(box <= crisp_box(b, A) + 1e-6)
    crisp = torch.max(A + p.unsqueeze(0) - 1, dim=1).values.clamp(0, 1)
    assert torch.all(dia >= crisp - 1e-6)


def test_gradients_flow_to_the_selected_neighbours():
    A, b = random_model(40, seed=3)
    A.requires_grad_(True)
    out = F.necessity(b, A, tau=TAU, top_k=6)[:, 0]
    out.sum().backward()
    nz = (A.grad.abs() > 0).sum(dim=1)
    live = (out > 0) & (out < 1)  # rows not flattened by the [0,1] clamp
    assert torch.all(nz <= 6)
    assert torch.all(nz[live] >= 1)


def test_possibility_gradients_flow_to_the_selected_neighbours():
    A, b = random_model(40, seed=5)
    A.requires_grad_(True)
    out = F.possibility(b, A, tau=TAU, top_k=6)[:, 1]
    out.sum().backward()
    nz = (A.grad.abs() > 0).sum(dim=1)
    live = (out > 0) & (out < 1)
    assert torch.all(nz <= 6)
    assert torch.all(nz[live] >= 1)


@pytest.mark.parametrize("bad", [0, -1])
def test_invalid_top_k_raises(bad):
    A, b = random_model(8, seed=1)
    with pytest.raises(ValueError):
        F.necessity(b, A, tau=TAU, top_k=bad)
    with pytest.raises(ValueError):
        F.possibility(b, A, tau=TAU, top_k=bad)
    with pytest.raises(ValueError):
        nn.Necessity(top_k=bad)
    with pytest.raises(ValueError):
        nn.Possibility(top_k=bad)


# -- modules, inference, and the deprecated accessibility-module top_k -----------


def test_modules_thread_top_k_to_functional():
    A, b = random_model(32, seed=9)
    box = nn.Necessity(tau=TAU, top_k=4)
    dia = nn.Possibility(tau=TAU, top_k=4)
    assert torch.equal(box(b, A), F.necessity(b, A, tau=TAU, top_k=4))
    assert torch.equal(dia(b, A), F.possibility(b, A, tau=TAU, top_k=4))
    assert "top_k=4" in repr(box) and "top_k=4" in repr(dia)
    # default is still the unmasked operator
    assert torch.equal(nn.Necessity(tau=TAU)(b, A), F.necessity(b, A, tau=TAU))


def test_kripke_model_threads_top_k():
    torch.manual_seed(0)
    model = torchmodal.KripkeModel(
        num_worlds=32, accessibility=nn.LearnableAccessibility(32), tau=TAU, top_k=4
    )
    model.add_proposition("p", learnable=False)
    model.get_proposition("p").set_bounds(random_model(32, seed=2)[1])
    A = model.get_accessibility()
    b = model.get_bounds("p")
    assert torch.equal(model.necessity("p"), F.necessity(b, A, tau=TAU, top_k=4))
    assert torch.equal(model.possibility("p"), F.possibility(b, A, tau=TAU, top_k=4))


def test_systems_thread_top_k():
    A, b = random_model(16, seed=4)
    K = torchmodal.EpistemicOperator(tau=TAU, top_k=4)
    assert torch.equal(K(b, A[0]), F.necessity(b, A[0:1], tau=TAU, top_k=4).squeeze(0))
    T = torchmodal.TemporalOperator(num_steps=16, tau=TAU, top_k=4)
    R = T.build_forward_accessibility()
    assert torch.equal(T.globally(b, R), F.necessity(b, R, tau=TAU, top_k=4))
    assert torch.equal(T.finally_(b, R), F.possibility(b, R, tau=TAU, top_k=4))
    M = torchmodal.MultiAgentKripke(num_agents=4, num_steps=4, tau=TAU, top_k=3)
    assert M.box.top_k == 3 and M.diamond.top_k == 3 and M.temporal.box.top_k == 3


def test_upward_downward_top_k_is_sound_and_dependency_free():
    """Inference with top_k: the □ node's upward bound is sound w.r.t. the
    full-row crisp minimum, and the downward rule (which uses the full A)
    still runs; with top_k=None the result is unchanged from before."""
    A, b = random_model(24, seed=6)
    graph = FormulaGraph()
    graph.add_atomic("p")
    graph.add_necessity("box_p", "p")
    graph.add_possibility("dia_p", "p")

    def run(top_k):
        bounds = {
            "p": b.clone(),
            "box_p": torch.zeros(24, 2) + torch.tensor([0.0, 1.0]),
            "dia_p": torch.zeros(24, 2) + torch.tensor([0.0, 1.0]),
        }
        return upward_downward(graph, bounds, A, tau=TAU, top_k=top_k)

    ref = run(None)
    plain = upward_downward(
        graph,
        {
            "p": b.clone(),
            "box_p": torch.zeros(24, 2) + torch.tensor([0.0, 1.0]),
            "dia_p": torch.zeros(24, 2) + torch.tensor([0.0, 1.0]),
        },
        A,
        tau=TAU,
    )
    for name in ("p", "box_p", "dia_p"):
        assert torch.equal(ref[name], plain[name])

    out = run(4)
    assert torch.all(out["box_p"][:, 0] <= crisp_box(b, A) + 1e-6)
    assert torch.all(out["dia_p"][:, 1] >= crisp_dia(b, A).clamp(0, 1) - 1e-6)
    assert torch.all(out["p"][:, 0] <= b[:, 0] + 1e-6)  # leaves not moved past truth
    assert torch.all(out["p"][:, 1] >= b[:, 1] - 1e-6)


@pytest.mark.parametrize(
    "make",
    [
        lambda: nn.FixedAccessibility(torch.rand(6, 6), top_k=2),
        lambda: nn.LearnableAccessibility(6, top_k=2),
        lambda: nn.MetricAccessibility(6, embed_dim=4, top_k=2),
        lambda: nn.AttentionAccessibility(input_dim=8, num_heads=2, top_k=2),
    ],
)
def test_accessibility_top_k_is_deprecated_and_no_longer_masks(make):
    with pytest.warns(DeprecationWarning, match="unsound"):
        access = make()
    if isinstance(access, nn.AttentionAccessibility):
        A = access(torch.randn(6, 8))
    else:
        A = access()
    # the deprecated argument does not zero anything
    assert torch.all(A > 0)
    assert "top_k" not in repr(access) or "top_k=None" in repr(access)


def test_accessibility_sparsify_is_an_explicit_modelling_choice():
    torch.manual_seed(0)
    R = torch.rand(6, 6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no deprecation noise for sparsify
        access = nn.FixedAccessibility(R, sparsify=2)
    A = access()
    assert torch.equal(A, top_k_mask(R, 2))
    assert torch.all((A > 0).sum(dim=1) == 2)
    assert "sparsify=2" in repr(access)
