"""Tests for torchmodal.diagnostics and functional.box_width_entropy."""

import math

import pytest
import torch
import torch.nn as nn

from torchmodal import functional as F
from torchmodal.diagnostics import (
    GradientHealthError,
    assert_has_signal,
    gradient_health,
)


def _nested_box(A, depth, n=8, tau=0.1):
    def run():
        bounds = torch.ones(n, 2)
        for _ in range(depth):
            bounds = F.necessity(bounds, A, tau=tau)
        return bounds

    return run


class TestBoxWidthEntropy:
    def test_equals_the_conv_pool_smooth_min_gap(self):
        """The identity the function is defined by, to 1e-12."""
        torch.manual_seed(0)
        worst = 0.0
        for _ in range(500):
            n = int(torch.randint(2, 10, (1,)).item())
            tau = float(torch.empty(1).uniform_(0.03, 1.0))
            A = torch.rand(n, n, dtype=torch.float64)
            bounds = torch.rand(n, 2, dtype=torch.float64).sort(dim=1).values
            terms = (1.0 - A) + bounds[:, 1].unsqueeze(0)
            gap = F.conv_pool(terms, -terms, tau=tau, dim=1) - F.smooth_min(
                terms, tau=tau, dim=1
            )
            got = F.box_width_entropy(A, bounds, tau=tau)
            worst = max(worst, (got - gap).abs().max().item())
        assert worst < 1e-12

    def test_equals_the_box_interval_when_unclamped(self):
        """With L == U and no clamping, it *is* U_box - L_box."""
        torch.manual_seed(1)
        checked = 0
        for _ in range(500):
            n = int(torch.randint(2, 10, (1,)).item())
            tau = float(torch.empty(1).uniform_(0.03, 0.5))
            A = torch.rand(n, n, dtype=torch.float64)
            v = torch.rand(n, dtype=torch.float64)
            bounds = torch.stack([v, v], dim=-1)
            terms = (1.0 - A) + v.unsqueeze(0)
            raw_lo = F.smooth_min(terms, tau=tau, dim=1)
            raw_hi = F.conv_pool(terms, -terms, tau=tau, dim=1)
            if raw_lo.min() < 0 or raw_hi.max() > 1:
                continue  # the necessity output clamp would engage
            checked += 1
            box = F.necessity(bounds, A, tau=tau)
            assert torch.allclose(
                F.box_width_entropy(A, bounds, tau=tau),
                box[:, 1] - box[:, 0],
                atol=1e-12,
            )
        assert checked > 50

    def test_bounded_by_tau_log_n(self):
        torch.manual_seed(2)
        for _ in range(300):
            n = int(torch.randint(2, 10, (1,)).item())
            A = torch.rand(n, n)
            bounds = torch.rand(n, 2).sort(dim=1).values
            width = F.box_width_entropy(A, bounds, tau=0.1)
            assert width.max().item() <= 0.1 * math.log(n) + 1e-6

    def test_equality_when_all_terms_tie(self):
        n, tau = 7, 0.1
        width = F.box_width_entropy(
            torch.ones(n, n), torch.full((n, 2), 0.5), tau=tau
        )
        assert width[0].item() == pytest.approx(tau * math.log(n), abs=1e-6)

    def test_non_negative(self):
        torch.manual_seed(3)
        for _ in range(100):
            n = int(torch.randint(1, 10, (1,)).item())
            width = F.box_width_entropy(
                torch.rand(n, n), torch.rand(n, 2).sort(dim=1).values, tau=0.1
            )
            assert (width >= 0.0).all()

    def test_single_kept_term_has_zero_width(self):
        A = torch.rand(9, 9)
        bounds = torch.rand(9, 2).sort(dim=1).values
        width = F.box_width_entropy(A, bounds, tau=0.1, top_k=1)
        assert torch.allclose(width, torch.zeros(9), atol=1e-9)

    def test_top_k_matches_full_when_k_covers_every_term(self):
        A = torch.rand(6, 6)
        bounds = torch.rand(6, 2).sort(dim=1).values
        assert torch.allclose(
            F.box_width_entropy(A, bounds, tau=0.1, top_k=6),
            F.box_width_entropy(A, bounds, tau=0.1),
        )

    def test_accepts_point_valued_bounds(self):
        A = torch.rand(5, 5)
        v = torch.rand(5)
        assert torch.allclose(
            F.box_width_entropy(A, v, tau=0.1),
            F.box_width_entropy(A, torch.stack([v, v], dim=-1), tau=0.1),
        )


class TestGradientHealthDetectsDeadTerms:
    def test_nested_box_is_healthy_while_shallow(self):
        for depth in (1, 2, 3, 4):
            A = torch.ones(8, 8, requires_grad=True)
            report = gradient_health(_nested_box(A, depth), {"A": A})
            assert report["healthy"], (depth, report["issues"])

    def test_nested_box_is_dead_once_it_floors(self):
        for depth in (5, 6):
            A = torch.ones(8, 8, requires_grad=True)
            report = gradient_health(_nested_box(A, depth), {"A": A})
            assert not report["healthy"]
            assert report["outputs"]["output.L"]["dead"]
            assert report["outputs"]["output.L"]["pinned_at_floor"]
            assert any("dead" in w for w in report["warnings"])

    def test_splits_bounds_into_endpoints(self):
        """L and U fail in opposite directions and must be checked apart."""
        A = torch.ones(8, 8, requires_grad=True)
        report = gradient_health(_nested_box(A, 6), {"A": A})
        assert set(report["outputs"]) == {"output.L", "output.U"}
        assert report["outputs"]["output.L"]["pinned_at_floor"]
        assert report["outputs"]["output.U"]["pinned_at_ceiling"]
        assert not report["outputs"]["output.L"]["pinned_at_ceiling"]
        assert not report["outputs"]["output.U"]["pinned_at_floor"]

    def test_reports_a_vacuous_bound(self):
        A = torch.ones(8, 8, requires_grad=True)
        report = gradient_health(_nested_box(A, 6), {"A": A})
        assert report["vacuous"] == ["output"]

    def test_detects_untils_disconnected_relation(self):
        T = 6
        phi = torch.stack([torch.full((T,), 0.9), torch.ones(T)], dim=-1)
        psi = torch.zeros(T, 2)
        psi[T - 1] = 1.0
        A = torch.triu(torch.ones(T, T)).requires_grad_(True)
        report = gradient_health(lambda: F.until(phi, psi, A), {"A": A})
        assert not report["healthy"]
        assert any("no autograd path" in i for i in report["issues"])
        assert report["params"]["A"]["grad_vanished"]

    def test_healthy_for_a_shallow_box(self):
        torch.manual_seed(0)
        A = torch.rand(5, 5, requires_grad=True)
        bounds = torch.rand(5, 2).sort(dim=1).values
        report = gradient_health(
            lambda: F.necessity(bounds, A, tau=0.1), {"A": A}
        )
        assert report["healthy"], report["issues"]
        assert report["params"]["A"]["grad_max_abs"] > 0.0

    def test_flags_a_detached_parameter(self):
        A = torch.rand(5, 5, requires_grad=True)
        unused = torch.rand(3, requires_grad=True)
        bounds = torch.rand(5, 2).sort(dim=1).values
        report = gradient_health(
            lambda: F.necessity(bounds, A, tau=0.1),
            {"A": A, "unused": unused},
        )
        assert not report["healthy"]
        assert report["params"]["unused"]["grad_vanished"]
        assert any("unused" in i for i in report["issues"])

    def test_a_pinned_endpoint_alone_does_not_make_it_unhealthy(self):
        """A single box over an all-true proposition pins U at 1.

        That is a *correct* upper bound, not a defect. Whether its
        gradient survives the output clamp at exactly 1.0 is a torch
        version convention (2.8 passes 1.0, 2.14 passes 0.0), so the
        endpoint is reported in ``warnings`` and must not affect
        ``healthy`` — otherwise this tool's verdict would depend on the
        installed torch.
        """
        A = torch.ones(4, 4, requires_grad=True)
        report = gradient_health(
            lambda: F.necessity(torch.ones(4, 2), A, tau=0.1), {"A": A}
        )
        assert report["outputs"]["output.U"]["pinned_at_ceiling"]
        assert any(
            "saturated" in w or "dead" in w for w in report["warnings"]
        )
        assert report["healthy"]
        assert report["issues"] == []

    def test_verdict_is_independent_of_clamp_boundary_gradients(self):
        """The bound is informative, so the report is healthy either way.

        Pins the regression behind the torch 2.8 -> 2.14 clamp change:
        `necessity` over a non-degenerate frame yields a bound that is not
        vacuous, and a non-vacuous bound with live gradient is healthy
        regardless of what the clamp does at its boundary.
        """
        torch.manual_seed(0)
        A = torch.rand(6, 6, requires_grad=True)
        bounds = torch.rand(6, 2).sort(dim=1).values
        report = gradient_health(
            lambda: F.necessity(bounds, A, tau=0.1), {"A": A}
        )
        assert report["vacuous"] == []
        assert report["healthy"]


class TestGradientHealthApi:
    def test_accepts_a_module(self):
        model = nn.Linear(3, 2)
        report = gradient_health(
            lambda: torch.sigmoid(model(torch.rand(4, 3))), model
        )
        assert set(report["params"]) == {"weight", "bias"}

    def test_accepts_a_bare_tensor(self):
        x = torch.rand(4, requires_grad=True)
        report = gradient_health(lambda: torch.sigmoid(x), x)
        assert "param" in report["params"]

    def test_accepts_a_sequence(self):
        a = torch.rand(4, requires_grad=True)
        b = torch.rand(4, requires_grad=True)
        report = gradient_health(lambda: torch.sigmoid(a + b), [a, b])
        assert set(report["params"]) == {"param0", "param1"}

    def test_forwards_args_and_kwargs(self):
        A = torch.rand(5, 5, requires_grad=True)
        bounds = torch.rand(5, 2).sort(dim=1).values
        report = gradient_health(F.necessity, {"A": A}, bounds, A, tau=0.1)
        assert report["healthy"]

    def test_renames_outputs(self):
        x = torch.rand(4, requires_grad=True)
        report = gradient_health(
            lambda: torch.sigmoid(x), x, names=["p"], bounds=False
        )
        assert set(report["outputs"]) == {"p"}

    def test_bounds_false_keeps_the_tensor_whole(self):
        A = torch.rand(5, 5, requires_grad=True)
        bounds = torch.rand(5, 2).sort(dim=1).values
        report = gradient_health(
            lambda: F.necessity(bounds, A, tau=0.1), {"A": A}, bounds=False
        )
        assert set(report["outputs"]) == {"output"}

    def test_bounds_true_rejects_a_non_bounds_output(self):
        x = torch.rand(4, 3, requires_grad=True)
        with pytest.raises(ValueError, match="last dimension"):
            gradient_health(lambda: torch.sigmoid(x), x, bounds=True)

    def test_rejects_a_non_tensor_result(self):
        x = torch.rand(4, requires_grad=True)
        with pytest.raises(TypeError, match="must return a Tensor"):
            gradient_health(lambda: "not a tensor", x)

    def test_does_not_populate_dot_grad(self):
        """The diagnostic must not disturb a live training loop."""
        A = torch.rand(5, 5, requires_grad=True)
        bounds = torch.rand(5, 2).sort(dim=1).values
        gradient_health(lambda: F.necessity(bounds, A, tau=0.1), {"A": A})
        assert A.grad is None


class TestAssertHasSignal:
    def test_raises_on_a_collapsed_term(self):
        A = torch.ones(8, 8, requires_grad=True)
        with pytest.raises(GradientHealthError, match="vacuous"):
            assert_has_signal(_nested_box(A, 6), {"A": A})

    def test_raises_on_a_disconnected_relation(self):
        T = 6
        phi = torch.stack([torch.full((T,), 0.9), torch.ones(T)], dim=-1)
        psi = torch.zeros(T, 2)
        psi[T - 1] = 1.0
        A = torch.triu(torch.ones(T, T)).requires_grad_(True)
        with pytest.raises(GradientHealthError, match="autograd path"):
            assert_has_signal(lambda: F.until(phi, psi, A), {"A": A})

    def test_includes_the_message_prefix(self):
        A = torch.ones(8, 8, requires_grad=True)
        with pytest.raises(GradientHealthError, match="deep nest"):
            assert_has_signal(_nested_box(A, 6), {"A": A}, msg="deep nest")

    def test_returns_the_report_when_healthy(self):
        torch.manual_seed(0)
        A = torch.rand(5, 5, requires_grad=True)
        bounds = torch.rand(5, 2).sort(dim=1).values
        report = assert_has_signal(
            lambda: F.necessity(bounds, A, tau=0.1), {"A": A}
        )
        assert report["healthy"]

    def test_is_an_assertion_error(self):
        assert issubclass(GradientHealthError, AssertionError)
