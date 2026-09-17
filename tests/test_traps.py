"""Regression tests pinning the library's known traps.

Every test in this file asserts a *limitation*. They exist so that a
behaviour someone might "fix" by accident cannot silently return, and so
that the documented numbers stay true. Each one corresponds to an
admonition in a docstring; if a test here starts failing, the matching
docstring is now wrong and must be updated with it.

Covered:

- :func:`torchmodal.functional.until` is inert with respect to its
  accessibility and its temperature.
- :func:`torchmodal.functional.conv_pool` is not monotone.
- :func:`torchmodal.functional.contradiction` has a dead zone after a
  modal neuron, of width exactly
  :func:`torchmodal.functional.box_width_entropy`.
- Nested :func:`torchmodal.functional.necessity` floors at a predictable
  depth.
"""

import math
import warnings

import pytest
import torch

from torchmodal import functional as F
from torchmodal import utils


def _chain(T=6, cut=None):
    """A linear chain 0 -> 1 -> ... -> T-1, optionally with an edge cut."""
    A = torch.zeros(T, T)
    A[torch.arange(T - 1), torch.arange(1, T)] = 1.0
    if cut is not None:
        A[cut] = 0.0
    return A


def _until_fixture(T=6, L_phi=0.9):
    """phi true to degree L_phi everywhere; psi true only at the last step."""
    phi = torch.stack([torch.full((T,), L_phi), torch.ones(T)], dim=-1)
    psi = torch.zeros(T, 2)
    psi[T - 1] = 1.0
    return phi, psi


class TestUntilIsInert:
    """`until` reads its accessibility for its size and nothing else."""

    def test_identical_for_any_accessibility(self):
        phi, psi = _until_fixture()
        triu = torch.triu(torch.ones(6, 6))
        zeros = torch.zeros(6, 6)
        assert torch.equal(
            F.until(phi, psi, triu), F.until(phi, psi, zeros)
        )

    def test_identical_after_cutting_an_edge(self):
        """Deleting an edge of the chain does not change the result.

        This is the defect `until_graph` exists to fix: `until` cannot see
        that a path became impassable.
        """
        phi, psi = _until_fixture()
        assert torch.equal(
            F.until(phi, psi, _chain()),
            F.until(phi, psi, _chain(cut=(2, 3))),
        )

    def test_no_autograd_path_to_the_relation(self):
        phi, psi = _until_fixture()
        A = _chain().requires_grad_(True)
        out = F.until(phi, psi, A)
        assert out.grad_fn is None
        assert not out.requires_grad

    def test_identical_for_any_tau(self):
        phi, psi = _until_fixture()
        A = _chain()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assert torch.equal(
                F.until(phi, psi, A, tau=0.1),
                F.until(phi, psi, A, tau=10.0),
            )

    def test_passing_tau_warns(self):
        phi, psi = _until_fixture()
        with pytest.warns(DeprecationWarning, match="unused"):
            F.until(phi, psi, _chain(), tau=0.1)

    def test_omitting_tau_does_not_warn(self):
        phi, psi = _until_fixture()
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            F.until(phi, psi, _chain())

    def test_lukasiewicz_sweep_floors_the_lower_bound(self):
        """The documented decay: 1 - L_phi is lost per step.

        With L_phi = 0.9 over 6 steps the lower bounds are exactly
        0.5, 0.6, ..., 1.0 — the operator, not the data.
        """
        phi, psi = _until_fixture()
        L = F.until(phi, psi, _chain())[:, 0]
        expected = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        assert torch.allclose(L, expected, atol=1e-6)

    def test_long_horizon_reaches_zero(self):
        """Long enough, and the lower bound is dead: 0 with no gradient."""
        T = 20
        phi, psi = _until_fixture(T=T, L_phi=0.9)
        L = F.until(phi, psi, _chain(T))[:, 0]
        assert L[0].item() == pytest.approx(0.0, abs=1e-6)


class TestConvPoolIsNotMonotone:
    """`conv_pool(x, -x)` can decrease when an entry of x increases."""

    def test_raising_a_term_lowers_the_pool(self):
        lo = F.conv_pool(
            torch.tensor([0.0, 1.0]), -torch.tensor([0.0, 1.0]), tau=0.1
        )
        hi = F.conv_pool(
            torch.tensor([0.0, 2.0]), -torch.tensor([0.0, 2.0]), tau=0.1
        )
        assert hi.item() < lo.item()

    def test_gradient_has_a_negative_entry(self):
        x = torch.tensor([0.3, 0.9], requires_grad=True)
        F.conv_pool(x, -x, tau=0.1).backward()
        assert x.grad[1].item() < 0.0
        assert torch.allclose(
            x.grad, torch.tensor([1.0123266, -0.01232643]), atol=1e-6
        )

    def test_matches_the_documented_derivative(self):
        """df/dx_k = w_k (1 - (x_k - f)/tau), negative iff x_k - f > tau."""
        torch.manual_seed(0)
        for _ in range(50):
            n = int(torch.randint(2, 8, (1,)).item())
            tau = float(torch.empty(1).uniform_(0.05, 0.5))
            x = torch.rand(n, dtype=torch.float64, requires_grad=True)
            f = F.conv_pool(x, -x, tau=tau)
            f.backward()
            with torch.no_grad():
                w = torch.softmax(-x / tau, dim=0)
                analytic = w * (1.0 - (x - f) / tau)
            assert torch.allclose(x.grad, analytic, atol=1e-10)


class TestContradictionDeadZone:
    """`contradiction` after a box neuron is inert below the box width."""

    @staticmethod
    def _loss_for_crossing(c, n, tau=0.1):
        A = torch.ones(n, n)
        bounds = torch.stack(
            [torch.full((n,), 0.5 + c / 2), torch.full((n,), 0.5 - c / 2)],
            dim=-1,
        )
        return F.contradiction(F.necessity(bounds, A, tau=tau))

    @pytest.mark.parametrize("n", [3, 6, 10])
    def test_zero_below_the_box_width(self, n):
        tau = 0.1
        width = tau * math.log(n)
        assert self._loss_for_crossing(width * 0.9, n).item() == 0.0

    @pytest.mark.parametrize("n", [3, 6, 10])
    def test_no_gradient_inside_the_dead_zone(self, n):
        tau = 0.1
        c = torch.tensor(tau * math.log(n) * 0.9, requires_grad=True)
        A = torch.ones(n, n)
        bounds = torch.stack(
            [(0.5 + c / 2).expand(n), (0.5 - c / 2).expand(n)], dim=-1
        )
        loss = F.contradiction(F.necessity(bounds, A, tau=tau))
        (grad,) = torch.autograd.grad(loss, c, allow_unused=True)
        assert grad is None or grad.item() == 0.0

    @pytest.mark.parametrize("n", [3, 6, 10])
    def test_dead_zone_edge_equals_box_width_entropy(self, n):
        """The dead zone is *exactly* the box width, not approximately."""
        tau = 0.1
        point = torch.full((n, 2), 0.5)
        width = F.box_width_entropy(
            torch.ones(n, n), point, tau=tau
        )[0].item()

        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if self._loss_for_crossing(mid, n, tau).item() > 0:
                hi = mid
            else:
                lo = mid
        assert hi == pytest.approx(width, abs=1e-6)

    def test_linear_with_unit_slope_per_world_past_the_edge(self):
        n = 6
        a = self._loss_for_crossing(0.6, n).item()
        b = self._loss_for_crossing(0.8, n).item()
        assert (b - a) / 0.2 == pytest.approx(float(n), rel=1e-4)


class TestNestedNecessityFloors:
    """Each box level costs tau*H(w); a deep enough nest is dead."""

    @staticmethod
    def _depths(A, k=6, tau=0.1, n=8):
        bounds = torch.ones(n, 2)
        out = []
        for _ in range(k):
            bounds = F.necessity(bounds, A, tau=tau)
            out.append(bounds[0, 0].item())
        return out

    def test_complete_frame_matches_the_documented_table(self):
        vals = self._depths(torch.ones(8, 8))
        expected = [0.7921, 0.5841, 0.3762, 0.1682, 0.0, 0.0]
        assert vals == pytest.approx(expected, abs=1e-3)

    def test_complete_frame_floors_at_the_predicted_depth(self):
        tau = 0.1
        k_star = math.ceil(1.0 / (tau * math.log(8)))
        vals = self._depths(torch.ones(8, 8), k=k_star)
        assert vals[-1] == pytest.approx(0.0, abs=1e-6)
        assert vals[-2] > 0.0

    @pytest.mark.parametrize(
        "builder, per_level",
        [
            (lambda: torch.ones(8, 8), 0.1 * math.log(8)),
            (lambda: utils.build_ring_accessibility(8), 0.1 * math.log(2)),
            (
                lambda: utils.build_ring_accessibility(8, bidirectional=True),
                0.1 * math.log(3),
            ),
        ],
    )
    def test_per_level_loss_is_the_frames_branching(self, builder, per_level):
        vals = self._depths(builder(), k=3)
        assert (vals[0] - vals[1]) == pytest.approx(per_level, abs=1e-3)

    def test_per_level_loss_equals_box_width_entropy(self):
        A = utils.build_ring_accessibility(8, bidirectional=True)
        width = F.box_width_entropy(A, torch.ones(8, 2), tau=0.1)[0].item()
        vals = self._depths(A, k=3)
        assert (vals[0] - vals[1]) == pytest.approx(width, abs=1e-3)
