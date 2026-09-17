"""Tests for torchmodal.functional.until_graph."""

import math

import pytest
import torch

from torchmodal import functional as F
from torchmodal.diagnostics import gradient_health


def _chain(T=6, cut=None):
    A = torch.zeros(T, T)
    A[torch.arange(T - 1), torch.arange(1, T)] = 1.0
    if cut is not None:
        A[cut] = 0.0
    return A


def _fixture(T=6, L_phi=0.9):
    phi = torch.stack([torch.full((T,), L_phi), torch.ones(T)], dim=-1)
    psi = torch.zeros(T, 2)
    psi[T - 1] = 1.0
    return phi, psi


class TestUntilGraphSeesTheRelation:
    """The property that distinguishes it from `until`."""

    def test_intact_chain_does_not_floor(self):
        """Gödel connectives are idempotent: no 1 - L_phi decay per step."""
        phi, psi = _fixture()
        L = F.until_graph(phi, psi, _chain())[:, 0]
        expected = torch.tensor([0.9, 0.9, 0.9, 0.9, 0.9, 1.0])
        assert torch.allclose(L, expected, atol=1e-4)

    def test_cutting_the_path_is_detected(self):
        """Cut 2 -> 3 and worlds 0..2 can no longer reach psi."""
        phi, psi = _fixture()
        L = F.until_graph(phi, psi, _chain(cut=(2, 3)))[:, 0]
        expected = torch.tensor([0.0, 0.0, 0.0, 0.9, 0.9, 1.0])
        assert torch.allclose(L, expected, atol=1e-4)

    def test_gradient_reaches_the_relation(self):
        phi, psi = _fixture()
        A = _chain().requires_grad_(True)
        out = F.until_graph(phi, psi, A)
        (grad,) = torch.autograd.grad(out[0, 0], A)
        assert grad.abs().max().item() == pytest.approx(0.5, abs=1e-3)

    def test_carries_training_signal(self):
        phi, psi = _fixture()
        A = _chain().requires_grad_(True)
        report = gradient_health(
            lambda: F.until_graph(phi, psi, A), {"A": A}
        )
        assert report["healthy"], report["issues"]


class TestUntilGraphSoundness:
    def test_bounds_are_ordered_and_in_range(self):
        torch.manual_seed(0)
        for _ in range(200):
            n = int(torch.randint(2, 8, (1,)).item())
            A = (torch.rand(n, n) > 0.5).float()
            phi = torch.rand(n, 2).sort(dim=1).values
            psi = torch.rand(n, 2).sort(dim=1).values
            out = F.until_graph(phi, psi, A)
            assert (out[:, 0] <= out[:, 1] + 1e-6).all()
            assert (out >= -1e-6).all() and (out <= 1 + 1e-6).all()

    def test_brackets_the_crisp_value_on_a_chain(self):
        """L <= crisp <= U for a Boolean frame and crisp propositions."""
        T = 6
        A = _chain(T)
        phi = torch.ones(T, 2)
        psi = torch.zeros(T, 2)
        psi[T - 1] = 1.0
        out = F.until_graph(phi, psi, A, tau=0.01)
        # Crisp EU: every world reaches the last, so the value is 1.
        assert (out[:, 1] >= 1.0 - 1e-3).all()
        assert (out[:, 0] <= 1.0 + 1e-6).all()

    def test_psi_true_everywhere_is_true_everywhere(self):
        T = 5
        out = F.until_graph(torch.zeros(T, 2), torch.ones(T, 2), _chain(T))
        assert torch.allclose(out, torch.ones(T, 2), atol=1e-6)

    def test_annealing_tightens_the_bounds(self):
        phi, psi = _fixture()
        A = _chain()
        wide = F.until_graph(phi, psi, A, tau=0.3, tau_decay=1.0)
        tight = F.until_graph(phi, psi, A, tau=0.3, tau_decay=0.5)
        width_wide = (wide[:, 1] - wide[:, 0]).sum()
        width_tight = (tight[:, 1] - tight[:, 0]).sum()
        assert width_tight <= width_wide + 1e-6

    def test_gap_is_bounded_by_the_geometric_series(self):
        """Total slack <= tau log|W| / (1 - rho)."""
        T, tau, rho = 6, 0.1, 0.5
        phi = torch.ones(T, 2)
        psi = torch.zeros(T, 2)
        psi[T - 1] = 1.0
        out = F.until_graph(phi, psi, _chain(T), tau=tau, tau_decay=rho)
        budget = tau * math.log(T) / (1 - rho)
        assert (out[:, 1] - out[:, 0]).max().item() <= budget + 1e-6


class TestUntilGraphBoxQuantifier:
    def test_box_is_vacuous_at_a_dead_end(self):
        """AU is unsound on a non-serial frame — measured, not theoretical.

        The chain's last world has no successor, so `box U` is vacuously
        satisfied and the result is 0.9 everywhere no matter how the chain
        is connected.
        """
        phi, psi = _fixture()
        intact = F.until_graph(phi, psi, _chain(), quantifier="box")[:, 0]
        cut = F.until_graph(
            phi, psi, _chain(cut=(2, 3)), quantifier="box"
        )[:, 0]
        assert torch.allclose(intact, cut, atol=1e-6)
        assert torch.allclose(intact[:5], torch.full((5,), 0.9), atol=1e-4)


class TestUntilGraphApi:
    def test_point_valued_round_trip(self):
        phi, psi = _fixture()
        out = F.until_graph(phi[:, 0], psi[:, 0], _chain())
        assert out.shape == (6,)
        assert torch.allclose(
            out, torch.tensor([0.9, 0.9, 0.9, 0.9, 0.9, 1.0]), atol=1e-4
        )

    def test_rejects_unknown_quantifier(self):
        phi, psi = _fixture()
        with pytest.raises(ValueError, match="quantifier"):
            F.until_graph(phi, psi, _chain(), quantifier="sometimes")

    @pytest.mark.parametrize("rho", [0.0, -0.5, 1.5])
    def test_rejects_bad_tau_decay(self, rho):
        phi, psi = _fixture()
        with pytest.raises(ValueError, match="tau_decay"):
            F.until_graph(phi, psi, _chain(), tau_decay=rho)

    def test_converges_before_the_iteration_cap(self):
        phi, psi = _fixture()
        few = F.until_graph(phi, psi, _chain(), max_iter=12)
        many = F.until_graph(phi, psi, _chain(), max_iter=50)
        assert torch.allclose(few, many, atol=1e-6)

    def test_handles_a_cyclic_frame(self):
        """`until` is defined only on a total order; this is not."""
        A = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        )
        phi = torch.ones(3, 2)
        psi = torch.zeros(3, 2)
        psi[2] = 1.0
        out = F.until_graph(phi, psi, A)
        assert (out[:, 0] > 0.5).all()
