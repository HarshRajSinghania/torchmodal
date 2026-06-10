"""Tests for torchmodal.functional."""

import warnings

import torch

from torchmodal import functional as F


class TestSmoothMin:
    def test_below_true_min(self):
        """smooth_min must be a lower bound on min."""
        x = torch.tensor([0.3, 0.7, 0.5])
        result = F.smooth_min(x, tau=0.1)
        assert result.item() <= x.min().item() + 1e-6

    def test_converges_to_min(self):
        """As tau -> 0, smooth_min -> min."""
        x = torch.tensor([0.3, 0.7, 0.5])
        result = F.smooth_min(x, tau=0.001)
        assert abs(result.item() - 0.3) < 0.01

    def test_differentiable(self):
        x = torch.tensor([0.3, 0.7, 0.5], requires_grad=True)
        loss = F.smooth_min(x, tau=0.1)
        loss.backward()
        assert x.grad is not None


class TestSmoothMax:
    def test_above_true_max(self):
        """smooth_max must be an upper bound on max."""
        x = torch.tensor([0.3, 0.7, 0.5])
        result = F.smooth_max(x, tau=0.1)
        assert result.item() >= x.max().item() - 1e-6

    def test_converges_to_max(self):
        x = torch.tensor([0.3, 0.7, 0.5])
        result = F.smooth_max(x, tau=0.001)
        assert abs(result.item() - 0.7) < 0.01


class TestLegacyAliases:
    def test_softmin_warns(self):
        x = torch.tensor([0.3, 0.7, 0.5])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = F.softmin(x, tau=0.1)
            assert len(w) == 1
            assert "deprecated" in str(w[0].message).lower()
        expected = F.smooth_min(x, tau=0.1)
        assert torch.allclose(result, expected)

    def test_softmax_warns(self):
        x = torch.tensor([0.3, 0.7, 0.5])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = F.softmax(x, tau=0.1)
            assert len(w) == 1
            assert "deprecated" in str(w[0].message).lower()
        expected = F.smooth_max(x, tau=0.1)
        assert torch.allclose(result, expected)


class TestConvPool:
    def test_with_positive_z_lower_bound_max(self):
        """conv_pool(x, x) is a lower bound on max."""
        x = torch.tensor([0.2, 0.8, 0.5])
        result = F.conv_pool(x, x, tau=0.1)
        assert result.item() <= x.max().item() + 0.01
        assert result.item() >= x.min().item() - 0.01

    def test_with_negative_z_upper_bound_min(self):
        """conv_pool(x, -x) is an upper bound on min."""
        x = torch.tensor([0.2, 0.8, 0.5])
        result = F.conv_pool(x, -x, tau=0.1)
        assert result.item() >= x.min().item() - 0.01


class TestConnectives:
    def test_negation(self):
        x = torch.tensor([0.0, 0.5, 1.0])
        result = F.negation(x)
        expected = torch.tensor([1.0, 0.5, 0.0])
        assert torch.allclose(result, expected)

    def test_conjunction(self):
        a = torch.tensor([1.0, 0.7, 0.3])
        b = torch.tensor([1.0, 0.5, 0.2])
        result = F.conjunction(a, b)
        expected = torch.tensor([1.0, 0.2, 0.0])
        assert torch.allclose(result, expected, atol=1e-6)

    def test_disjunction(self):
        a = torch.tensor([0.0, 0.7, 0.8])
        b = torch.tensor([0.0, 0.5, 0.6])
        result = F.disjunction(a, b)
        expected = torch.tensor([0.0, 1.0, 1.0])
        assert torch.allclose(result, expected, atol=1e-6)

    def test_implication(self):
        result = F.implication(torch.tensor(1.0), torch.tensor(0.0))
        assert abs(result.item()) < 1e-6
        result = F.implication(torch.tensor(0.0), torch.tensor(0.3))
        assert abs(result.item() - 1.0) < 1e-6


class TestNecessity:
    def test_all_true_reflexive(self):
        """□ϕ should be high if ϕ is true in all accessible worlds."""
        prop = torch.tensor([[0.9, 1.0], [0.9, 1.0], [0.9, 1.0]])
        A = torch.eye(3)
        result = F.necessity(prop, A, tau=0.1)
        assert result[:, 0].min().item() > 0.5

    def test_one_false_lowers_box(self):
        """If one accessible world has false ϕ, □ϕ drops."""
        prop = torch.tensor([[0.9, 1.0], [0.1, 0.2], [0.9, 1.0]])
        A = torch.ones(3, 3)
        result = F.necessity(prop, A, tau=0.1)
        assert result[0, 0].item() < 0.5

    def test_point_valued(self):
        """Test with point-valued (1D) input."""
        prop = torch.tensor([0.9, 0.1, 0.8])
        A = torch.eye(3)
        result = F.necessity(prop, A, tau=0.1)
        assert result.shape == (3,)


class TestPossibility:
    def test_one_true_is_enough(self):
        """♢ϕ should be high if at least one accessible world has ϕ true."""
        prop = torch.tensor([[0.1, 0.2], [0.9, 1.0], [0.1, 0.2]])
        A = torch.ones(3, 3)
        result = F.possibility(prop, A, tau=0.1)
        assert result[0, 1].item() > 0.5

    def test_duality_upper_bound(self):
        """♢ϕ upper ≡ ¬□¬ϕ upper — modal duality via logsumexp identity."""
        prop = torch.tensor([[0.7, 0.9], [0.3, 0.5]])
        A = torch.ones(2, 2) * 0.8
        tau = 0.01
        dia = F.possibility(prop, A, tau=tau)
        neg_prop = torch.stack([1.0 - prop[:, 1], 1.0 - prop[:, 0]], dim=-1)
        box_neg = F.necessity(neg_prop, A, tau=tau)
        neg_box_neg = torch.stack(
            [1.0 - box_neg[:, 1], 1.0 - box_neg[:, 0]], dim=-1
        )
        assert torch.allclose(dia[:, 1], neg_box_neg[:, 1], atol=0.05)


class TestUntil:
    def test_goal_already_true(self):
        """If ψ is true at t=0, ϕ U ψ should be true at t=0."""
        phi = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        psi = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        A = torch.triu(torch.ones(3, 3))
        result = F.until(phi, psi, A, tau=0.1)
        assert result[0, 0].item() >= 0.99

    def test_hold_then_goal(self):
        """ϕ holds for steps 0,1 and ψ becomes true at step 2."""
        phi = torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
        psi = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
        A = torch.triu(torch.ones(3, 3))
        result = F.until(phi, psi, A, tau=0.1)
        assert result[0, 0].item() >= 0.99

    def test_hold_fails_before_goal(self):
        """ϕ fails at step 1 but ψ isn't true until step 2 → Until fails."""
        phi = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        psi = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
        A = torch.triu(torch.ones(3, 3))
        result = F.until(phi, psi, A, tau=0.1)
        assert result[0, 0].item() < 0.5

    def test_point_valued(self):
        """Test with point-valued (1D) input."""
        phi = torch.tensor([1.0, 1.0, 0.0])
        psi = torch.tensor([0.0, 0.0, 1.0])
        A = torch.triu(torch.ones(3, 3))
        result = F.until(phi, psi, A, tau=0.1)
        assert result.shape == (3,)

    def test_differentiable(self):
        """Gradients flow through the Until operator."""
        phi = torch.tensor([[0.8, 0.9], [0.7, 0.8]], requires_grad=True)
        psi = torch.tensor([[0.1, 0.2], [0.9, 1.0]], requires_grad=True)
        A = torch.triu(torch.ones(2, 2))
        result = F.until(phi, psi, A, tau=0.1)
        loss = result.sum()
        loss.backward()
        assert phi.grad is not None
        assert psi.grad is not None


class TestContradiction:
    def test_no_contradiction(self):
        bounds = torch.tensor([[0.3, 0.7], [0.5, 0.9]])
        assert F.contradiction(bounds).item() == 0.0

    def test_has_contradiction(self):
        bounds = torch.tensor([[0.8, 0.3], [0.5, 0.9]])
        assert F.contradiction(bounds).item() > 0.0
        assert abs(F.contradiction(bounds).item() - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# Gradient flow tests
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Verify gradients flow correctly through all operators."""

    def test_necessity_grad_wrt_prop_bounds(self):
        prop = torch.tensor([[0.7, 0.9], [0.3, 0.5]], requires_grad=True)
        A = torch.ones(2, 2)
        result = F.necessity(prop, A, tau=0.1)
        loss = result.sum()
        loss.backward()
        assert prop.grad is not None
        assert not torch.all(prop.grad == 0)

    def test_necessity_grad_wrt_accessibility(self):
        prop = torch.tensor([[0.7, 0.9], [0.3, 0.5]])
        A = torch.ones(2, 2, requires_grad=True)
        result = F.necessity(prop, A, tau=0.1)
        loss = result.sum()
        loss.backward()
        assert A.grad is not None
        assert not torch.all(A.grad == 0)

    def test_possibility_grad_wrt_prop_bounds(self):
        prop = torch.tensor([[0.7, 0.9], [0.3, 0.5]], requires_grad=True)
        A = torch.ones(2, 2)
        result = F.possibility(prop, A, tau=0.1)
        loss = result.sum()
        loss.backward()
        assert prop.grad is not None
        assert not torch.all(prop.grad == 0)

    def test_possibility_grad_wrt_accessibility(self):
        prop = torch.tensor([[0.7, 0.9], [0.3, 0.5]])
        A = torch.ones(2, 2, requires_grad=True)
        result = F.possibility(prop, A, tau=0.1)
        loss = result.sum()
        loss.backward()
        assert A.grad is not None
        assert not torch.all(A.grad == 0)

    def test_conjunction_grad(self):
        a = torch.tensor([0.8], requires_grad=True)
        b = torch.tensor([0.7], requires_grad=True)
        result = F.conjunction(a, b)
        result.sum().backward()
        assert a.grad is not None
        assert b.grad is not None

    def test_implication_grad(self):
        a = torch.tensor([0.8], requires_grad=True)
        b = torch.tensor([0.3], requires_grad=True)
        result = F.implication(a, b)
        result.sum().backward()
        assert a.grad is not None
        assert b.grad is not None

    def test_contradiction_grad(self):
        bounds = torch.tensor([[0.8, 0.3]], requires_grad=True)
        loss = F.contradiction(bounds)
        loss.backward()
        assert bounds.grad is not None
        assert not torch.all(bounds.grad == 0)

    def test_necessity_no_vanishing_grad(self):
        prop = torch.tensor(
            [[0.6, 0.8], [0.4, 0.6], [0.7, 0.9]], requires_grad=True
        )
        A = torch.ones(3, 3)
        result = F.necessity(prop, A, tau=0.5)
        loss = result.sum()
        loss.backward()
        assert prop.grad.abs().max().item() > 1e-4

    def test_possibility_no_vanishing_grad(self):
        prop = torch.tensor(
            [[0.6, 0.8], [0.4, 0.6], [0.7, 0.9]], requires_grad=True
        )
        A = torch.ones(3, 3)
        result = F.possibility(prop, A, tau=0.5)
        loss = result.sum()
        loss.backward()
        assert prop.grad.abs().max().item() > 1e-4
