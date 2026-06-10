"""Tests for torchmodal.systems (higher-level modal logics)."""

import torch

import torchmodal


class TestEpistemicOperator:
    def test_knowledge_true(self):
        """K_a(ϕ) should be high if ϕ is true in all accessible worlds."""
        K = torchmodal.EpistemicOperator(tau=0.1)
        prop = torch.tensor([[0.9, 1.0], [0.85, 0.95], [0.8, 0.9]])
        agent_access = torch.tensor([1.0, 1.0, 1.0])
        result = K(prop, agent_access)
        assert result[0].item() > 0.5

    def test_knowledge_false_if_one_world_false(self):
        K = torchmodal.EpistemicOperator(tau=0.1)
        prop = torch.tensor([[0.9, 1.0], [0.05, 0.1], [0.9, 1.0]])
        agent_access = torch.tensor([1.0, 1.0, 1.0])
        result = K(prop, agent_access)
        assert result[0].item() < 0.5


class TestTemporalOperator:
    def test_globally(self):
        temporal = torchmodal.TemporalOperator(num_steps=3)
        A = temporal.build_forward_accessibility()
        prop = torch.tensor([[0.9, 1.0], [0.85, 0.95], [0.1, 0.2]])
        result = temporal.globally(prop, A)
        assert result[0, 0].item() < 0.5

    def test_finally(self):
        temporal = torchmodal.TemporalOperator(num_steps=3)
        A = temporal.build_forward_accessibility()
        prop = torch.tensor([[0.1, 0.2], [0.1, 0.2], [0.9, 1.0]])
        result = temporal.finally_(prop, A)
        assert result[0, 1].item() > 0.3

    def test_until(self):
        """Until: ϕ holds at steps 0,1 and ψ becomes true at step 2."""
        temporal = torchmodal.TemporalOperator(num_steps=3)
        A = temporal.build_forward_accessibility()
        phi = torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
        psi = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
        result = temporal.until(phi, psi, A)
        assert result[0, 0].item() > 0.5

    def test_until_fails_when_hold_breaks(self):
        """Until fails when ϕ breaks before ψ becomes true."""
        temporal = torchmodal.TemporalOperator(num_steps=3)
        A = temporal.build_forward_accessibility()
        phi = torch.tensor([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        psi = torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
        result = temporal.until(phi, psi, A)
        assert result[0, 0].item() < 0.5


class TestMultiAgentKripke:
    def test_creation(self):
        mk = torchmodal.MultiAgentKripke(
            num_agents=3, num_steps=2
        )
        assert mk.num_states == 6

    def test_epistemic_accessibility(self):
        mk = torchmodal.MultiAgentKripke(
            num_agents=3, num_steps=1
        )
        A = mk.get_epistemic_accessibility()
        assert A.shape == (3, 3)
        assert torch.allclose(A.diagonal(), torch.ones(3))

    def test_K_operator(self):
        mk = torchmodal.MultiAgentKripke(
            num_agents=3, num_steps=1, tau=0.1
        )
        prop = torch.tensor([[0.8, 1.0], [0.7, 0.9], [0.6, 0.8]])
        K_phi = mk.K(prop)
        assert K_phi.shape == (3, 2)


class TestUtils:
    def test_anneal_temperature(self):
        tau = torchmodal.anneal_temperature(
            epoch=0, total_epochs=100, tau_start=2.0, tau_end=0.1
        )
        assert abs(tau - 2.0) < 0.01

        tau = torchmodal.anneal_temperature(
            epoch=99, total_epochs=100, tau_start=2.0, tau_end=0.1
        )
        assert abs(tau - 0.1) < 0.01

    def test_build_sudoku_accessibility(self):
        A = torchmodal.build_sudoku_accessibility(3)
        assert A.shape == (81, 81)
        assert A[0].sum().item() == 20.0

    def test_build_ring_accessibility(self):
        A = torchmodal.build_ring_accessibility(5)
        assert A.shape == (5, 5)
        for i in range(5):
            assert A[i, i].item() == 1.0
            assert A[i, (i + 1) % 5].item() == 1.0
