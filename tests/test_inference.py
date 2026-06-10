"""Tests for torchmodal.inference."""

import torch

from torchmodal import FormulaGraph, upward_downward


class TestFormulaGraph:
    def test_topological_order(self):
        graph = FormulaGraph()
        graph.add_atomic("p")
        graph.add_atomic("q")
        graph.add_conjunction("p_and_q", "p", "q")
        graph.add_necessity("box_p_and_q", "p_and_q")

        order = graph.topological_order()
        assert order.index("p") < order.index("p_and_q")
        assert order.index("q") < order.index("p_and_q")
        assert order.index("p_and_q") < order.index("box_p_and_q")

    def test_is_acyclic_true(self):
        """A proper formula DAG should be acyclic."""
        graph = FormulaGraph()
        graph.add_atomic("p")
        graph.add_atomic("q")
        graph.add_conjunction("p_and_q", "p", "q")
        graph.add_necessity("box_pq", "p_and_q")
        assert graph.is_acyclic()

    def test_is_acyclic_single_node(self):
        """A single atomic node is trivially acyclic."""
        graph = FormulaGraph()
        graph.add_atomic("p")
        assert graph.is_acyclic()

    def test_until_node(self):
        """FormulaGraph supports Until nodes."""
        graph = FormulaGraph()
        graph.add_atomic("phi")
        graph.add_atomic("psi")
        graph.add_until("phi_until_psi", "phi", "psi")
        order = graph.topological_order()
        assert order.index("phi") < order.index("phi_until_psi")
        assert order.index("psi") < order.index("phi_until_psi")


class TestUpwardDownward:
    def test_tightens_bounds(self):
        """Inference should tighten bounds (not widen)."""
        graph = FormulaGraph()
        graph.add_atomic("p")
        graph.add_atomic("q")
        graph.add_conjunction("p_and_q", "p", "q")

        bounds = {
            "p": torch.tensor([[0.8, 1.0], [0.3, 0.5]]),
            "q": torch.tensor([[0.7, 0.9], [0.6, 0.8]]),
            "p_and_q": torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        }

        A = torch.eye(2)
        result = upward_downward(graph, bounds, A)

        assert result["p_and_q"][0, 0].item() > 0.0
        assert result["p_and_q"][0, 1].item() < 1.0

    def test_modal_inference(self):
        """Test inference with a necessity node."""
        graph = FormulaGraph()
        graph.add_atomic("p")
        graph.add_necessity("box_p", "p")

        bounds = {
            "p": torch.tensor([[0.9, 1.0], [0.1, 0.2], [0.8, 0.9]]),
            "box_p": torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
        }

        A = torch.ones(3, 3)
        result = upward_downward(graph, bounds, A, tau=0.1)

        assert result["box_p"][0, 0].item() < 0.5

    def test_until_inference(self):
        """Test inference with an Until node."""
        graph = FormulaGraph()
        graph.add_atomic("phi")
        graph.add_atomic("psi")
        graph.add_until("phi_U_psi", "phi", "psi")

        bounds = {
            "phi": torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]]),
            "psi": torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]]),
            "phi_U_psi": torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
        }

        A = torch.triu(torch.ones(3, 3))
        result = upward_downward(graph, bounds, A, tau=0.1)

        # ϕ holds at t=0,1 and ψ at t=2 → Until is true at t=0
        assert result["phi_U_psi"][0, 0].item() > 0.5

    def test_converges(self):
        """Should converge within max_iterations."""
        graph = FormulaGraph()
        graph.add_atomic("p")
        graph.add_negation("not_p", "p")

        bounds = {
            "p": torch.tensor([[0.3, 0.7]]),
            "not_p": torch.tensor([[0.0, 1.0]]),
        }

        A = torch.ones(1, 1)
        result = upward_downward(
            graph, bounds, A, max_iterations=20
        )
        assert abs(result["not_p"][0, 0].item() - 0.3) < 0.1
        assert abs(result["not_p"][0, 1].item() - 0.7) < 0.1


class TestInferenceGradientFlow:
    """Verify gradients flow through the inference loop."""

    def test_grad_through_upward_pass(self):
        graph = FormulaGraph()
        graph.add_atomic("p")
        graph.add_atomic("q")
        graph.add_conjunction("p_and_q", "p", "q")

        p_bounds = torch.tensor([[0.8, 1.0], [0.3, 0.5]], requires_grad=True)
        q_bounds = torch.tensor([[0.7, 0.9], [0.6, 0.8]], requires_grad=True)

        bounds = {
            "p": p_bounds,
            "q": q_bounds,
            "p_and_q": torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        }

        A = torch.eye(2)
        result = upward_downward(graph, bounds, A, max_iterations=1)

        loss = result["p_and_q"].sum()
        loss.backward()
        assert p_bounds.grad is not None
        assert q_bounds.grad is not None

    def test_grad_through_modal_inference(self):
        graph = FormulaGraph()
        graph.add_atomic("p")
        graph.add_necessity("box_p", "p")

        p_bounds = torch.tensor(
            [[0.9, 1.0], [0.1, 0.2], [0.8, 0.9]], requires_grad=True
        )

        bounds = {
            "p": p_bounds,
            "box_p": torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]),
        }

        A = torch.ones(3, 3)
        result = upward_downward(graph, bounds, A, tau=0.1, max_iterations=1)

        loss = result["box_p"].sum()
        loss.backward()
        assert p_bounds.grad is not None
        assert not torch.all(p_bounds.grad == 0)

    def test_grad_through_accessibility(self):
        graph = FormulaGraph()
        graph.add_atomic("p")
        graph.add_necessity("box_p", "p")

        bounds = {
            "p": torch.tensor([[0.9, 1.0], [0.1, 0.2]]),
            "box_p": torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        }

        A = torch.ones(2, 2, requires_grad=True)
        result = upward_downward(graph, bounds, A, tau=0.1, max_iterations=1)

        loss = result["box_p"].sum()
        loss.backward()
        assert A.grad is not None
        assert not torch.all(A.grad == 0)
