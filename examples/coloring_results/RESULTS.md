# Graph k-Coloring Benchmark — MLNN vs baseline matrix

_FULL: 10 planted-3-colorable graphs/tier; MLNN 5 seeds x 2000 epochs._

Proper-coloring solve-rate per method x difficulty:

| Method | Family | Easy | Medium | Hard | Overall | mean s |
|---|---|---|---|---|---|---|
| Backtracking(DSATUR) | exact | 100% | 100% | 100% | **100%** | 0.00 |
| OR-Tools CP-SAT | exact | 100% | 100% | 100% | **100%** | 0.01 |
| Z3 (SMT) | exact | 100% | 100% | 100% | **100%** | 0.01 |
| PycoSAT (SAT) | exact | 100% | 100% | 100% | **100%** | 0.00 |
| python-constraint | exact | 100% | 100% | 100% | **100%** | 0.00 |
| Greedy DSATUR | heuristic | 100% | 100% | 100% | **100%** | 0.00 |
| SimulatedAnnealing | metaheuristic | 100% | 100% | 100% | **100%** | 0.04 |
| Min-Conflicts | metaheuristic | 100% | 100% | 100% | **100%** | 0.25 |
| MLNN (modal, ours) | differentiable | 100% | 90% | 100% | **97%** | 1.64 |
| SemanticLoss | differentiable | 100% | 90% | 70% | **87%** | 0.51 |
| RRN-style GNN | differentiable | 100% | 90% | 100% | **97%** | 2.39 |
| Soft-nonmodal(abl.) | differentiable | 20% | 90% | 70% | **60%** | 0.79 |

**Capability 2 (learnable A):** from 60 valid colorings of a hidden 16-node graph, the learned A_theta recovers the true adjacency at **edge-recovery AUC = 1.000** (the inspectable-accessibility result Sudoku cannot give).

