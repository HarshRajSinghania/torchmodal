# Sudoku Benchmark — MLNN vs baseline matrix

_SMOKE run: 2 puzzles/tier; MLNN 2 seeds x 600 epochs._

Solve-rate (% boards fully solved) per method x difficulty:

| Method | Family | Easy | Medium | Hard | Overall | mean s |
|---|---|---|---|---|---|---|
| Backtracking(MRV) | exact | 100% | 100% | 100% | **100%** | 0.04 |
| OR-Tools CP-SAT | exact | 100% | 100% | 100% | **100%** | 0.32 |
| Z3 (SMT) | exact | 100% | 100% | 100% | **100%** | 0.78 |
| PycoSAT (SAT) | exact | 100% | 100% | 100% | **100%** | 0.14 |
| python-constraint | exact | 100% | 100% | 100% | **100%** | 0.06 |
| Propagation-only | propagation | 100% | 100% | 50% | **83%** | 0.11 |
| SimulatedAnnealing | metaheuristic | 100% | 50% | 0% | **50%** | 6.55 |
| Min-Conflicts | metaheuristic | 100% | 0% | 0% | **33%** | 8.55 |
| MLNN (modal, ours) | differentiable | 100% | 0% | 0% | **33%** | 18.72 |
| SemanticLoss | differentiable | 100% | 100% | 50% | **83%** | 17.31 |
| Soft-nonmodal(abl.) | differentiable | 100% | 50% | 0% | **50%** | 5.02 |

Mean cell accuracy per method x difficulty:

| Method | Easy | Medium | Hard |
|---|---|---|---|
| Backtracking(MRV) | 100.0% | 100.0% | 100.0% |
| OR-Tools CP-SAT | 100.0% | 100.0% | 100.0% |
| Z3 (SMT) | 100.0% | 100.0% | 100.0% |
| PycoSAT (SAT) | 100.0% | 100.0% | 100.0% |
| python-constraint | 100.0% | 100.0% | 100.0% |
| Propagation-only | 100.0% | 100.0% | 71.6% |
| SimulatedAnnealing | 100.0% | 94.4% | 56.8% |
| Min-Conflicts | 100.0% | 90.1% | 68.5% |
| MLNN (modal, ours) | 100.0% | 86.4% | 67.9% |
| SemanticLoss | 100.0% | 100.0% | 76.5% |
| Soft-nonmodal(abl.) | 100.0% | 89.5% | 65.4% |
