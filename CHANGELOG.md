# Changelog

All notable changes to `torchmodal` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.1] — 2026-06-09

### Added

- **`functional.until(phi, psi, R)`** — the temporal *Until* operator
  (`U_t = psi_t OR (phi_t AND U_{t+1})`), computed as a backward dynamic-programming
  sweep over the temporal accessibility relation. Exposed on formula graphs via
  `FormulaGraph.add_until(...)` and in `systems.TemporalOperator` (which now covers
  G, F, **and U**).
- **`losses.SemanticLoss`** — the Semantic Loss baseline (Xu et al., 2018) for
  comparing MLNNs against non-modal neurosymbolic constraints, including
  **`forward_mutual_exclusive(probs)`** for one-hot / mutual-exclusion constraints
  (e.g. "each node takes exactly one colour"). Note: the PyPI `0.1.0.6` build shipped
  a partial `SemanticLoss` without `forward_mutual_exclusive`; code calling that
  method against `0.1.0.x` fails with `AttributeError`.
- **`nn.AttentionAccessibility`** — attention-based accessibility head
  (`O(d^2)` parameters) for rich per-world features and asymmetric relations,
  alongside the existing Fixed / Learnable / Metric heads.
- **`FormulaGraph.is_acyclic()`** — cycle detection guarding the DAG invariant
  required by the convergence theorem; call before running inference on
  user-constructed graphs.
- `functional.contradiction(bounds, upper=None)` now accepts a separate upper-bound
  tensor in addition to the packed `[L, U]` form.
- **Examples**: `graph_coloring_benchmark.py` (12-method solver comparison on
  planted-colourable graphs + inductive constraint-graph recovery),
  `sudoku_benchmark.py`, `baseline_comparison.py`, `regen_coloring_figs.py`, and the
  `MLNN_AccesbilityScalabilityAblation.ipynb` notebook (dense-vs-metric accessibility
  sweep, N = 20 to 20,000 worlds).

### Changed

- **Aggregator naming**: `functional.softmin` / `functional.softmax` are renamed to
  **`smooth_min` / `smooth_max`** to avoid confusion with the standard
  probability-normalizing `torch.softmax` (used internally by `conv_pool`).
  The `nn.Softmin` / `nn.Softmax` modules are likewise renamed to
  **`nn.SmoothMin` / `nn.SmoothMax`**. The old names remain available as
  deprecated aliases (functions and module factories), so existing code keeps
  working.
- All example scripts now pin the local package onto `sys.path` before
  `import torchmodal`, so running them from a checkout always uses that checkout
  rather than a previously installed release.
- Expanded docstrings and module documentation throughout
  (`functional`, `inference`, `kripke`, `losses`, `systems`, `nn`).

## [0.1.0.x] — 2025

- Initial public releases: differentiable necessity / possibility neurons with
  sound `[L, U]` bounds, Łukasiewicz connectives, learnable accessibility
  (direct, metric), upward–downward inference, contradiction loss, epistemic /
  doxastic / temporal / multi-agent operators.
- Bugfix: necessity neuron `[L, U]` bound computation used `p` where the bound
  tensor was intended (thanks Noor Naddour).
