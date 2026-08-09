# Changelog

All notable changes to `torchmodal` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] — 2026-08-09

This release completes the downward half of `inference.upward_downward`. Before
it, the downward pass covered three of the eight node types and silently skipped
disjunction and every modal operator, so the paper's stated inverse-update rules
did not all correspond to shipped code.

### Added

- **Downward inverse for `DISJUNCTION`** — previously the downward pass had no
  `DISJUNCTION` branch at all, so an asserted disjunction propagated nothing to
  its disjuncts. Adds the Łukasiewicz inverses for `parent = min(1, a + b)`:
  `L_a ← max(L_a, L_parent − U_b)` (clamped at 0) and `U_a ← min(U_a, U_parent)`,
  applied symmetrically. Asserting `a ∨ b` true with `b` known false now pins
  `a` true.
- **Downward inverses for `NECESSITY` and `POSSIBILITY`** — the downward pass
  previously skipped all modal nodes, on the grounds that inverting an
  aggregation over `Ã` has no canonical per-world factorisation. That is true of
  only one endpoint per operator. A universally quantified lower bound
  distributes over the neighbourhood and an existential upper bound caps every
  disjunct, giving two sound, canonical rules:

  ```
  □ϕ:  L_ϕ[w'] ← max( L_ϕ[w'],  max_w ( L_parent[w] − 1 + A[w,w'] ) )
  ♢ϕ:  U_ϕ[w'] ← min( U_ϕ[w'],  min_w ( U_parent[w] + 1 − A[w,w'] ) )
  ```

  The opposite directions (`□` upper, `♢` lower) constrain an aggregate without
  identifying which neighbour realises it, and remain un-inverted. Both rules are
  sound at any temperature, because `smooth_min` under-estimates `min` and
  `smooth_max` over-estimates `max`, and both are inert on masked pairs
  (`A = 0` contributes `L_parent − 1 ≤ 0` and `U_parent + 1 ≥ 1`), so they are
  safe under top-`k` sparsification.
- **Non-convergence `RuntimeWarning`** — `upward_downward` now warns when
  `max_iterations` is exhausted before `convergence_threshold` is met. The
  returned bounds are still sound (every update is a pure tightening), but they
  are not the fixed point, and this previously failed silently.

### Changed

- **Documented why the two passes are iterated rather than run once.** A single
  upward sweep is exact for the upward system alone, and a single downward sweep
  is exact given fixed parent bounds, but the joint fixed point generally needs
  more than one round: the downward pass tightens a leaf the upward pass has
  already consumed, leaving any sibling formula that shares that leaf stale.
  Since shared subformulae are exactly what the downward pass exists for, the
  single-sweep reading of the convergence result does not apply to the combined
  system. `inference.py`'s module docstring now states this and enumerates which
  endpoints each node type inverts.

### Fixed

- **Downward conjunction inverse in `inference.upward_downward`** — the previous
  update `U_child ← min(U_child, U_parent)` is only sound when the sibling's lower
  bound is 1 and could otherwise exclude a child's true value (e.g. `a = 0.9,
  b = 0.2` clamped `U_a` to `0.2`). Replaced with the general Łukasiewicz inverse
  `U_a ← min(U_a, U_parent + 1 − L_b)` (clamped to 1), and added the sound
  lower-bound update `L_child ← max(L_child, L_parent)`, so asserting a conjunction
  true now propagates truth to both conjuncts. Regression tests added.

### Notes for users

- The new rules only ever *tighten* bounds, so any bracket that was sound before
  remains sound. Two consequences are worth knowing about: inference on graphs
  containing disjunctions or modal nodes may now return strictly tighter bounds
  than 0.1.1 did, and an infeasible assertion over a modal node can now drive an
  atomic child to `L > U`. The latter is the intended contradiction signal — it
  is what `functional.contradiction` and `L_contra` consume — but it means leaves
  are no longer guaranteed to satisfy `L ≤ U` after a downward pass.
- Verified by a randomised soundness check over 300 models (every leaf pinned to
  a point value, every compound node left at `[0, 1]`): no downward rule excluded
  a true leaf value.

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
