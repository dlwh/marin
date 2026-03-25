# Synthetic Reasoning Bootstrap: Research Logbook

## Thread
- Issue: https://github.com/marin-community/marin/issues/4148
- Branch: `codex/research-synthetic-reasoning-bootstrap`

## Scope
- Goal: Build a scalable, fully mechanical synthetic reasoning data pipeline (no LLM labels) across many domains and emit OpenAI chat-format training data.
- Primary metric(s):
  - Verified examples emitted per domain.
  - Verification pass rate per domain.
  - Total dataset size and domain balance.
- Constraints:
  - Deterministic generation from seeds.
  - Every example must have a mechanical verifier.
  - Prefer compact, automation-friendly JSONL.GZ artifacts.

## Baseline
- Date: 2026-02-22
- Code refs:
  - `/Users/dlwh/.codex/worktrees/ea86/marin/lib/marin/tools/generate_step_math_dataset.py`
  - `/Users/dlwh/.codex/worktrees/ea86/marin/lib/marin/tools/linearize_step_math_to_chat.py`
- Baseline numbers:
  - One-domain step-math generator validated on 300 examples.
  - OAI conversion validated on the same 300 examples.

## Kickoff
- Motivation:
  - Establish a non-distilled reasoning bootstrap corpus with strict ground-truth provenance.
- Problem statement:
  - Current pipeline is narrow (mostly arithmetic/algebra). We need 10-20+ mechanical domains with unified schema and scalable generation.
- Success criteria:
  - 10-20 domain solvers implemented.
  - ~1,000 verified examples per domain emitted to OAI format.
  - Standardized schema + manifest for scalable multi-million expansion.
- Stop criteria:
  - At least 10 domains with passing verification and generated artifacts.
  - Manifested outputs with reproducible commands and seed controls.

## Domain Inventory
- Candidate domains discussed:
  - algebra/arithmetic rewrites
  - graph shortest paths (BFS/Dijkstra)
  - topological sort / connected components
  - dynamic programming (LCS/edit distance/coin change/knapsack)
  - SAT/propositional logic
  - symbolic execution (future)
  - automata transformations (future)
  - sorting/search algorithms
- First-wave domains selected:
  - `euclid_gcd`
  - `binary_search`
  - `insertion_sort`
  - `bfs_shortest_path`
  - `dijkstra_shortest_path`
  - `topological_sort`
  - `connected_components`
  - `lcs_dp`
  - `edit_distance_dp`
  - `coin_change_dp`
  - `knapsack_01_dp`
  - `parentheses_balance`
  - `propositional_entailment`
  - `lis_dp`
  - `interval_scheduling`
  - `prim_mst`
  - `floyd_warshall_apsp`
  - `kmp_string_search`
  - `union_find_connectivity`
  - `n_queens_backtracking`

## Prior Work
- [The CLRS Algorithmic Reasoning Benchmark](https://arxiv.org/abs/2205.15659): standardized algorithmic task generator with intermediate algorithm traces.
- [CLRS repository](https://github.com/google-deepmind/clrs): codebase with dataset generation hooks, probes, and extension points for new algorithms.
- [The CLRS-Text Algorithmic Reasoning Language Benchmark](https://arxiv.org/abs/2406.04229): text rendering of CLRS traces, close to our trace-to-language objective.
- [Chain of Execution Supervision Promotes General Reasoning in LLMs](https://arxiv.org/abs/2510.23629): execution-trace-grounded rationale generation at larger scale.
- [VeriCoT: Neuro-symbolic Chain-of-Thought Validation](https://arxiv.org/abs/2511.04662): verification-oriented reasoning pipeline relevant for step validity checks.
- [Mathsteps](https://github.com/google/mathsteps): OSS symbolic algebra step solver (archived) useful for deterministic math traces.

## Experiment IDs
- Prefix: `SYNR`
- Initial matrix:
  - `SYNR-001`: Implement standardized framework + first-wave domains.
  - `SYNR-002`: Generate 1k/domain and emit canonical + OAI datasets.
  - `SYNR-003`: Validate counts/pass rates and produce manifest.
  - `SYNR-004`: Expand to 20 domains and regenerate 1k/domain.

## Experiment Log
### 2026-02-22 23:05 - SYNR-001 kickoff
- Hypothesis:
  - A shared domain registry + verifier interface can support 10+ domains without ad-hoc scripts.
- Command:
  - N/A (implementation phase).
- Config:
  - Seeded deterministic generation, per-domain outputs, OAI linearization.
- Result:
  - In progress.
- Interpretation:
  - Framework will be used as base for scaling to millions.
- Next action:
  - Implement solvers + run SYNR-002 generation.

### 2026-02-22 23:10 - SYNR-001 framework implementation
- Hypothesis:
  - One shared tool with a registry + common schema can support many domains and remain scalable.
- Command:
  - `uv run --package marin pytest tests/tools/test_generate_synthetic_reasoning_dataset.py tests/tools/test_generate_step_math_dataset.py tests/tools/test_linearize_step_math_to_chat.py -q`
- Config:
  - 13 domain solvers in a single generator with canonical + OAI outputs and manifest.
- Result:
  - Tests passed (`8 passed`).
- Interpretation:
  - Framework is stable enough for first large generation run.
- Next action:
  - Run 1,000 examples/domain across all domains (SYNR-002).

### 2026-02-22 23:11 - SYNR-002 first large generation
- Hypothesis:
  - The framework can emit balanced multi-domain data at requested scale with full mechanical verification.
- Command:
  - `uv run python /Users/dlwh/.codex/worktrees/ea86/marin/lib/marin/tools/generate_synthetic_reasoning_dataset.py --output-dir /Users/dlwh/.codex/worktrees/ea86/marin/synthetic_reasoning_v1_13x1000_seed42 --domains all --examples-per-domain 1000 --seed 42 --shard-size 1000 --max-attempt-multiplier 30`
- Config:
  - Domains: 13
  - Examples/domain: 1,000
  - Total target: 13,000
  - Operation names omitted in OAI responses by default
- Result:
  - 13,000 / 13,000 emitted.
  - Verification pass rate: 1.0 for each domain.
  - Output path: `/Users/dlwh/.codex/worktrees/ea86/marin/synthetic_reasoning_v1_13x1000_seed42`
  - Output size: ~5.8M compressed.
- Interpretation:
  - First-wave multi-domain mechanical reasoning dataset generation is complete and reproducible.
- Next action:
  - Promote to larger-scale sharded runs and add additional domains (symbolic execution DSL, automata, SAT/CNF).

### 2026-02-22 23:49 - SYNR-004 domain expansion to 20 domains
- Hypothesis:
  - Adding seven additional algorithmic domains should remain fully verifiable and keep generation stable.
- Command:
  - `uv run python /Users/dlwh/.codex/worktrees/ea86/marin/lib/marin/tools/generate_synthetic_reasoning_dataset.py --output-dir /Users/dlwh/.codex/worktrees/ea86/marin/synthetic_reasoning_v1_20x1000_seed42 --domains all --examples-per-domain 1000 --seed 42 --shard-size 1000 --max-attempt-multiplier 40`
- Config:
  - Domains: 20
  - Examples/domain: 1,000
  - Total target: 20,000
- Result:
  - 20,000 / 20,000 emitted.
  - Verification pass rate: 1.0 for each domain.
  - Output path: `/Users/dlwh/.codex/worktrees/ea86/marin/synthetic_reasoning_v1_20x1000_seed42`
  - Output size: ~8.8M compressed.
- Interpretation:
  - The standardized framework scales cleanly from 13 to 20 domains without manual per-domain orchestration.
- Next action:
  - Add symbolic execution DSL + CNF SAT proof domains and launch million-scale sharded sweeps.

### 2026-02-22 23:56 - SYNR-005 CLRS-style profile mapping
- Hypothesis:
  - CLRS-like task aliases can be mapped onto our native solver set while preserving schema compatibility for corpus mixing.
- Command:
  - `uv run python /Users/dlwh/.codex/worktrees/ea86/marin/lib/marin/tools/generate_synthetic_reasoning_dataset.py --output-dir /Users/dlwh/.codex/worktrees/ea86/marin/synthetic_reasoning_clrs_style_10x1000_seed42 --profile clrs_style --domains all --examples-per-domain 1000 --seed 42 --shard-size 1000 --max-attempt-multiplier 40`
- Config:
  - Profile: `clrs_style`
  - Tasks: 10 CLRS-like aliases mapped to native solvers
  - Examples/task: 1,000
- Result:
  - 10,000 / 10,000 emitted.
  - Output path: `/Users/dlwh/.codex/worktrees/ea86/marin/synthetic_reasoning_clrs_style_10x1000_seed42`
  - OAI rows include both alias domain (`clrs_*`) and backing `solver_domain`.
- Interpretation:
  - CLRS-style subset is now first-class and immediately mixable with native corpora.
- Next action:
  - Add merge utility/manifests for weighted mixing across native + clrs_style + step-math.
