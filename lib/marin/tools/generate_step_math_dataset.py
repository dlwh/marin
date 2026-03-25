# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate a deterministic step-by-step math dataset with mechanical solvers.

This tool generates arithmetic-expression simplification and linear-equation solving
examples, extracts transformation steps from the OSS `mathsteps` solver, and verifies
every transition with SymPy before writing JSONL.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    standard_transformations,
    parse_expr,
)

logger = logging.getLogger(__name__)

MATHSTEPS_VERSION = "0.2.0"
TRANSFORMATIONS = (*standard_transformations, implicit_multiplication_application, convert_xor)

SolverKind = Literal["solve_equation", "simplify_expression"]


@dataclass(frozen=True)
class ProblemSpec:
    """Specification for one generated problem."""

    solver_kind: SolverKind
    domain: str
    question: str
    solver_input: str
    metadata: dict[str, Any]


def _parse_math_expression(expression: str) -> sympy.Expr:
    """Parse a mathsteps-style expression into a SymPy expression."""
    return parse_expr(expression, transformations=TRANSFORMATIONS, evaluate=True)


def _parse_equation(equation_text: str) -> sympy.Equality:
    """Parse a text equation with exactly one equals sign."""
    if equation_text.count("=") != 1:
        raise ValueError(f"Expected one '=' in equation: {equation_text}")
    left_text, right_text = equation_text.split("=", maxsplit=1)
    left_expr = _parse_math_expression(left_text.strip())
    right_expr = _parse_math_expression(right_text.strip())
    return sympy.Eq(left_expr, right_expr)


def _canonical_solution_representation(solution_set: sympy.Set) -> tuple[str, ...]:
    """Convert a SymPy solution set to a deterministic tuple representation."""
    if isinstance(solution_set, sympy.FiniteSet):
        return tuple(sorted(str(sympy.simplify(value)) for value in solution_set))
    return (str(solution_set),)


def verify_expression_transition(before: str, after: str) -> tuple[bool, str]:
    """Verify equivalence of two expressions."""
    try:
        before_expr = _parse_math_expression(before)
        after_expr = _parse_math_expression(after)
    except Exception as exc:
        return False, f"expression_parse_error: {exc}"

    try:
        equivalent = sympy.simplify(before_expr - after_expr) == 0
        return (True, "ok") if equivalent else (False, "expressions_not_equivalent")
    except Exception as exc:
        return False, f"expression_compare_error: {exc}"


def verify_equation_transition(before: str, after: str, variable_name: str = "x") -> tuple[bool, str]:
    """Verify that two equations have identical solution sets for one variable."""
    variable = sympy.symbols(variable_name)
    try:
        before_eq = _parse_equation(before)
        after_eq = _parse_equation(after)
    except Exception as exc:
        return False, f"equation_parse_error: {exc}"

    try:
        before_set = sympy.solveset(before_eq, variable, domain=sympy.S.Complexes)
        after_set = sympy.solveset(after_eq, variable, domain=sympy.S.Complexes)
        before_repr = _canonical_solution_representation(before_set)
        after_repr = _canonical_solution_representation(after_set)
        return (True, "ok") if before_repr == after_repr else (False, "solution_sets_differ")
    except Exception as exc:
        return False, f"equation_compare_error: {exc}"


def _format_linear_side(coeff: int, const: int) -> str:
    """Format a linear side as '<coeff>x + <const>' with compact signs."""
    parts: list[str] = []
    if coeff != 0:
        if coeff == 1:
            parts.append("x")
        elif coeff == -1:
            parts.append("-x")
        else:
            parts.append(f"{coeff}x")

    if const != 0:
        const_abs = abs(const)
        if parts:
            sign = "+" if const > 0 else "-"
            parts.append(f"{sign} {const_abs}")
        else:
            parts.append(str(const))

    return " ".join(parts) if parts else "0"


def _generate_expression(rng: np.random.Generator, depth: int) -> str:
    """Generate a random arithmetic expression with safe denominators."""
    if depth <= 0:
        value = int(rng.integers(-12, 13))
        if value == 0:
            value = 1
        return str(value)

    op = str(rng.choice(["+", "-", "*", "/"], p=[0.35, 0.25, 0.3, 0.1]))
    left = _generate_expression(rng, depth - 1)
    if op == "/":
        denominator = int(rng.integers(1, 10))
        return f"({left}) / {denominator}"

    right = _generate_expression(rng, depth - 1)
    return f"({left}) {op} ({right})"


def generate_arithmetic_problem(rng: np.random.Generator) -> ProblemSpec:
    """Generate one arithmetic simplification problem."""
    depth = int(rng.integers(1, 4))
    expression = _generate_expression(rng, depth)
    return ProblemSpec(
        solver_kind="simplify_expression",
        domain="arithmetic",
        question=f"Simplify the expression: {expression}",
        solver_input=expression,
        metadata={"depth": depth},
    )


def generate_linear_equation_problem(rng: np.random.Generator) -> ProblemSpec:
    """Generate one linear equation with an integer ground-truth solution."""
    x_solution = int(rng.integers(-12, 13))

    left_coeff = int(rng.integers(-9, 10))
    while left_coeff == 0:
        left_coeff = int(rng.integers(-9, 10))

    right_coeff = int(rng.integers(-9, 10))
    while right_coeff == left_coeff:
        right_coeff = int(rng.integers(-9, 10))

    left_const = int(rng.integers(-20, 21))
    right_const = (left_coeff - right_coeff) * x_solution + left_const

    left_side = _format_linear_side(left_coeff, left_const)
    right_side = _format_linear_side(right_coeff, right_const)
    equation = f"{left_side} = {right_side}"

    return ProblemSpec(
        solver_kind="solve_equation",
        domain="algebra_linear_equation",
        question=f"Solve for x: {equation}",
        solver_input=equation,
        metadata={"x_solution": x_solution},
    )


def ensure_mathsteps_installed(mathsteps_home: Path, version: str = MATHSTEPS_VERSION) -> None:
    """Install mathsteps into a local cache directory if needed."""
    package_dir = mathsteps_home.expanduser().resolve()
    package_path = package_dir / "node_modules" / "mathsteps"
    if package_path.exists():
        return

    package_dir.mkdir(parents=True, exist_ok=True)
    if not (package_dir / "package.json").exists():
        subprocess.run(["npm", "init", "-y"], cwd=package_dir, check=True, capture_output=True, text=True)

    logger.info("Installing mathsteps@%s into %s", version, package_dir)
    subprocess.run(
        ["npm", "install", f"mathsteps@{version}", "--silent"],
        cwd=package_dir,
        check=True,
        capture_output=True,
        text=True,
    )


def run_mathsteps_batch(
    specs: list[ProblemSpec],
    bridge_path: Path,
    mathsteps_home: Path,
) -> list[dict[str, Any]]:
    """Run mathsteps for a batch of ProblemSpec entries via the Node bridge."""
    payload = {
        "problems": [
            {
                "kind": spec.solver_kind,
                "input": spec.solver_input,
            }
            for spec in specs
        ]
    }

    node_path = (mathsteps_home.expanduser().resolve() / "node_modules").as_posix()
    env = os.environ.copy()
    existing_node_path = env.get("NODE_PATH")
    env["NODE_PATH"] = f"{node_path}{os.pathsep}{existing_node_path}" if existing_node_path else node_path
    process = subprocess.run(
        ["node", bridge_path.as_posix()],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    if process.returncode != 0:
        raise RuntimeError(f"mathsteps bridge failed: {process.stderr.strip()}")

    try:
        data = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"mathsteps bridge returned invalid JSON: {process.stdout}") from exc

    results = data.get("results")
    if not isinstance(results, list):
        raise RuntimeError(f"mathsteps bridge missing results list: {data}")
    return results


def _verify_step(spec: ProblemSpec, step: dict[str, Any]) -> tuple[bool, str]:
    before = str(step["before"])
    after = str(step["after"])
    if spec.solver_kind == "solve_equation":
        return verify_equation_transition(before, after)
    return verify_expression_transition(before, after)


def _canonical_final_answer(spec: ProblemSpec) -> str:
    if spec.solver_kind == "solve_equation":
        variable = sympy.symbols("x")
        equation = _parse_equation(spec.solver_input)
        solution_set = sympy.solveset(equation, variable, domain=sympy.S.Complexes)
        return str(solution_set)

    expression = _parse_math_expression(spec.solver_input)
    return str(sympy.simplify(expression))


def _materialize_record(
    spec: ProblemSpec,
    solver_result: dict[str, Any],
    record_id: str,
    seed: int,
) -> dict[str, Any] | None:
    if not solver_result.get("ok"):
        return None

    raw_steps = solver_result.get("steps", [])
    if not isinstance(raw_steps, list) or not raw_steps:
        return None

    steps: list[dict[str, Any]] = []
    all_steps_valid = True
    for step_index, raw_step in enumerate(raw_steps):
        step_valid, verify_reason = _verify_step(spec, raw_step)
        all_steps_valid = all_steps_valid and step_valid
        before = str(raw_step["before"])
        after = str(raw_step["after"])
        change_type = str(raw_step.get("change_type", "UNKNOWN"))
        steps.append(
            {
                "index": step_index,
                "change_type": change_type,
                "before": before,
                "after": after,
                "text": f"{change_type}: {before} -> {after}",
                "verification_passed": step_valid,
                "verification_reason": verify_reason,
            }
        )

    final_state = str(solver_result.get("final_state", steps[-1]["after"]))
    record = {
        "id": record_id,
        "seed": seed,
        "domain": spec.domain,
        "question": spec.question,
        "solver_kind": spec.solver_kind,
        "solver_input": spec.solver_input,
        "steps": steps,
        "solver_final_state": final_state,
        "canonical_final_answer": _canonical_final_answer(spec),
        "metadata": spec.metadata,
        "verification": {"all_steps_valid": all_steps_valid},
    }
    return record


def build_dataset(
    num_examples: int,
    seed: int,
    algebra_ratio: float,
    batch_size: int,
    max_attempts: int,
    bridge_path: Path,
    mathsteps_home: Path,
    allow_unverified: bool,
) -> list[dict[str, Any]]:
    """Generate and verify a dataset of mathsteps traces."""
    if num_examples <= 0:
        raise ValueError("num_examples must be positive")
    if not 0.0 <= algebra_ratio <= 1.0:
        raise ValueError("algebra_ratio must be in [0, 1]")

    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    attempts = 0

    while len(records) < num_examples and attempts < max_attempts:
        pending_specs: list[ProblemSpec] = []
        while (
            len(pending_specs) < batch_size
            and len(records) + len(pending_specs) < num_examples
            and attempts < max_attempts
        ):
            attempts += 1
            if float(rng.random()) < algebra_ratio:
                pending_specs.append(generate_linear_equation_problem(rng))
            else:
                pending_specs.append(generate_arithmetic_problem(rng))

        if not pending_specs:
            break

        solver_results = run_mathsteps_batch(pending_specs, bridge_path=bridge_path, mathsteps_home=mathsteps_home)
        for spec, solver_result in zip(pending_specs, solver_results, strict=True):
            record_id = f"stepmath_{len(records):08d}"
            record = _materialize_record(spec, solver_result, record_id, seed)
            if record is None:
                continue
            if not allow_unverified and not record["verification"]["all_steps_valid"]:
                continue
            records.append(record)
            if len(records) >= num_examples:
                break

    if len(records) < num_examples:
        raise RuntimeError(
            "Failed to generate requested examples. "
            f"requested={num_examples} produced={len(records)} attempts={attempts}"
        )

    return records


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write records to JSONL or JSONL.GZ depending on output suffix."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".gz":
        with gzip.open(output_path, "wt", encoding="utf-8") as out_file:
            for record in records:
                out_file.write(json.dumps(record, ensure_ascii=True) + "\n")
        return

    with output_path.open("w", encoding="utf-8") as out_file:
        for record in records:
            out_file.write(json.dumps(record, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate step-by-step arithmetic/algebra data from mathsteps.")
    parser.add_argument("--output", required=True, help="Output path (.jsonl or .jsonl.gz)")
    parser.add_argument("--num-examples", type=int, default=1000, help="Number of examples to emit")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--algebra-ratio", type=float, default=0.5, help="Fraction of linear equation examples")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for mathsteps bridge calls")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=20000,
        help="Maximum sampled candidates before failing",
    )
    parser.add_argument(
        "--mathsteps-home",
        type=Path,
        default=Path("~/.cache/marin/mathsteps"),
        help="Directory that stores npm-installed mathsteps",
    )
    parser.add_argument(
        "--bridge-path",
        type=Path,
        default=Path(__file__).with_name("mathsteps_bridge.js"),
        help="Path to Node bridge script",
    )
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Keep examples even if a solver transition fails SymPy verification",
    )
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Do not auto-install mathsteps; fail if it is missing",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    mathsteps_home = args.mathsteps_home.expanduser().resolve()
    bridge_path = args.bridge_path.resolve()
    if not bridge_path.exists():
        raise FileNotFoundError(f"Bridge script not found: {bridge_path}")

    if not args.skip_bootstrap:
        ensure_mathsteps_installed(mathsteps_home, version=MATHSTEPS_VERSION)
    else:
        expected_package = mathsteps_home / "node_modules" / "mathsteps"
        if not expected_package.exists():
            raise FileNotFoundError(
                f"mathsteps not found at {expected_package}. Run without --skip-bootstrap to install it."
            )

    records = build_dataset(
        num_examples=args.num_examples,
        seed=args.seed,
        algebra_ratio=args.algebra_ratio,
        batch_size=args.batch_size,
        max_attempts=args.max_attempts,
        bridge_path=bridge_path,
        mathsteps_home=mathsteps_home,
        allow_unverified=args.allow_unverified,
    )
    output_path = Path(args.output).expanduser().resolve()
    write_jsonl(records, output_path)
    logger.info("Wrote %d examples to %s", len(records), output_path)


if __name__ == "__main__":
    main()
