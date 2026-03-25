#!/usr/bin/env node
/**
 * Minimal JSON bridge for mathsteps.
 *
 * Input (stdin):
 *   {"problems":[{"kind":"solve_equation"|"simplify_expression","input":"..."}]}
 *
 * Output (stdout):
 *   {"results":[{"ok":true,"steps":[...],"final_state":"..."}, ...]}
 */

const fs = require("fs");
const mathsteps = require("mathsteps");

function stepToRecord(kind, status) {
  if (kind === "solve_equation") {
    if (!status.oldEquation || !status.newEquation) {
      return null;
    }
    return {
      change_type: status.changeType || "UNKNOWN",
      before: status.oldEquation.ascii(),
      after: status.newEquation.ascii(),
    };
  }

  if (!status.oldNode || !status.newNode) {
    return null;
  }
  return {
    change_type: status.changeType || "UNKNOWN",
    before: String(status.oldNode),
    after: String(status.newNode),
  };
}

function collectSteps(kind, statuses, out) {
  for (const status of statuses || []) {
    const record = stepToRecord(kind, status);
    if (record) {
      out.push(record);
    }
    if (status && Array.isArray(status.substeps) && status.substeps.length > 0) {
      collectSteps(kind, status.substeps, out);
    }
  }
}

function solveOne(problem) {
  const kind = problem.kind;
  const input = problem.input;
  if (kind !== "solve_equation" && kind !== "simplify_expression") {
    return { ok: false, error: `Unsupported kind: ${kind}` };
  }

  try {
    const statuses =
      kind === "solve_equation" ? mathsteps.solveEquation(input) : mathsteps.simplifyExpression(input);
    const steps = [];
    collectSteps(kind, statuses, steps);
    const finalState = steps.length > 0 ? steps[steps.length - 1].after : input;
    return {
      ok: true,
      steps,
      final_state: finalState,
    };
  } catch (error) {
    return {
      ok: false,
      error: String(error),
    };
  }
}

function main() {
  const payload = JSON.parse(fs.readFileSync(0, "utf8"));
  const problems = payload.problems || [];
  const results = problems.map((problem) => solveOne(problem));
  process.stdout.write(JSON.stringify({ results }));
}

main();
