# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate a multi-domain, fully mechanical synthetic reasoning dataset.

This tool is intended for large-scale generation. It produces:
- canonical records with structured steps and verification metadata
- OpenAI chat-format rows (`messages`) derived deterministically from canonical rows
- a manifest with per-domain counts, pass rates, and output shard paths
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import itertools
import json
import math
import zlib
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclasses.dataclass(frozen=True)
class StepRecord:
    """A single deterministic transition."""

    operation: str
    before: Any
    after: Any
    details: str


@dataclasses.dataclass(frozen=True)
class ExampleRecord:
    """Canonical example emitted by one domain solver."""

    prompt: str
    problem: dict[str, Any]
    steps: list[StepRecord]
    final_answer: Any
    difficulty: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class DomainJob:
    """Resolved output domain + backing solver pairing."""

    output_domain: str
    solver_domain: str
    profile: str
    profile_task: str | None = None


class DomainSolver(ABC):
    """Base class for all synthetic reasoning domains."""

    name: str

    @abstractmethod
    def generate(self, rng: np.random.Generator) -> ExampleRecord:
        """Generate one solved example with intermediate steps."""

    @abstractmethod
    def verify(self, example: ExampleRecord) -> tuple[bool, str]:
        """Verify final answer (and optionally step consistency)."""


def _step(operation: str, before: Any, after: Any, details: str) -> StepRecord:
    return StepRecord(operation=operation, before=before, after=after, details=details)


def _format_state(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _pick_variant(options: list[str], *parts: str) -> str:
    key = "|".join(parts)
    idx = int(zlib.crc32(key.encode("utf-8")) % len(options))
    return options[idx]


def _summarize_value(value: Any, depth: int = 0, max_items: int = 5) -> str:
    """Compact, deterministic state summary for NL adapters."""
    if depth >= 2:
        return "..."
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda kv: str(kv[0]))
        parts = [f"{k}={_summarize_value(v, depth + 1, max_items=max_items)}" for k, v in items[:max_items]]
        suffix = ", ..." if len(items) > max_items else ""
        return "{ " + ", ".join(parts) + suffix + " }"
    if isinstance(value, list):
        shown = [_summarize_value(v, depth + 1, max_items=max_items) for v in value[:max_items]]
        suffix = ", ..." if len(value) > max_items else ""
        return "[" + ", ".join(shown) + suffix + "]"
    if isinstance(value, tuple):
        shown = [_summarize_value(v, depth + 1, max_items=max_items) for v in value[:max_items]]
        suffix = ", ..." if len(value) > max_items else ""
        return "(" + ", ".join(shown) + suffix + ")"
    if value is None:
        return "None"
    return str(value)


OPERATION_LABELS: dict[str, str] = {
    "EUCLID_MOD": "apply Euclid's remainder step",
    "COMPARE": "compare against the target",
    "INSERTION_PASS": "insert the current key",
    "POP_NODE": "expand the frontier node",
    "POP_MIN": "extract the current minimum-distance node",
    "KAHN_POP": "remove a zero-indegree node",
    "NEW_COMPONENT": "start and finish one component traversal",
    "DP_CELL": "update one DP cell",
    "DP_AMOUNT": "update minimum coins for this amount",
    "DP_INDEX": "update one DP index state",
    "SCAN_CHAR": "update running parenthesis balance",
    "TRUTH_TABLE_ROW": "evaluate one truth-table assignment",
    "GREEDY_DECISION": "apply the greedy acceptance rule",
    "PRIM_ADD_EDGE": "add the minimum crossing edge",
    "FW_UPDATE": "improve one all-pairs distance entry",
    "LPS_MATCH": "extend the prefix table match",
    "LPS_FALLBACK": "fallback in the prefix table",
    "LPS_ZERO": "record zero prefix match",
    "SCAN_MATCH": "advance matching pointers",
    "PATTERN_FOUND": "confirm a full pattern match",
    "SCAN_FALLBACK": "fallback pattern pointer via LPS",
    "SCAN_SHIFT": "shift text pointer forward",
    "UNION": "merge two disjoint sets",
    "QUERY": "answer connectivity query",
    "PLACE_QUEEN": "place a queen safely",
    "BACKTRACK": "undo a placement and backtrack",
    "TRY_POSITION": "reject an unsafe queen position",
}


def _operation_phrase(operation: str) -> str:
    return OPERATION_LABELS.get(operation, operation.replace("_", " ").lower())


def _render_step_symbolic(
    step: StepRecord,
    *,
    include_operation_names: bool,
) -> str:
    if include_operation_names:
        return f"[{step.operation}] {_format_state(step.before)} -> {_format_state(step.after)}"
    return f"{_format_state(step.before)} -> {_format_state(step.after)}"


def _render_step_nl(
    *,
    step: StepRecord,
    step_index: int,
    domain_name: str,
    row_id: str,
) -> str:
    action = _operation_phrase(step.operation)
    lead = _pick_variant(
        [
            "Next, we",
            "Then we",
            "At this step, we",
            "Now we",
        ],
        row_id,
        domain_name,
        step.operation,
        str(step_index),
    )
    connector = _pick_variant(
        [
            "so",
            "therefore",
            "which gives",
            "yielding",
        ],
        row_id,
        "connector",
        step.operation,
        str(step_index),
    )
    before_s = _summarize_value(step.before)
    after_s = _summarize_value(step.after)
    detail = step.details.strip()
    detail_clause = f" ({detail})." if detail else "."
    return f"{lead} {action}: from {before_s}, {connector} {after_s}{detail_clause}"


def _render_step_hybrid(
    *,
    step: StepRecord,
    step_index: int,
    domain_name: str,
    row_id: str,
    include_operation_names: bool,
) -> str:
    nl = _render_step_nl(step=step, step_index=step_index, domain_name=domain_name, row_id=row_id)
    symbolic = _render_step_symbolic(step, include_operation_names=include_operation_names)
    return f"{nl} Symbolic: {symbolic}"


def _render_assistant_response(
    *,
    example: ExampleRecord,
    include_operation_names: bool,
    assistant_style: str,
    domain_name: str,
    row_id: str,
) -> str:
    if assistant_style == "symbolic":
        title = "Step-by-step solution:"
    elif assistant_style == "nl_template":
        title = _pick_variant(
            [
                "Reasoning walkthrough:",
                "Step-by-step reasoning:",
                "Solution reasoning:",
            ],
            row_id,
            domain_name,
            "title",
        )
    elif assistant_style == "hybrid":
        title = "Step-by-step reasoning (natural language + symbolic):"
    else:
        raise ValueError(f"Unknown assistant_style: {assistant_style}")

    lines = [title]
    for idx, step in enumerate(example.steps, start=1):
        if assistant_style == "symbolic":
            body = _render_step_symbolic(step, include_operation_names=include_operation_names)
        elif assistant_style == "nl_template":
            body = _render_step_nl(step=step, step_index=idx, domain_name=domain_name, row_id=row_id)
        else:
            body = _render_step_hybrid(
                step=step,
                step_index=idx,
                domain_name=domain_name,
                row_id=row_id,
                include_operation_names=include_operation_names,
            )
        lines.append(f"{idx}. {body}")

    final_prefix = _pick_variant(
        ["Final answer", "Therefore, the answer", "So the final result"],
        row_id,
        domain_name,
        "final",
    )
    lines.append(f"{final_prefix}: {_format_state(example.final_answer)}")
    return "\n".join(lines)


def _is_subsequence(candidate: str, target: str) -> bool:
    idx = 0
    for ch in target:
        if idx < len(candidate) and candidate[idx] == ch:
            idx += 1
    return idx == len(candidate)


def _build_undirected_graph(
    rng: np.random.Generator,
    n: int,
    extra_edge_prob: float = 0.25,
) -> list[tuple[int, int]]:
    """Build a random connected undirected graph via spanning tree + extra edges."""
    edges: set[tuple[int, int]] = set()
    for node in range(1, n):
        parent = int(rng.integers(0, node))
        edges.add((min(node, parent), max(node, parent)))

    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in edges and float(rng.random()) < extra_edge_prob:
                edges.add((u, v))
    return sorted(edges)


def _adjacency_from_undirected(n: int, edges: list[tuple[int, int]]) -> dict[int, list[int]]:
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for node in adj:
        adj[node].sort()
    return adj


def _adjacency_weighted_from_undirected(
    n: int, weighted_edges: list[tuple[int, int, int]]
) -> dict[int, list[tuple[int, int]]]:
    adj: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n)}
    for u, v, w in weighted_edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
    for node in adj:
        adj[node].sort(key=lambda x: (x[0], x[1]))
    return adj


def _path_from_parent(parent: dict[int, int | None], src: int, dst: int) -> list[int] | None:
    if dst not in parent:
        return None
    path = [dst]
    cur = dst
    while cur != src:
        nxt = parent.get(cur)
        if nxt is None:
            return None
        path.append(nxt)
        cur = nxt
    path.reverse()
    return path


class EuclidGcdDomain(DomainSolver):
    name = "euclid_gcd"

    def generate(self, rng: np.random.Generator) -> ExampleRecord:
        a = int(rng.integers(2, 3000))
        b = int(rng.integers(2, 3000))
        x, y = max(a, b), min(a, b)
        steps: list[StepRecord] = []
        while y != 0:
            q, r = divmod(x, y)
            steps.append(
                _step(
                    "EUCLID_MOD",
                    {"x": x, "y": y},
                    {"x": y, "y": r},
                    f"{x} = {q}*{y} + {r}",
                )
            )
            x, y = y, r

        prompt = f"Use Euclid's algorithm to compute gcd({a}, {b})."
        return ExampleRecord(
            prompt=prompt,
            problem={"a": a, "b": b},
            steps=steps,
            final_answer=x,
            difficulty={"max_value": max(a, b)},
        )

    def verify(self, example: ExampleRecord) -> tuple[bool, str]:
        a = int(example.problem["a"])
        b = int(example.problem["b"])
        expected = math.gcd(a, b)
        if int(example.final_answer) != expected:
            return False, "gcd_mismatch"
        return True, "ok"


class BinarySearchDomain(DomainSolver):
    name = "binary_search"

    def generate(self, rng: np.random.Generator) -> ExampleRecord:
        n = int(rng.integers(8, 21))
        universe = np.arange(-80, 81)
        arr = sorted(int(x) for x in rng.choice(universe, size=n, replace=False))
        choose_present = float(rng.random()) < 0.7
        if choose_present:
            target = int(arr[int(rng.integers(0, n))])
        else:
            missing = sorted(set(range(-100, 101)) - set(arr))
            target = int(missing[int(rng.integers(0, len(missing)))])

        lo, hi = 0, n - 1
        steps: list[StepRecord] = []
        answer = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            val = arr[mid]
            before = {"lo": lo, "hi": hi, "mid": mid, "arr[mid]": val}
            if val == target:
                answer = mid
                after = {"result": "found", "index": mid}
                steps.append(_step("COMPARE", before, after, "target found"))
                break
            if val < target:
                lo = mid + 1
                after = {"result": "search_right", "new_lo": lo, "hi": hi}
                steps.append(_step("COMPARE", before, after, "arr[mid] < target"))
            else:
                hi = mid - 1
                after = {"result": "search_left", "lo": lo, "new_hi": hi}
                steps.append(_step("COMPARE", before, after, "arr[mid] > target"))

        if answer == -1:
            steps.append(_step("TERMINATE", {"lo": lo, "hi": hi}, {"index": -1}, "interval exhausted"))

        prompt = f"Given sorted array {arr}, find the index of {target} using binary search. Return -1 if absent."
        return ExampleRecord(
            prompt=prompt,
            problem={"array": arr, "target": target},
            steps=steps,
            final_answer=answer,
            difficulty={"n": n},
        )

    def verify(self, example: ExampleRecord) -> tuple[bool, str]:
        arr = [int(x) for x in example.problem["array"]]
        target = int(example.problem["target"])
        expected = arr.index(target) if target in arr else -1
        if int(example.final_answer) != expected:
            return False, "index_mismatch"
        return True, "ok"


class InsertionSortDomain(DomainSolver):
    name = "insertion_sort"

    def generate(self, rng: np.random.Generator) -> ExampleRecord:
        n = int(rng.integers(6, 13))
        values = [int(x) for x in rng.integers(-60, 61, size=n)]
        arr = list(values)
        steps: list[StepRecord] = []

        for i in range(1, len(arr)):
            key = arr[i]
            before_arr = list(arr)
            j = i - 1
            while j >= 0 and arr[j] > key:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
            after_arr = list(arr)
            steps.append(
                _step(
                    "INSERTION_PASS",
                    {"i": i, "array": before_arr},
                    {"i": i, "array": after_arr},
                    f"insert key={key}",
                )
            )

        prompt = f"Sort the array using insertion sort: {values}"
        return ExampleRecord(
            prompt=prompt,
            problem={"array": values},
            steps=steps,
            final_answer=arr,
            difficulty={"n": n},
        )

    def verify(self, example: ExampleRecord) -> tuple[bool, str]:
        original = [int(x) for x in example.problem["array"]]
        expected = sorted(original)
        if [int(x) for x in example.final_answer] != expected:
            return False, "sorted_array_mismatch"
        return True, "ok"


class BfsShortestPathDomain(DomainSolver):
    name = "bfs_shortest_path"

    def generate(self, rng: np.random.Generator) -> ExampleRecord:
        n = int(rng.integers(6, 11))
        edges = _build_undirected_graph(rng, n, extra_edge_prob=0.23)
        adj = _adjacency_from_undirected(n, edges)
        src = int(rng.integers(0, n))
        dst = int(rng.integers(0, n))
        while dst == src:
            dst = int(rng.integers(0, n))

        queue = deque([src])
        parent: dict[int, int | None] = {src: None}
        steps: list[StepRecord] = []

        while queue:
            node = queue.popleft()
            before = {"queue": list(queue), "node": node}
            if node == dst:
                steps.append(_step("POP_NODE", before, {"queue": list(queue), "status": "target_found"}, "stop"))
                break

            newly_discovered: list[int] = []
            for nei in adj[node]:
                if nei not in parent:
                    parent[nei] = node
                    queue.append(nei)
                    newly_discovered.append(nei)

            after = {"queue": list(queue), "new_neighbors": newly_discovered}
            steps.append(_step("POP_NODE", before, after, "expand frontier"))

        path = _path_from_parent(parent, src, dst)
        if path is None:
            final_answer: dict[str, Any] = {"distance": None, "path": None}
        else:
            final_answer = {"distance": len(path) - 1, "path": path}

        prompt = (
            f"Find the shortest path from {src} to {dst} in the unweighted graph with "
            f"{n} nodes and edges {[[u, v] for (u, v) in edges]} using BFS."
        )
        return ExampleRecord(
            prompt=prompt,
            problem={"n": n, "edges": [[u, v] for (u, v) in edges], "src": src, "dst": dst},
            steps=steps,
            final_answer=final_answer,
            difficulty={"n": n, "m": len(edges)},
        )

    def verify(self, example: ExampleRecord) -> tuple[bool, str]:
        n = int(example.problem["n"])
        edges = [(int(u), int(v)) for u, v in example.problem["edges"]]
        src = int(example.problem["src"])
        dst = int(example.problem["dst"])
        adj = _adjacency_from_undirected(n, edges)

        queue = deque([src])
        dist = {src: 0}
        parent: dict[int, int | None] = {src: None}
        while queue:
            node = queue.popleft()
            if node == dst:
                break
            for nei in adj[node]:
                if nei not in dist:
                    dist[nei] = dist[node] + 1
                    parent[nei] = node
                    queue.append(nei)

        expected_path = _path_from_parent(parent, src, dst)
        if expected_path is None:
            expected = {"distance": None, "path": None}
        else:
            expected = {"distance": len(expected_path) - 1, "path": expected_path}

        if example.final_answer != expected:
            return False, "bfs_answer_mismatch"
        return True, "ok"


class DijkstraShortestPathDomain(DomainSolver):
    name = "dijkstra_shortest_path"

    def generate(self, rng: np.random.Generator) -> ExampleRecord:
        n = int(rng.integers(6, 11))
        base_edges = _build_undirected_graph(rng, n, extra_edge_prob=0.25)
        weighted_edges = [(u, v, int(rng.integers(1, 10))) for u, v in base_edges]
        adj = _adjacency_weighted_from_undirected(n, weighted_edges)
        src = int(rng.integers(0, n))
        dst = int(rng.integers(0, n))
        while dst == src:
            dst = int(rng.integers(0, n))

        import heapq

        pq: list[tuple[int, int]] = [(0, src)]
        dist = {src: 0}
        parent: dict[int, int | None] = {src: None}
        steps: list[StepRecord] = []

        while pq:
            d, node = heapq.heappop(pq)
            if d != dist.get(node):
                continue
            before = {"node": node, "distance": d}
            if node == dst:
                steps.append(_step("POP_MIN", before, {"status": "target_found"}, "terminate"))
                break
            relaxations: list[dict[str, int]] = []
            for nei, w in adj[node]:
                cand = d + w
                if cand < dist.get(nei, 10**9):
                    dist[nei] = cand
                    parent[nei] = node
                    heapq.heappush(pq, (cand, nei))
                    relaxations.append({"to": nei, "new_dist": cand, "weight": w})
            after = {"relaxations": relaxations}
            steps.append(_step("POP_MIN", before, after, "relax outgoing edges"))

        path = _path_from_parent(parent, src, dst)
        if path is None:
            final_answer: dict[str, Any] = {"distance": None, "path": None}
        else:
            final_answer = {"distance": dist[dst], "path": path}

        prompt = (
            f"Find the shortest path from {src} to {dst} in the weighted undirected graph "
            f"with edges {[[u, v, w] for (u, v, w) in weighted_edges]} using Dijkstra."
        )
        return ExampleRecord(
            prompt=prompt,
            problem={"n": n, "edges": [[u, v, w] for u, v, w in weighted_edges], "src": src, "dst": dst},
            steps=steps,
            final_answer=final_answer,
            difficulty={"n": n, "m": len(weighted_edges)},
        )

    def verify(self, example: ExampleRecord) -> tuple[bool, str]:
        n = int(example.problem["n"])
        weighted_edges = [(int(u), int(v), int(w)) for u, v, w in example.problem["edges"]]
        src = int(example.problem["src"])
        dst = int(example.problem["dst"])
        adj = _adjacency_weighted_from_undirected(n, weighted_edges)
        import heapq

        pq: list[tuple[int, int]] = [(0, src)]
        dist = {src: 0}
        parent: dict[int, int | None] = {src: None}
        while pq:
            d, node = heapq.heappop(pq)
            if d != dist.get(node):
                continue
            if node == dst:
                break
            for nei, w in adj[node]:
                cand = d + w
                if cand < dist.get(nei, 10**9):
                    dist[nei] = cand
                    parent[nei] = node
                    heapq.heappush(pq, (cand, nei))

        path = _path_from_parent(parent, src, dst)
        if path is None:
            expected = {"distance": None, "path": None}
        else:
            expected = {"distance": dist[dst], "path": path}

        if example.final_answer != expected:
            return False, "dijkstra_answer_mismatch"
        return True, "ok"


class TopologicalSortDomain(DomainSolver):
    name = "topological_sort"

    def generate(self, rng: np.random.Generator) -> ExampleRecord:
        n = int(rng.integers(6, 12))
        order = list(range(n))
        rng.shuffle(order)
        inv = {node: i for i, node in enumerate(order)}
        edges: set[tuple[int, int]] = set()
        for i in range(n - 1):
            edges.add((order[i], order[i + 1]))
        for u in range(n):
            for v in range(n):
                if inv[u] < inv[v] and (u, v) not in edges and float(rng.random()) < 0.18:
                    edges.add((u, v))
        edge_list = sorted(edges)

        indeg = {i: 0 for i in range(n)}
        adj = {i: [] for i in range(n)}
        for u, v in edge_list:
            adj[u].append(v)
            indeg[v] += 1
        for u in adj:
            adj[u].sort()

        import heapq

        pq = [node for node, deg in indeg.items() if deg == 0]
        heapq.heapify(pq)
        topo: list[int] = []
        steps: list[StepRecord] = []
        while pq:
            node = heapq.heappop(pq)
            before = {"available": sorted([*pq, node]), "picked": node}
            topo.append(node)
            newly_available: list[int] = []
            for nei in adj[node]:
                indeg[nei] -= 1
                if indeg[nei] == 0:
                    heapq.heappush(pq, nei)
                    newly_available.append(nei)
            after = {"newly_available": sorted(newly_available), "topo_prefix": list(topo)}
            steps.append(_step("KAHN_POP", before, after, "remove zero-indegree node"))

        prompt = (
            f"Compute a topological ordering for DAG with nodes 0..{n - 1} "
            f"and edges {[[u, v] for u, v in edge_list]}."
        )
        return ExampleRecord(
            prompt=prompt,
            problem={"n": n, "edges": [[u, v] for u, v in edge_list]},
            steps=steps,
            final_answer=topo,
            difficulty={"n": n, "m": len(edge_list)},
        )

    def verify(self, example: ExampleRecord) -> tuple[bool, str]:
        n = int(example.problem["n"])
        edges = [(int(u), int(v)) for u, v in example.problem["edges"]]
        topo = [int(x) for x in example.final_answer]
        if len(topo) != n or sorted(topo) != list(range(n)):
            return False, "invalid_topological_permutation"
        pos = {node: idx for idx, node in enumerate(topo)}
        for u, v in edges:
            if pos[u] >= pos[v]:
                return False, "edge_order_violation"
        return True, "ok"


class ConnectedComponentsDomain(DomainSolver):
    name = "connected_components"

    def generate(self, rng: np.random.Generator) -> ExampleRecord:
        n = int(rng.integers(7, 14))
        edges_set: set[tuple[int, int]] = set()
        for u in range(n):
            for v in range(u + 1, n):
                if float(rng.random()) < 0.2:
                    edges_set.add((u, v))
        edges = sorted(edges_set)
        adj = _adjacency_from_undirected(n, edges)

        visited: set[int] = set()
        components: list[list[int]] = []
        steps: list[StepRecord] = []

        for node in range(n):
            if node in visited:
                continue
            queue = deque([node])
            visited.add(node)
            comp: list[int] = []
            while queue:
                cur = queue.popleft()
                comp.append(cur)
                for nei in adj[cur]:
                    if nei not in visited:
                        visited.add(nei)
                        queue.append(nei)
            comp_sorted = sorted(comp)
            components.append(comp_sorted)
            steps.append(
                _step(
                    "NEW_COMPONENT",
                    {"seed_node": node},
                    {"component": comp_sorted, "num_seen": len(visited)},
                    "bfs from unseen node",
                )
            )

        components_sorted = sorted(components, key=lambda c: (c[0], len(c)))
        prompt = (
            f"Find all connected components in the undirected graph with {n} nodes "
            f"and edges {[[u, v] for u, v in edges]}."
        )
        return ExampleRecord(
            prompt=prompt,
            problem={"n": n, "edges": [[u, v] for u, v in edges]},
            steps=steps,
            final_answer={"components": components_sorted},
            difficulty={"n": n, "m": len(edges)},
        )

    def verify(self, example: ExampleRecord) -> tuple[bool, str]:
        n = int(example.problem["n"])
        edges = [(int(u), int(v)) for u, v in example.problem["edges"]]
        adj = _adjacency_from_undirected(n, edges)
        visited: set[int] = set()
        expected: list[list[int]] = []
        for node in range(n):
            if node in visited:
                continue
            queue = deque([node])
            visited.add(node)
            comp: list[int] = []
            while queue:
                cur = queue.popleft()
                comp.append(cur)
                for nei in adj[cur]:
                    if nei not in visited:
                        visited.add(nei)
                        queue.append(nei)
            expected.append(sorted(comp))

        expected_sorted = sorted(expected, key=lambda c: (c[0], len(c)))
        actual = [[int(x) for x in comp] for comp in example.final_answer["components"]]
        actual_sorted = sorted([sorted(c) for c in actual], key=lambda c: (c[0], len(c)))
        if actual_sorted != expected_sorted:
            return False, "components_mismatch"
        return True, "ok"


class LcsDpDomain(DomainSolver):
    name = "lcs_dp"

    def generate(self, rng: np.random.Generator) -> ExampleRecord:
        alphabet = list("abcd")
        m = int(rng.integers(4, 9))
        n = int(rng.integers(4, 9))
        a = "".join(str(rng.choice(alphabet)) for _ in range(m))
        b = "".join(str(rng.choice(alphabet)) for _ in range(n))
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        steps: list[StepRecord] = []

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                before = {"i": i, "j": j, "a_i": a[i - 1], "b_j": b[j - 1], "prev": dp[i][j]}
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    details = "chars match"
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                    details = "chars differ"
                after = {"dp_ij": dp[i][j]}
                steps.append(_step("DP_CELL", before, after, details))

        i, j = m, n
        lcs_chars: list[str] = []
        while i > 0 and j > 0:
            if a[i - 1] == b[j - 1]:
                lcs_chars.append(a[i - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] >= dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
        lcs = "".join(reversed(lcs_chars))

        prompt = f"Compute the longest common subsequence (LCS) of '{a}' and '{b}' using dynamic programming."
        return ExampleRecord(
            prompt=prompt,
            problem={"a": a, "b": b},
            steps=steps,
            final_answer={"length": dp[m][n], "lcs": lcs},
            difficulty={"m": m, "n": n},
        )

    def verify(self, example: ExampleRecord) -> tuple[bool, str]:
        a = str(example.problem["a"])
        b = str(example.problem["b"])
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        expected_len = dp[m][n]
        ans_len = int(example.final_answer["length"])
        lcs = str(example.final_answer["lcs"])
        if ans_len != expected_len:
            return False, "lcs_length_mismatch"
        if len(lcs) != expected_len:
            return False, "lcs_string_length_mismatch"
        if not _is_subsequence(lcs, a) or not _is_subsequence(lcs, b):
            return False, "lcs_not_subsequence"
        return True, "ok"


class EditDistanceDpDomain(DomainSolver):
    name = "edit_distance_dp"

    def generate(self, rng: np.random.Generator) -> ExampleRecord:
        alphabet = list("abcd")
        m = int(rng.integers(4, 9))
        n = int(rng.integers(4, 9))
        a = "".join(str(rng.choice(alphabet)) for _ in range(m))
        b = "".join(str(rng.choice(alphabet)) for _ in range(n))
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        steps: list[StepRecord] = []
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                before = {"i": i, "j": j, "a_i": a[i - 1], "b_j": b[j - 1], "prev": dp[i][j]}
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # delete
                    dp[i][j - 1] + 1,  # insert
                    dp[i - 1][j - 1] + cost,  # substitute / match
                )
                details = "match_or_substitute_with_insert_delete"
                steps.append(_step("DP_CELL", before, {"dp_ij": dp[i][j]}, details))

        prompt = f"Compute the Levenshtein edit distance between '{a}' and '{b}' using dynamic programming."
        return ExampleRecord(
            prompt=prompt,
            problem={"a": a, "b": b},
            steps=steps,
            final_answer=dp[m][n],
            difficulty={"m": m, "n": n},
        )

    def verify(self, example: ExampleRecord) -> tuple[bool, str]:
        a = str(example.problem["a"])
        b = str(example.problem["b"])
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        if int(example.final_answer) != dp[m][n]:
            return False, "edit_distance_mismatch"
        return True, "ok"


class CoinChangeDpDomain(DomainSolver):
    name = "coin_change_dp"

    def generate(self, rng: np.random.Generator) -> ExampleRecord:
        k = int(rng.integers(3, 6))
        pool = sorted(set(int(x) for x in rng.integers(2, 16, size=10)))
        coins = [1]
        for val in pool:
            if val not in coins:
                coins.append(val)
            if len(coins) >= k:
                break
        coins = sorted(coins)
        amount = int(rng.integers(15, 81))

        inf = 10**9
        dp = [inf] * (amount + 1)
        dp[0] = 0
        steps: list[StepRecord] = []
        for x in range(1, amount + 1):
            before = {"amount": x, "prev": dp[x]}
            best = inf
            best_coin = None
            for coin in coins:
                if coin <= x and dp[x - coin] + 1 < best:
                    best = dp[x - coin] + 1
                    best_coin = coin
            dp[x] = best
            after = {"min_coins": dp[x], "chosen_coin": best_coin}
            steps.append(_step("DP_AMOUNT", before, after, "best among valid coin transitions"))

        prompt = f"Given coin denominations {coins}, find the minimum number of coins to make amount {amount}."
        return ExampleRecord(
            prompt=prompt,
            problem={"coins": coins, "amount": amount},
            steps=steps,
            final_answer=dp[amount],
            difficulty={"num_coins": len(coins), "amount": amount},
        )

    def verify(self, example: ExampleRecord) -> tuple[bool, str]:
        coins = [int(x) for x in example.problem["coins"]]
        amount = int(example.problem["amount"])
        inf = 10**9
        dp = [inf] * (amount + 1)
        dp[0] = 0
        for x in range(1, amount + 1):
            for coin in coins:
                if coin <= x:
                    dp[x] = min(dp[x], dp[x - coin] + 1)
        if int(example.final_answer) != dp[amount]:
            return False, "coin_change_mismatch"
        return True, "ok"


class Knapsack01DpDomain(DomainSolver):
    name = "knapsack_01_dp"

    def generate(self, rng: np.random.Generator) -> ExampleRecord:
        n = int(rng.integers(5, 9))
        weights = [int(x) for x in rng.integers(1, 11, size=n)]
        values = [int(x) for x in rng.integers(1, 26, size=n)]
        capacity = int(rng.integers(12, 31))

        dp = [[0] * (capacity + 1) for _ in range(n + 1)]
        steps: list[StepRecord] = []
        for i in range(1, n + 1):
            wi = weights[i - 1]
            vi = values[i - 1]
            for cap in range(capacity + 1):
                before = {"i": i, "cap": cap, "prev": dp[i][cap]}
                best = dp[i - 1][cap]
                if wi <= cap:
                    best = max(best, dp[i - 1][cap - wi] + vi)
                dp[i][cap] = best
                steps.append(_step("DP_CELL", before, {"dp_i_cap": best}, "take_or_skip_item"))

        chosen: list[int] = []
        cap = capacity
        for i in range(n, 0, -1):
            if dp[i][cap] != dp[i - 1][cap]:
                chosen.append(i - 1)
                cap -= weights[i - 1]
        chosen.reverse()

        prompt = (
            f"Solve 0/1 knapsack with capacity {capacity}, weights {weights}, and values {values}. "
            "Return maximum achievable value."
        )
        return ExampleRecord(
            prompt=prompt,
            problem={"weights": weights, "values": values, "capacity": capacity},
            steps=steps,
            final_answer={"max_value": dp[n][capacity], "chosen_items": chosen},
            difficulty={"n_items": n, "capacity": capacity},
        )

    def verify(self, example: ExampleRecord) -> tuple[bool, str]:
        weights = [int(x) for x in example.problem["weights"]]
        values = [int(x) for x in example.problem["values"]]
        capacity = int(example.problem["capacity"])
        n = len(weights)
        best = 0
        for mask in range(1 << n):
            tw = 0
            tv = 0
            for i in range(n):
                if mask & (1 << i):
                    tw += weights[i]
                    tv += values[i]
            if tw <= capacity:
                best = max(best, tv)
        if int(example.final_answer["max_value"]) != best:
            return False, "knapsack_max_value_mismatch"
        return True, "ok"


class ParenthesesBalanceDomain(DomainSolver):
    name = "parentheses_balance"

    def generate(self, rng: np.random.Generator) -> ExampleRecord:
        n = int(rng.integers(8, 25))
        if n % 2 == 1:
            n += 1
        balanced_mode = float(rng.random()) < 0.5

        if balanced_mode:
            opens_left = n // 2
            closes_left = n // 2
            balance = 0
            chars: list[str] = []
            for _ in range(n):
                if opens_left == 0:
                    chars.append(")")
                    closes_left -= 1
                    balance -= 1
                    continue
                if closes_left == 0:
                    chars.append("(")
                    opens_left -= 1
                    balance += 1
                    continue
                if balance == 0:
                    pick_open = True
                else:
                    pick_open = bool(rng.integers(0, 2))
                if pick_open and opens_left > 0:
                    chars.append("(")
                    opens_left -= 1
                    balance += 1
                else:
                    chars.append(")")
                    closes_left -= 1
                    balance -= 1
            s = "".join(chars)
            # Repair if needed
            if not self._is_balanced(s):
                s = "(" * (n // 2) + ")" * (n // 2)
        else:
            s = "".join(str(rng.choice(["(", ")"])) for _ in range(n))

        steps: list[StepRecord] = []
        balance = 0
        first_error_index: int | None = None
        for idx, ch in enumerate(s):
            before = {"index": idx, "char": ch, "balance": balance}
            if ch == "(":
                balance += 1
            else:
                balance -= 1
            if balance < 0 and first_error_index is None:
                first_error_index = idx
            after = {"balance": balance}
            steps.append(_step("SCAN_CHAR", before, after, "update running balance"))
        balanced = balance == 0 and first_error_index is None
        final_answer = {"balanced": balanced, "first_error_index": first_error_index}

        prompt = f"Determine whether the parentheses string is balanced: '{s}'"
        return ExampleRecord(
            prompt=prompt,
            problem={"s": s},
            steps=steps,
            final_answer=final_answer,
            difficulty={"length": len(s)},
        )

    @staticmethod
    def _is_balanced(s: str) -> bool:
        bal = 0
        for ch in s:
            bal += 1 if ch == "(" else -1
            if bal < 0:
                return False
        return bal == 0

    def verify(self, example: ExampleRecord) -> tuple[bool, str]:
        s = str(example.problem["s"])
        bal = 0
        first_error = None
        for idx, ch in enumerate(s):
            bal += 1 if ch == "(" else -1
            if bal < 0 and first_error is None:
                first_error = idx
        expected = {"balanced": bal == 0 and first_error is None, "first_error_index": first_error}
        if example.final_answer != expected:
            return False, "parentheses_result_mismatch"
        return True, "ok"


BoolAst = tuple[str, Any, Any] | tuple[str, Any] | tuple[str]


def _rand_bool_ast(rng: np.random.Generator, vars_: list[str], depth: int) -> BoolAst:
    if depth <= 0 or float(rng.random()) < 0.35:
        return ("var", str(rng.choice(vars_)))
    op = str(rng.choice(["not", "and", "or", "imp"]))
    if op == "not":
        return ("not", _rand_bool_ast(rng, vars_, depth - 1))
    left = _rand_bool_ast(rng, vars_, depth - 1)
    right = _rand_bool_ast(rng, vars_, depth - 1)
    return (op, left, right)


def _eval_bool_ast(ast: BoolAst, env: dict[str, bool]) -> bool:
    op = ast[0]
    if op == "var":
        return bool(env[str(ast[1])])  # type: ignore[index]
    if op == "not":
        return not _eval_bool_ast(ast[1], env)  # type: ignore[index]
    if op == "and":
        return _eval_bool_ast(ast[1], env) and _eval_bool_ast(ast[2], env)  # type: ignore[index]
    if op == "or":
        return _eval_bool_ast(ast[1], env) or _eval_bool_ast(ast[2], env)  # type: ignore[index]
    if op == "imp":
        return (not _eval_bool_ast(ast[1], env)) or _eval_bool_ast(ast[2], env)  # type: ignore[index]
    raise ValueError(f"Unknown op: {op}")


def _bool_ast_to_str(ast: BoolAst) -> str:
    op = ast[0]
    if op == "var":
        return str(ast[1])  # type: ignore[index]
    if op == "not":
        return f"(not {_bool_ast_to_str(ast[1])})"  # type: ignore[index]
    if op == "and":
        return f"({_bool_ast_to_str(ast[1])} and {_bool_ast_to_str(ast[2])})"  # type: ignore[index]
    if op == "or":
        return f"({_bool_ast_to_str(ast[1])} or {_bool_ast_to_str(ast[2])})"  # type: ignore[index]
    if op == "imp":
        return f"({_bool_ast_to_str(ast[1])} -> {_bool_ast_to_str(ast[2])})"  # type: ignore[index]
    raise ValueError(f"Unknown op: {op}")


class PropositionalEntailmentDomain(DomainSolver):
    name = "propositional_entailment"

    def generate(self, rng: np.random.Generator) -> ExampleRecord:
        all_vars = ["p", "q", "r"]
        n_vars = int(rng.integers(2, 4))
        vars_ = all_vars[:n_vars]
        premise = _rand_bool_ast(rng, vars_, depth=2)
        conclusion = _rand_bool_ast(rng, vars_, depth=2)
        premise_str = _bool_ast_to_str(premise)
        conclusion_str = _bool_ast_to_str(conclusion)

        steps: list[StepRecord] = []
        entails = True
        counterexample: dict[str, bool] | None = None
        for values in itertools.product([False, True], repeat=n_vars):
            env = {var: bool(val) for var, val in zip(vars_, values, strict=True)}
            p_val = _eval_bool_ast(premise, env)
            c_val = _eval_bool_ast(conclusion, env)
            before = {"assignment": env}
            after = {"premise": p_val, "conclusion": c_val}
            steps.append(_step("TRUTH_TABLE_ROW", before, after, "evaluate premise and conclusion"))
            if p_val and not c_val and counterexample is None:
                entails = False
                counterexample = env

        final_answer = {"entails": entails, "counterexample": counterexample}
        prompt = f"Using a truth table, determine whether premise {premise_str} entails conclusion {conclusion_str}."
        return ExampleRecord(
            prompt=prompt,
            problem={"variables": vars_, "premise_ast": premise, "conclusion_ast": conclusion},
            steps=steps,
            final_answer=final_answer,
            difficulty={"n_vars": n_vars},
        )

    def verify(self, example: ExampleRecord) -> tuple[bool, str]:
        vars_ = [str(v) for v in example.problem["variables"]]
        premise = example.problem["premise_ast"]
        conclusion = example.problem["conclusion_ast"]
        entails = True
        counterexample = None
        for values in itertools.product([False, True], repeat=len(vars_)):
            env = {var: bool(val) for var, val in zip(vars_, values, strict=True)}
            p_val = _eval_bool_ast(premise, env)
            c_val = _eval_bool_ast(conclusion, env)
            if p_val and not c_val and counterexample is None:
                entails = False
                counterexample = env
        expected = {"entails": entails, "counterexample": counterexample}
        if example.final_answer != expected:
            return False, "entailment_mismatch"
        return True, "ok"


class LisDpDomain(DomainSolver):
    name = "lis_dp"

    def generate(self, rng: np.random.Generator) -> ExampleRecord:
        n = int(rng.integers(7, 13))
        arr = [int(x) for x in rng.integers(-30, 41, size=n)]
        dp = [1] * n
        prev = [-1] * n
        steps: list[StepRecord] = []

        for i in range(n):
            best_len = 1
            best_prev = -1
            for j in range(i):
                if arr[j] < arr[i] and dp[j] + 1 > best_len:
                    best_len = dp[j] + 1
                    best_prev = j
            before = {"i": i, "value": arr[i], "old_len": dp[i], "old_prev": prev[i]}
            dp[i] = best_len
            prev[i] = best_prev
            after = {"new_len": dp[i], "new_prev": prev[i]}
            steps.append(_step("DP_INDEX", before, after, "best increasing predecessor"))

        best_idx = max(range(n), key=lambda i: (dp[i], -i))
        lis_vals: list[int] = []
        cur = best_idx
        while cur != -1:
            lis_vals.append(arr[cur])
            cur = prev[cur]
        lis_vals.reverse()

        prompt = f"Find a longest increasing subsequence of array {arr} using dynamic programming."
        return ExampleRecord(
            prompt=prompt,
            problem={"array": arr},
            steps=steps,
            final_answer={"length": dp[best_idx], "lis": lis_vals},
            difficulty={"n": n},
        )

    def verify(self, example: ExampleRecord) -> tuple[bool, str]:
        arr = [int(x) for x in example.problem["array"]]
        n = len(arr)
        dp = [1] * n
        for i in range(n):
            for j in range(i):
                if arr[j] < arr[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        expected_len = max(dp)
        actual_len = int(example.final_answer["length"])
        lis = [int(x) for x in example.final_answer["lis"]]
        if actual_len != expected_len:
            return False, "lis_length_mismatch"
        if len(lis) != expected_len:
            return False, "lis_sequence_length_mismatch"
        if any(lis[i] >= lis[i + 1] for i in range(len(lis) - 1)):
            return False, "lis_not_strictly_increasing"
        k = 0
        for v in arr:
            if k < len(lis) and v == lis[k]:
                k += 1
        if k != len(lis):
            return False, "lis_not_subsequence"
        return True, "ok"


class IntervalSchedulingDomain(DomainSolver):
    name = "interval_scheduling"

    def generate(self, rng: np.random.Generator) -> ExampleRecord:
        n = int(rng.integers(8, 17))
        intervals: list[tuple[int, int]] = []
        for _ in range(n):
            start = int(rng.integers(0, 40))
            duration = int(rng.integers(1, 13))
            end = start + duration
            intervals.append((start, end))
        ordered = sorted(intervals, key=lambda x: (x[1], x[0]))

        selected: list[tuple[int, int]] = []
        last_end = -(10**9)
        steps: list[StepRecord] = []
        for interval in ordered:
            start, end = interval
            before = {"interval": [start, end], "last_end": last_end}
            if start >= last_end:
                selected.append(interval)
                last_end = end
                after = {"decision": "take", "new_last_end": last_end}
            else:
                after = {"decision": "skip", "new_last_end": last_end}
            steps.append(_step("GREEDY_DECISION", before, after, "earliest-finish-time rule"))

        prompt = f"Select a maximum set of non-overlapping intervals from {[[s, e] for s, e in intervals]}."
        return ExampleRecord(
            prompt=prompt,
            problem={"intervals": [[s, e] for s, e in intervals]},
            steps=steps,
            final_answer={"count": len(selected), "selected": [[s, e] for s, e in selected]},
            difficulty={"n": n},
        )

    def verify(self, example: ExampleRecord) -> tuple[bool, str]:
        intervals = [(int(s), int(e)) for s, e in example.problem["intervals"]]
        ordered = sorted(intervals, key=lambda x: (x[1], x[0]))
        expected: list[tuple[int, int]] = []
        last_end = -(10**9)
        for s, e in ordered:
            if s >= last_end:
                expected.append((s, e))
                last_end = e

        actual = [(int(s), int(e)) for s, e in example.final_answer["selected"]]
        if actual != expected:
            return False, "interval_schedule_mismatch"
        if int(example.final_answer["count"]) != len(expected):
            return False, "interval_count_mismatch"
        return True, "ok"


class PrimMstDomain(DomainSolver):
    name = "prim_mst"

    def generate(self, rng: np.random.Generator) -> ExampleRecord:
        n = int(rng.integers(6, 12))
        base_edges = _build_undirected_graph(rng, n, extra_edge_prob=0.25)
        weighted_edges = [(u, v, int(rng.integers(1, 16))) for u, v in base_edges]
        adj = _adjacency_weighted_from_undirected(n, weighted_edges)

        import heapq

        start = 0
        in_mst = {start}
        pq: list[tuple[int, int, int]] = []
        for nei, w in adj[start]:
            heapq.heappush(pq, (w, start, nei))
        mst_edges: list[tuple[int, int, int]] = []
        total_weight = 0
        steps: list[StepRecord] = []

        while pq and len(in_mst) < n:
            w, u, v = heapq.heappop(pq)
            if v in in_mst:
                continue
            before = {"picked_edge": [u, v, w], "num_in_mst": len(in_mst)}
            in_mst.add(v)
            mst_edges.append((u, v, w))
            total_weight += w
            pushed = 0
            for nxt, nw in adj[v]:
                if nxt not in in_mst:
                    heapq.heappush(pq, (nw, v, nxt))
                    pushed += 1
            after = {"num_in_mst": len(in_mst), "pushed_edges": pushed, "total_weight": total_weight}
            steps.append(_step("PRIM_ADD_EDGE", before, after, "add minimum crossing edge"))

        prompt = (
            "Find an MST using Prim's algorithm for weighted graph edges "
            f"{[[u, v, w] for u, v, w in weighted_edges]}."
        )
        return ExampleRecord(
            prompt=prompt,
            problem={"n": n, "edges": [[u, v, w] for u, v, w in weighted_edges]},
            steps=steps,
            final_answer={"total_weight": total_weight, "edges": [[u, v, w] for u, v, w in mst_edges]},
            difficulty={"n": n, "m": len(weighted_edges)},
        )

    def verify(self, example: ExampleRecord) -> tuple[bool, str]:
        n = int(example.problem["n"])
        edges = [(int(u), int(v), int(w)) for u, v, w in example.problem["edges"]]
        actual_weight = int(example.final_answer["total_weight"])
        actual_edges = [(int(u), int(v), int(w)) for u, v, w in example.final_answer["edges"]]
        if len(actual_edges) != n - 1:
            return False, "mst_edge_count_mismatch"

        parent = list(range(n))
        rank = [0] * n

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> bool:
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1
            return True

        min_weight = 0
        used = 0
        for u, v, w in sorted(edges, key=lambda x: (x[2], x[0], x[1])):
            if union(u, v):
                min_weight += w
                used += 1
                if used == n - 1:
                    break
        if actual_weight != min_weight:
            return False, "mst_weight_mismatch"
        return True, "ok"


class FloydWarshallDomain(DomainSolver):
    name = "floyd_warshall_apsp"

    @staticmethod
    def _to_json_matrix(dist: list[list[float]]) -> list[list[int | None]]:
        out: list[list[int | None]] = []
        for row in dist:
            out.append([None if x >= 10**12 else int(x) for x in row])
        return out

    def generate(self, rng: np.random.Generator) -> ExampleRecord:
        n = int(rng.integers(4, 8))
        inf = 10**12
        dist = [[inf] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
        edges: list[tuple[int, int, int]] = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if float(rng.random()) < 0.35:
                    w = int(rng.integers(1, 16))
                    edges.append((i, j, w))
                    if w < dist[i][j]:
                        dist[i][j] = w

        steps: list[StepRecord] = []
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    before_val = dist[i][j]
                    via = dist[i][k] + dist[k][j]
                    if via < before_val:
                        dist[i][j] = via
                        steps.append(
                            _step(
                                "FW_UPDATE",
                                {"k": k, "i": i, "j": j, "old": None if before_val >= inf else int(before_val)},
                                {"new": int(via)},
                                "improve via intermediate k",
                            )
                        )

        prompt = (
            f"Compute all-pairs shortest paths with Floyd-Warshall on directed weighted edges "
            f"{[[u, v, w] for u, v, w in edges]} for nodes 0..{n - 1}."
        )
        return ExampleRecord(
            prompt=prompt,
            problem={"n": n, "edges": [[u, v, w] for u, v, w in edges]},
            steps=steps,
            final_answer={"distance_matrix": self._to_json_matrix(dist)},
            difficulty={"n": n, "m": len(edges)},
        )

    def verify(self, example: ExampleRecord) -> tuple[bool, str]:
        n = int(example.problem["n"])
        edges = [(int(u), int(v), int(w)) for u, v, w in example.problem["edges"]]
        inf = 10**12
        dist = [[inf] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
        for u, v, w in edges:
            dist[u][v] = min(dist[u][v], w)
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        expected = FloydWarshallDomain._to_json_matrix(dist)
        if example.final_answer["distance_matrix"] != expected:
            return False, "floyd_warshall_mismatch"
        return True, "ok"


class KmpSearchDomain(DomainSolver):
    name = "kmp_string_search"

    @staticmethod
    def _build_lps(pattern: str) -> tuple[list[int], list[StepRecord]]:
        lps = [0] * len(pattern)
        steps: list[StepRecord] = []
        length = 0
        i = 1
        while i < len(pattern):
            before = {"i": i, "length": length, "char_i": pattern[i], "char_len": pattern[length]}
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                steps.append(_step("LPS_MATCH", before, {"new_lps_i": lps[i], "new_length": length}, "extend prefix"))
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                    steps.append(_step("LPS_FALLBACK", before, {"new_length": length}, "fallback via lps"))
                else:
                    lps[i] = 0
                    steps.append(_step("LPS_ZERO", before, {"new_lps_i": 0}, "no prefix suffix"))
                    i += 1
        return lps, steps

    def generate(self, rng: np.random.Generator) -> ExampleRecord:
        alphabet = list("abcd")
        p_len = int(rng.integers(3, 8))
        t_len = int(rng.integers(12, 31))
        pattern = "".join(str(rng.choice(alphabet)) for _ in range(p_len))
        force_present = float(rng.random()) < 0.7
        if force_present and p_len <= t_len:
            pos = int(rng.integers(0, t_len - p_len + 1))
            text_chars = [str(rng.choice(alphabet)) for _ in range(t_len)]
            text_chars[pos : pos + p_len] = list(pattern)
            text = "".join(text_chars)
        else:
            text = "".join(str(rng.choice(alphabet)) for _ in range(t_len))

        lps, lps_steps = self._build_lps(pattern)
        steps = list(lps_steps)

        i = 0
        j = 0
        found = -1
        while i < len(text):
            before = {"i": i, "j": j, "text_i": text[i], "pat_j": pattern[j]}
            if text[i] == pattern[j]:
                i += 1
                j += 1
                after = {"i": i, "j": j}
                steps.append(_step("SCAN_MATCH", before, after, "advance both pointers"))
                if j == len(pattern):
                    found = i - j
                    steps.append(_step("PATTERN_FOUND", {"i": i, "j": j}, {"index": found}, "first occurrence"))
                    break
            else:
                if j != 0:
                    j = lps[j - 1]
                    steps.append(_step("SCAN_FALLBACK", before, {"i": i, "j": j}, "fallback pattern pointer"))
                else:
                    i += 1
                    steps.append(_step("SCAN_SHIFT", before, {"i": i, "j": j}, "shift text pointer"))

        prompt = f"Find first occurrence of pattern '{pattern}' in text '{text}' using KMP."
        return ExampleRecord(
            prompt=prompt,
            problem={"text": text, "pattern": pattern},
            steps=steps,
            final_answer={"index": found},
            difficulty={"text_len": len(text), "pattern_len": len(pattern)},
        )

    def verify(self, example: ExampleRecord) -> tuple[bool, str]:
        text = str(example.problem["text"])
        pattern = str(example.problem["pattern"])
        expected = text.find(pattern)
        if int(example.final_answer["index"]) != expected:
            return False, "kmp_index_mismatch"
        return True, "ok"


class UnionFindConnectivityDomain(DomainSolver):
    name = "union_find_connectivity"

    def generate(self, rng: np.random.Generator) -> ExampleRecord:
        n = int(rng.integers(6, 13))
        n_ops = int(rng.integers(14, 27))
        ops: list[dict[str, Any]] = []
        query_answers: list[dict[str, Any]] = []

        parent = list(range(n))
        rank = [0] * n

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> bool:
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1
            return True

        steps: list[StepRecord] = []
        for _ in range(n_ops):
            u = int(rng.integers(0, n))
            v = int(rng.integers(0, n))
            while v == u:
                v = int(rng.integers(0, n))
            if float(rng.random()) < 0.62:
                merged = union(u, v)
                op = {"op": "union", "u": u, "v": v}
                ops.append(op)
                steps.append(_step("UNION", {"u": u, "v": v}, {"merged": merged}, "union by rank"))
            else:
                connected = find(u) == find(v)
                op = {"op": "query", "u": u, "v": v}
                ops.append(op)
                answer = {"u": u, "v": v, "connected": connected}
                query_answers.append(answer)
                steps.append(_step("QUERY", {"u": u, "v": v}, answer, "connectivity query"))

        prompt = f"Process union-find operations on {n} nodes: {ops}. Return answers for all query ops."
        return ExampleRecord(
            prompt=prompt,
            problem={"n": n, "ops": ops},
            steps=steps,
            final_answer={"query_answers": query_answers},
            difficulty={"n": n, "n_ops": n_ops},
        )

    def verify(self, example: ExampleRecord) -> tuple[bool, str]:
        n = int(example.problem["n"])
        ops = example.problem["ops"]
        parent = list(range(n))
        rank = [0] * n

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> bool:
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1
            return True

        expected: list[dict[str, Any]] = []
        for op in ops:
            t = str(op["op"])
            u = int(op["u"])
            v = int(op["v"])
            if t == "union":
                union(u, v)
            elif t == "query":
                expected.append({"u": u, "v": v, "connected": find(u) == find(v)})
            else:
                return False, "unknown_union_find_op"

        if example.final_answer["query_answers"] != expected:
            return False, "union_find_query_mismatch"
        return True, "ok"


class NQueensDomain(DomainSolver):
    name = "n_queens_backtracking"

    def generate(self, rng: np.random.Generator) -> ExampleRecord:
        n = int(rng.integers(4, 8))
        cols = set()
        diag1 = set()  # r-c
        diag2 = set()  # r+c
        queens = [-1] * n
        steps: list[StepRecord] = []
        solved = False

        def backtrack(r: int) -> bool:
            nonlocal solved
            if r == n:
                solved = True
                return True
            for c in range(n):
                before = {"row": r, "col": c}
                safe = c not in cols and (r - c) not in diag1 and (r + c) not in diag2
                if safe:
                    cols.add(c)
                    diag1.add(r - c)
                    diag2.add(r + c)
                    queens[r] = c
                    steps.append(_step("PLACE_QUEEN", before, {"placed": True, "partial": list(queens)}, "safe"))
                    if backtrack(r + 1):
                        return True
                    cols.remove(c)
                    diag1.remove(r - c)
                    diag2.remove(r + c)
                    queens[r] = -1
                    steps.append(
                        _step(
                            "BACKTRACK",
                            {"row": r, "col": c},
                            {"placed": False, "partial": list(queens)},
                            "dead end",
                        )
                    )
                else:
                    steps.append(_step("TRY_POSITION", before, {"placed": False}, "conflict"))
            return False

        backtrack(0)
        prompt = f"Solve the {n}-queens problem with backtracking and return one valid column assignment per row."
        return ExampleRecord(
            prompt=prompt,
            problem={"n": n},
            steps=steps,
            final_answer={"solved": solved, "queens": queens},
            difficulty={"n": n},
        )

    def verify(self, example: ExampleRecord) -> tuple[bool, str]:
        n = int(example.problem["n"])
        solved = bool(example.final_answer["solved"])
        queens = [int(x) for x in example.final_answer["queens"]]
        if not solved:
            return False, "solver_reported_unsolved"
        if len(queens) != n or any(c < 0 or c >= n for c in queens):
            return False, "invalid_queens_assignment"
        if len(set(queens)) != n:
            return False, "column_conflict"
        diag1 = set()
        diag2 = set()
        for r, c in enumerate(queens):
            d1 = r - c
            d2 = r + c
            if d1 in diag1 or d2 in diag2:
                return False, "diagonal_conflict"
            diag1.add(d1)
            diag2.add(d2)
        return True, "ok"


DOMAIN_SOLVERS: dict[str, DomainSolver] = {
    solver.name: solver
    for solver in [
        EuclidGcdDomain(),
        BinarySearchDomain(),
        InsertionSortDomain(),
        BfsShortestPathDomain(),
        DijkstraShortestPathDomain(),
        TopologicalSortDomain(),
        ConnectedComponentsDomain(),
        LcsDpDomain(),
        EditDistanceDpDomain(),
        CoinChangeDpDomain(),
        Knapsack01DpDomain(),
        ParenthesesBalanceDomain(),
        PropositionalEntailmentDomain(),
        LisDpDomain(),
        IntervalSchedulingDomain(),
        PrimMstDomain(),
        FloydWarshallDomain(),
        KmpSearchDomain(),
        UnionFindConnectivityDomain(),
        NQueensDomain(),
    ]
}

CLRS_STYLE_DOMAIN_MAP: dict[str, str] = {
    "clrs_insertion_sort": "insertion_sort",
    "clrs_binary_search": "binary_search",
    "clrs_bfs": "bfs_shortest_path",
    "clrs_dijkstra": "dijkstra_shortest_path",
    "clrs_topological_sort": "topological_sort",
    "clrs_connected_components": "connected_components",
    "clrs_mst_prim": "prim_mst",
    "clrs_floyd_warshall": "floyd_warshall_apsp",
    "clrs_lcs_length": "lcs_dp",
    "clrs_lis": "lis_dp",
}


def list_domains() -> list[str]:
    return sorted(DOMAIN_SOLVERS.keys())


def list_clrs_domains() -> list[str]:
    return sorted(CLRS_STYLE_DOMAIN_MAP.keys())


def _resolve_domain_jobs(domain_args: list[str], profile: str) -> list[DomainJob]:
    if profile == "native":
        if domain_args == ["all"]:
            names = list_domains()
        else:
            invalid = [name for name in domain_args if name not in DOMAIN_SOLVERS]
            if invalid:
                raise ValueError(f"Unknown native domains: {invalid}. Available: {list_domains()}")
            names = domain_args
        return [DomainJob(output_domain=name, solver_domain=name, profile="native") for name in names]

    if profile == "clrs_style":
        if domain_args == ["all"]:
            names = list_clrs_domains()
        else:
            invalid = [name for name in domain_args if name not in CLRS_STYLE_DOMAIN_MAP]
            if invalid:
                raise ValueError(f"Unknown CLRS-style domains: {invalid}. Available: {list_clrs_domains()}")
            names = domain_args
        jobs: list[DomainJob] = []
        for name in names:
            solver_name = CLRS_STYLE_DOMAIN_MAP[name]
            jobs.append(
                DomainJob(
                    output_domain=name,
                    solver_domain=solver_name,
                    profile="clrs_style",
                    profile_task=name.removeprefix("clrs_"),
                )
            )
        return jobs

    raise ValueError(f"Unknown profile: {profile}")


def _stable_domain_seed(base_seed: int, domain_name: str) -> int:
    return (int(base_seed) + int(zlib.crc32(domain_name.encode("utf-8")))) % (2**32)


class ShardedJsonlWriter:
    """Append-only JSONL.GZ writer with deterministic shard rotation."""

    def __init__(self, root_dir: Path, prefix: str, shard_size: int) -> None:
        self.root_dir = root_dir
        self.prefix = prefix
        self.shard_size = shard_size
        self._shard_index = 0
        self._rows_in_shard = 0
        self._total_rows = 0
        self._handle = None
        self.paths: list[str] = []
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _open_new_shard(self) -> None:
        if self._handle is not None:
            self._handle.close()
        path = self.root_dir / f"{self.prefix}-{self._shard_index:05d}.jsonl.gz"
        self.paths.append(path.as_posix())
        self._handle = gzip.open(path, "wt", encoding="utf-8")
        self._rows_in_shard = 0
        self._shard_index += 1

    def write(self, row: dict[str, Any]) -> None:
        if self._handle is None or self._rows_in_shard >= self.shard_size:
            self._open_new_shard()
        assert self._handle is not None
        self._handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        self._rows_in_shard += 1
        self._total_rows += 1

    @property
    def total_rows(self) -> int:
        return self._total_rows

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _canonical_step_dict(step: StepRecord, index: int) -> dict[str, Any]:
    return {
        "index": index,
        "operation": step.operation,
        "before": step.before,
        "after": step.after,
        "details": step.details,
    }


def _make_canonical_row(
    *,
    domain_name: str,
    solver_domain: str,
    profile: str,
    profile_task: str | None,
    row_id: str,
    base_seed: int,
    domain_seed: int,
    index: int,
    example: ExampleRecord,
    verified: bool,
    verify_reason: str,
) -> dict[str, Any]:
    return {
        "id": row_id,
        "domain": domain_name,
        "solver_domain": solver_domain,
        "profile": profile,
        "profile_task": profile_task,
        "seed": base_seed,
        "domain_seed": domain_seed,
        "index": index,
        "prompt": example.prompt,
        "problem": example.problem,
        "steps": [_canonical_step_dict(step, i) for i, step in enumerate(example.steps)],
        "final_answer": example.final_answer,
        "difficulty": example.difficulty,
        "verification": {"passed": verified, "reason": verify_reason},
    }


def _make_oai_row(
    *,
    canonical_row: dict[str, Any],
    example: ExampleRecord,
    include_operation_names: bool,
    assistant_style: str,
) -> dict[str, Any]:
    profile = str(canonical_row.get("profile", "native"))
    domain = str(canonical_row["domain"])
    source = f"synthetic_reasoning/{domain}/v1"
    if profile != "native":
        source = f"synthetic_reasoning/{profile}/{domain}/v1"
    return {
        "id": canonical_row["id"],
        "source": source,
        "messages": [
            {"role": "user", "content": example.prompt},
            {
                "role": "assistant",
                "content": _render_assistant_response(
                    example=example,
                    include_operation_names=include_operation_names,
                    assistant_style=assistant_style,
                    domain_name=domain,
                    row_id=str(canonical_row["id"]),
                ),
            },
        ],
        "metadata": {
            "domain": canonical_row["domain"],
            "solver_domain": canonical_row.get("solver_domain", canonical_row["domain"]),
            "profile": canonical_row.get("profile", "native"),
            "profile_task": canonical_row.get("profile_task"),
            "assistant_style": assistant_style,
            "difficulty": example.difficulty,
            "verification": canonical_row["verification"],
            "seed": canonical_row["seed"],
            "domain_seed": canonical_row["domain_seed"],
            "index": canonical_row["index"],
        },
    }


def generate_dataset(
    *,
    output_dir: Path,
    domains: list[str],
    profile: str = "native",
    examples_per_domain: int,
    seed: int,
    shard_size: int,
    include_operation_names: bool,
    assistant_style: str,
    max_attempt_multiplier: int,
) -> dict[str, Any]:
    """Generate canonical + OAI datasets for selected domains and return a manifest."""
    output_dir = output_dir.expanduser().resolve()
    canonical_root = output_dir / "canonical"
    oai_root = output_dir / "oai_chat"
    canonical_root.mkdir(parents=True, exist_ok=True)
    oai_root.mkdir(parents=True, exist_ok=True)

    all_oai_writer = ShardedJsonlWriter(oai_root / "all_domains", "all_domains", shard_size=shard_size)
    manifest_domains: list[dict[str, Any]] = []

    domain_jobs = _resolve_domain_jobs(domains, profile=profile)

    for job in domain_jobs:
        domain_name = job.output_domain
        solver = DOMAIN_SOLVERS[job.solver_domain]
        domain_seed = _stable_domain_seed(seed, domain_name)
        rng = np.random.default_rng(domain_seed)
        canonical_writer = ShardedJsonlWriter(canonical_root / domain_name, domain_name, shard_size=shard_size)
        oai_writer = ShardedJsonlWriter(oai_root / domain_name, domain_name, shard_size=shard_size)

        emitted = 0
        attempts = 0
        failed_verification = 0
        step_total = 0
        max_attempts = max(examples_per_domain * max_attempt_multiplier, examples_per_domain)
        while emitted < examples_per_domain and attempts < max_attempts:
            attempts += 1
            example = solver.generate(rng)
            verified, reason = solver.verify(example)
            if not verified:
                failed_verification += 1
                continue

            row_id = f"{domain_name}_{seed}_{emitted:06d}"
            canonical_row = _make_canonical_row(
                domain_name=domain_name,
                solver_domain=job.solver_domain,
                profile=job.profile,
                profile_task=job.profile_task,
                row_id=row_id,
                base_seed=seed,
                domain_seed=domain_seed,
                index=emitted,
                example=example,
                verified=verified,
                verify_reason=reason,
            )
            oai_row = _make_oai_row(
                canonical_row=canonical_row,
                example=example,
                include_operation_names=include_operation_names,
                assistant_style=assistant_style,
            )

            canonical_writer.write(canonical_row)
            oai_writer.write(oai_row)
            all_oai_writer.write(oai_row)
            emitted += 1
            step_total += len(example.steps)

        canonical_writer.close()
        oai_writer.close()
        if emitted < examples_per_domain:
            raise RuntimeError(
                f"Domain {domain_name}: emitted={emitted} < requested={examples_per_domain}, attempts={attempts}"
            )

        manifest_domains.append(
            {
                "domain": domain_name,
                "solver_domain": job.solver_domain,
                "profile": job.profile,
                "profile_task": job.profile_task,
                "requested_examples": examples_per_domain,
                "emitted_examples": emitted,
                "attempts": attempts,
                "failed_verification": failed_verification,
                "verification_pass_rate": emitted / attempts if attempts else 0.0,
                "avg_steps_per_example": step_total / emitted if emitted else 0.0,
                "canonical_files": canonical_writer.paths,
                "oai_files": oai_writer.paths,
            }
        )

    all_oai_writer.close()

    totals = {
        "domains": len(domain_jobs),
        "examples_requested": examples_per_domain * len(domain_jobs),
        "examples_emitted": sum(int(row["emitted_examples"]) for row in manifest_domains),
        "combined_oai_files": all_oai_writer.paths,
    }
    manifest = {
        "schema_version": "synthetic_reasoning_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "seed": seed,
        "examples_per_domain": examples_per_domain,
        "shard_size": shard_size,
        "include_operation_names": include_operation_names,
        "assistant_style": assistant_style,
        "domains": manifest_domains,
        "totals": totals,
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a multi-domain mechanical synthetic reasoning dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for canonical/oai_chat shards and manifest.json",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["all"],
        help="Domain names or 'all'. Names depend on --profile. Use --list-domains to inspect available names.",
    )
    parser.add_argument(
        "--profile",
        choices=["native", "clrs_style"],
        default="native",
        help="Domain namespace/profile. 'clrs_style' maps CLRS-like aliases onto native solvers.",
    )
    parser.add_argument("--examples-per-domain", type=int, default=1000, help="Rows to emit per domain")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--shard-size", type=int, default=2000, help="Rows per shard file")
    parser.add_argument(
        "--assistant-style",
        choices=["symbolic", "nl_template", "hybrid"],
        default="symbolic",
        help="Assistant response rendering adapter for OAI rows.",
    )
    parser.add_argument(
        "--include-operation-names",
        action="store_true",
        help="Include operation labels in symbolic/hybrid assistant step text",
    )
    parser.add_argument("--max-attempt-multiplier", type=int, default=20, help="Max attempts = examples * multiplier")
    parser.add_argument("--list-domains", action="store_true", help="Print domain list and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_domains:
        names = list_domains() if args.profile == "native" else list_clrs_domains()
        for name in names:
            print(name)
        return

    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --list-domains is set")

    if args.examples_per_domain <= 0:
        raise ValueError("--examples-per-domain must be positive")
    if args.shard_size <= 0:
        raise ValueError("--shard-size must be positive")

    manifest = generate_dataset(
        output_dir=args.output_dir,
        domains=args.domains,
        profile=args.profile,
        examples_per_domain=args.examples_per_domain,
        seed=args.seed,
        shard_size=args.shard_size,
        include_operation_names=args.include_operation_names,
        assistant_style=args.assistant_style,
        max_attempt_multiplier=args.max_attempt_multiplier,
    )
    print(json.dumps({"status": "ok", "totals": manifest["totals"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
