"""Shared scenario loading and grid expansion for benchmarks."""

from __future__ import annotations

import copy
import json
from itertools import product
from pathlib import Path

_TOP_LEVEL_KEYS = frozenset({"name", "n_samples", "n_train", "n_test", "warmup_samples", "seeds", "p_noise"})


def load_scenario(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("scenario must be a JSON object")
    return payload


def warmup_samples(scenario: dict[str, object]) -> int:
    """Return the warmup prefix length for streaming (prequential) benchmarks."""
    warmup = scenario.get("warmup_samples", 0)
    if not isinstance(warmup, int) or isinstance(warmup, bool):
        raise TypeError("warmup_samples must be an integer")
    n_samples = scenario["n_samples"]
    if not isinstance(n_samples, int) or isinstance(n_samples, bool):
        raise TypeError("n_samples must be an integer")
    if not 0 <= warmup < n_samples:
        raise ValueError("warmup_samples must be non-negative and less than n_samples")
    return warmup


def expand_grid(scenario: dict[str, object]) -> list[dict[str, object]]:
    grid = scenario.get("grid")
    if not grid:
        return [scenario]
    if not isinstance(grid, dict) or not grid:
        raise TypeError("grid must be a non-empty object")
    keys = list(grid.keys())
    value_lists = []
    for key in keys:
        values = grid[key]
        if not isinstance(values, list) or not values:
            raise TypeError(f"grid[{key!r}] must be a non-empty list")
        value_lists.append(values)
    expanded: list[dict[str, object]] = []
    for combo in product(*value_lists):
        variant = copy.deepcopy(scenario)
        variant.pop("grid", None)
        for key, value in zip(keys, combo, strict=True):
            if key in _TOP_LEVEL_KEYS or key in variant:
                variant[key] = value
            else:
                environment = variant.setdefault("environment", {})
                if not isinstance(environment, dict):
                    raise TypeError("environment must be an object")
                environment[key] = value
        expanded.append(variant)
    return expanded
