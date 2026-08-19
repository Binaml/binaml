"""Input DGP and ground-truth function sampler shared by synthetic environments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from itertools import combinations
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BooleanDgpConfig:
    n_features: int
    q_max: int = 0
    p_min: float = 0.05
    p_max: float = 0.95
    p_sample_min_x: float = 1.0
    p_sample_max_x: float = 1.0
    min_term_degree: int = 1
    max_term_degree: int = 7
    p_negated_literal: float = 0.5

    def __post_init__(self) -> None:
        integer_fields = (
            "n_features",
            "q_max",
            "min_term_degree",
            "max_term_degree",
        )
        if any(isinstance(getattr(self, field), bool) or not isinstance(getattr(self, field), int) for field in integer_fields):
            raise TypeError("integer configuration fields must be integers, not booleans")
        if self.n_features < 1 or self.q_max < 0:
            raise ValueError("invalid dimensions")
        numeric = asdict(self)
        if not all(math.isfinite(float(value)) for key, value in numeric.items() if key not in integer_fields):
            raise ValueError("numeric configuration fields must be finite")
        probability_ranges = (
            (self.p_min, self.p_max),
            (self.p_sample_min_x, self.p_sample_max_x),
        )
        if any(not 0 <= low <= high <= 1 for low, high in probability_ranges):
            raise ValueError("probability ranges must lie in [0, 1]")
        if not 0 <= self.p_negated_literal <= 1:
            raise ValueError("probabilities must lie in [0, 1]")
        if not 1 <= self.min_term_degree <= self.max_term_degree <= self.n_features:
            raise ValueError("invalid term degree range")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BooleanDgpConfig:
        known = {field.name for field in fields(cls)}
        return cls(**{key: item for key, item in value.items() if key in known})

    @classmethod
    def from_regression_config(cls, config: object) -> BooleanDgpConfig:
        return cls(
            n_features=config.n_features,  # type: ignore[attr-defined]
            q_max=config.q_max,  # type: ignore[attr-defined]
            p_min=config.p_min,  # type: ignore[attr-defined]
            p_max=config.p_max,  # type: ignore[attr-defined]
            p_sample_min_x=config.p_sample_min_x,  # type: ignore[attr-defined]
            p_sample_max_x=config.p_sample_max_x,  # type: ignore[attr-defined]
            min_term_degree=config.min_term_degree,  # type: ignore[attr-defined]
            max_term_degree=config.max_term_degree,  # type: ignore[attr-defined]
            p_negated_literal=config.p_negated_literal,  # type: ignore[attr-defined]
        )

    @classmethod
    def from_classification_config(cls, config: object) -> BooleanDgpConfig:
        return cls.from_regression_config(config)


@dataclass(frozen=True)
class ConjunctionTerm:
    """One conjunction of literals."""

    feature_indices: tuple[int, ...]
    negated: tuple[bool, ...]

    def __post_init__(self) -> None:
        if len(self.feature_indices) != len(self.negated):
            raise ValueError("negated mask must match feature_indices length")
        if not self.feature_indices or len(set(self.feature_indices)) != len(self.feature_indices):
            raise ValueError("feature_indices must be non-empty and distinct")

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_indices": list(self.feature_indices),
            "negated": list(self.negated),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ConjunctionTerm:
        return cls(
            tuple(value["feature_indices"]),
            tuple(value["negated"]),
        )

    def evaluate(self, x: np.ndarray) -> int:
        return int(all((not x[i]) if flip else x[i] for i, flip in zip(self.feature_indices, self.negated, strict=True)))


@dataclass(frozen=True)
class FunctionSpec:
    """Ground-truth f_k as one conjunction of literals."""

    term: ConjunctionTerm

    def to_dict(self) -> dict[str, Any]:
        return self.term.to_dict()

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FunctionSpec:
        if "family" in value:
            raise ValueError("unsupported legacy function spec with family field")
        if "constant" in value or "terms" in value:
            raise ValueError("unsupported legacy ANF function spec")
        return cls(ConjunctionTerm.from_dict(value))

    def evaluate(self, x: np.ndarray) -> int:
        return self.term.evaluate(x)


@dataclass(frozen=True)
class ConditionalDag:
    order: tuple[int, ...]
    parents: tuple[tuple[int, ...], ...]
    cpts: tuple[tuple[float, ...], ...]
    node_sampling_probability: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": list(self.order),
            "parents": [list(parent_list) for parent_list in self.parents],
            "cpts": [list(cpt) for cpt in self.cpts],
            "node_sampling_probability": self.node_sampling_probability,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any], width: int) -> ConditionalDag:
        order = tuple(int(node) for node in value["order"])
        parents = tuple(tuple(int(parent) for parent in parent_list) for parent_list in value["parents"])
        cpts = tuple(tuple(float(probability) for probability in cpt) for cpt in value["cpts"])
        probability = float(value["node_sampling_probability"])
        if sorted(order) != list(range(width)) or len(parents) != width or len(cpts) != width:
            raise ValueError("invalid DAG shape")
        positions = {node: position for position, node in enumerate(order)}
        if not 0 <= probability <= 1:
            raise ValueError("invalid node sampling probability")
        for node, parent_list, cpt in zip(range(width), parents, cpts, strict=True):
            if len(set(parent_list)) != len(parent_list) or any(parent not in positions for parent in parent_list):
                raise ValueError("invalid DAG parents")
            if any(positions[parent] >= positions[node] for parent in parent_list):
                raise ValueError("DAG parent must precede its child")
            if len(cpt) != 2 ** len(parent_list) or any(not 0 <= entry <= 1 for entry in cpt):
                raise ValueError("invalid CPT")
        return cls(order, parents, cpts, probability)


def sample_conjunction_term(rng: np.random.Generator, config: BooleanDgpConfig) -> ConjunctionTerm:
    degree = int(rng.integers(config.min_term_degree, config.max_term_degree + 1))
    indices = tuple(int(index) for index in rng.choice(config.n_features, degree, replace=False))
    negated = tuple(bool(rng.random() < config.p_negated_literal) for _ in indices)
    return ConjunctionTerm(indices, negated)


def sample_function(rng: np.random.Generator, config: BooleanDgpConfig) -> FunctionSpec:
    return FunctionSpec(sample_conjunction_term(rng, config))


def sample_conditional_dag(
    rng: np.random.Generator,
    *,
    width: int,
    q_max: int,
    p_min: float,
    p_max: float,
    p_sample_min: float,
    p_sample_max: float,
) -> ConditionalDag:
    order = tuple(int(node) for node in rng.permutation(width))
    parents: list[tuple[int, ...]] = [()] * width
    cpts: list[tuple[float, ...]] = [()] * width
    for position, node in enumerate(order):
        earlier = order[:position]
        subsets = [subset for size in range(min(q_max, position) + 1) for subset in combinations(earlier, size)]
        parent_list = subsets[int(rng.integers(len(subsets)))]
        parents[node] = parent_list
        cpts[node] = tuple(float(value) for value in rng.uniform(p_min, p_max, 2 ** len(parent_list)))
    return ConditionalDag(order, tuple(parents), tuple(cpts), float(rng.uniform(p_sample_min, p_sample_max)))


def cpt_probability(dag: ConditionalDag, state: np.ndarray, node: int) -> float:
    index = 0
    for parent in dag.parents[node]:
        index = (index << 1) | int(state[parent])
    return dag.cpts[node][index]


def sample_ancestrally(dag: ConditionalDag, rng: np.random.Generator) -> np.ndarray:
    state = np.zeros(len(dag.order), dtype=np.uint8)
    for node in dag.order:
        state[node] = int(rng.random() < cpt_probability(dag, state, node))
    return state
