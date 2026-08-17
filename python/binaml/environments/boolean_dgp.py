"""Input DGP and boolean function sampler shared by synthetic environments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from itertools import combinations
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class BooleanDgpConfig:
    n_features: int
    schema_version: int = 4
    q_max: int = 0
    p_min: float = 0.05
    p_max: float = 0.95
    p_sample_min_x: float = 1.0
    p_sample_max_x: float = 1.0
    truth_table_function_probability: float = 1.0
    min_truth_table_function_arity: int = 1
    max_truth_table_function_arity: int = 3
    min_hamming_threshold_function_arity: int = 1
    max_hamming_threshold_function_arity: int = 3
    p_activation_min: float = 0.2
    p_activation_max: float = 0.8

    def __post_init__(self) -> None:
        integer_fields = (
            "schema_version",
            "n_features",
            "q_max",
            "max_truth_table_function_arity",
            "min_truth_table_function_arity",
            "max_hamming_threshold_function_arity",
        )
        if any(isinstance(getattr(self, field), bool) or not isinstance(getattr(self, field), int) for field in integer_fields):
            raise TypeError("integer configuration fields must be integers, not booleans")
        if self.schema_version != 4 or self.n_features < 1 or self.q_max < 0:
            raise ValueError("unsupported schema version or invalid dimensions")
        numeric = asdict(self)
        if not all(math.isfinite(float(value)) for key, value in numeric.items() if key not in integer_fields):
            raise ValueError("numeric configuration fields must be finite")
        probability_ranges = (
            (self.p_min, self.p_max),
            (self.p_sample_min_x, self.p_sample_max_x),
            (self.p_activation_min, self.p_activation_max),
        )
        if any(not 0 <= low <= high <= 1 for low, high in probability_ranges):
            raise ValueError("probability ranges must lie in [0, 1]")
        if not 0 <= self.truth_table_function_probability <= 1:
            raise ValueError("probabilities must lie in [0, 1]")
        if not 1 <= self.min_truth_table_function_arity <= self.max_truth_table_function_arity <= self.n_features:
            raise ValueError("invalid truth-table arity range")
        if not 1 <= self.min_hamming_threshold_function_arity <= self.max_hamming_threshold_function_arity <= self.n_features:
            raise ValueError("invalid Hamming-threshold arity range")

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
            truth_table_function_probability=config.truth_table_function_probability,  # type: ignore[attr-defined]
            min_truth_table_function_arity=getattr(config, "min_truth_table_function_arity", 1),
            max_truth_table_function_arity=config.max_truth_table_function_arity,  # type: ignore[attr-defined]
            min_hamming_threshold_function_arity=config.min_hamming_threshold_function_arity,  # type: ignore[attr-defined]
            max_hamming_threshold_function_arity=config.max_hamming_threshold_function_arity,  # type: ignore[attr-defined]
            p_activation_min=config.p_activation_min,  # type: ignore[attr-defined]
            p_activation_max=config.p_activation_max,  # type: ignore[attr-defined]
        )

    @classmethod
    def from_classification_config(cls, config: object) -> BooleanDgpConfig:
        return cls.from_regression_config(config)


@dataclass(frozen=True)
class BinaryFunctionSpec:
    feature_indices: tuple[int, ...]
    family: Literal["truth_table", "hamming_threshold"]
    truth_table: tuple[int, ...] | None = None
    activation_probability: float | None = None
    threshold: int | None = None

    def __post_init__(self) -> None:
        if not self.feature_indices or len(set(self.feature_indices)) != len(self.feature_indices):
            raise ValueError("feature_indices must be non-empty and distinct")
        if self.family == "truth_table":
            if self.truth_table is None or len(self.truth_table) != 2 ** len(self.feature_indices):
                raise ValueError("invalid truth-table function")
            if any(value not in (0, 1) for value in self.truth_table):
                raise ValueError("truth_table values must be binary")
            if self.activation_probability is None or not 0 <= self.activation_probability <= 1:
                raise ValueError("invalid activation probability")
        elif self.family == "hamming_threshold":
            if self.threshold is None or not 1 <= self.threshold <= len(self.feature_indices):
                raise ValueError("invalid Hamming threshold")
        else:
            raise ValueError("unknown function family")

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_indices": list(self.feature_indices),
            "family": self.family,
            "truth_table": list(self.truth_table) if self.truth_table is not None else None,
            "activation_probability": self.activation_probability,
            "threshold": self.threshold,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BinaryFunctionSpec:
        table = value.get("truth_table")
        return cls(
            tuple(value["feature_indices"]),
            value["family"],
            tuple(table) if table is not None else None,
            value.get("activation_probability"),
            value.get("threshold"),
        )

    def evaluate(self, x: np.ndarray) -> int:
        if self.family == "hamming_threshold":
            return int(sum(int(x[feature]) for feature in self.feature_indices) >= self.threshold)  # type: ignore[operator]
        index = 0
        for feature in self.feature_indices:
            index = (index << 1) | int(x[feature])
        return self.truth_table[index]  # type: ignore[index]


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


@dataclass(frozen=True)
class BooleanBatch:
    X: np.ndarray
    target_function: BinaryFunctionSpec
    config: BooleanDgpConfig
    seed: int


def sample_binary_function(rng: np.random.Generator, config: BooleanDgpConfig) -> BinaryFunctionSpec:
    if rng.random() < config.truth_table_function_probability:
        arity = int(
            rng.integers(
                config.min_truth_table_function_arity,
                config.max_truth_table_function_arity + 1,
            )
        )
        indices = tuple(int(index) for index in rng.choice(config.n_features, arity, replace=False))
        activation = float(rng.uniform(config.p_activation_min, config.p_activation_max))
        table = tuple(int(value) for value in rng.binomial(1, activation, 2**arity))
        return BinaryFunctionSpec(indices, "truth_table", table, activation)
    arity = int(rng.integers(config.min_hamming_threshold_function_arity, config.max_hamming_threshold_function_arity + 1))
    indices = tuple(int(index) for index in rng.choice(config.n_features, arity, replace=False))
    return BinaryFunctionSpec(indices, "hamming_threshold", threshold=int(rng.integers(1, arity + 1)))


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


def generate_boolean_batch(config: BooleanDgpConfig, n_samples: int, seed: int) -> BooleanBatch:
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 0:
        raise ValueError("n_samples must be a non-negative integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**128:
        raise ValueError("seed must be an integer in [0, 2**128)")
    target_rng, input_rng = (np.random.Generator(np.random.PCG64DXSM(stream)) for stream in np.random.SeedSequence(seed).spawn(2))
    target_function = sample_binary_function(target_rng, config)
    input_dag = sample_conditional_dag(
        input_rng,
        width=config.n_features,
        q_max=config.q_max,
        p_min=config.p_min,
        p_max=config.p_max,
        p_sample_min=config.p_sample_min_x,
        p_sample_max=config.p_sample_max_x,
    )
    X = np.empty((n_samples, config.n_features), dtype=np.uint8)
    for index in range(n_samples):
        X[index] = sample_ancestrally(input_dag, input_rng)
    return BooleanBatch(X, target_function, config, seed)
