"""Reproducible synthetic binary classification stream with paper-defined drift."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from .boolean_dgp import (
    BooleanDgpConfig,
    ConditionalDag,
    FunctionSpec,
    cpt_probability,
    sample_ancestrally,
    sample_conditional_dag,
    sample_function,
)
from .synthetic_drifting_regression import _canonical_json, _jsonify

GENERATOR_VERSION = "5.0.0-numpy-pcg64dxsm-anf"


@dataclass(frozen=True)
class SyntheticClassificationStreamConfig:
    n_features: int
    n_functions: int
    n_classes: int
    q_max: int = 0
    p_min: float = 0.05
    p_max: float = 0.95
    p_x: float = 0.0
    p_sample_min_x: float = 1.0
    p_sample_max_x: float = 1.0
    p_g: float = 0.0
    p_sample_min_g: float = 1.0
    p_sample_max_g: float = 1.0
    min_n_terms: int = 1
    max_n_terms: int = 10
    min_term_degree: int = 1
    max_term_degree: int = 7
    p_negated_literal: float = 0.5
    w_min: float = -1.0
    w_max: float = 1.0
    b_min: float = 0.0
    b_max: float = 0.0
    p_b: float = 0.0
    noise_std: float = 0.1
    weights: tuple[tuple[float, ...], ...] | None = None
    intercepts: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        integer_fields = (
            "n_features",
            "n_functions",
            "n_classes",
            "q_max",
            "min_n_terms",
            "max_n_terms",
            "min_term_degree",
            "max_term_degree",
        )
        if any(isinstance(getattr(self, field), bool) or not isinstance(getattr(self, field), int) for field in integer_fields):
            raise TypeError("integer configuration fields must be integers, not booleans")
        if self.n_features < 1 or self.n_functions < 1 or self.n_classes < 2 or self.q_max < 0:
            raise ValueError("invalid dimensions")
        numeric = asdict(self)
        if not all(
            math.isfinite(float(value))
            for key, value in numeric.items()
            if key not in integer_fields and key not in ("weights", "intercepts")
        ):
            raise ValueError("numeric configuration fields must be finite")
        probability_ranges = (
            (self.p_min, self.p_max),
            (self.p_sample_min_x, self.p_sample_max_x),
            (self.p_sample_min_g, self.p_sample_max_g),
        )
        if any(not 0 <= low <= high <= 1 for low, high in probability_ranges):
            raise ValueError("probability ranges must lie in [0, 1]")
        if any(not 0 <= value <= 1 for value in (self.p_x, self.p_g, self.p_b, self.p_negated_literal)):
            raise ValueError("probabilities must lie in [0, 1]")
        if self.w_min > self.w_max or self.b_min > self.b_max or self.noise_std < 0:
            raise ValueError("invalid scale range or noise standard deviation")
        if not 1 <= self.min_n_terms <= self.max_n_terms:
            raise ValueError("invalid ANF term-count range")
        if not 1 <= self.min_term_degree <= self.max_term_degree <= self.n_features:
            raise ValueError("invalid term degree range")
        if self.weights is not None:
            if len(self.weights) != self.n_classes or any(len(row) != self.n_functions for row in self.weights):
                raise ValueError("weights shape must match (n_classes, n_functions)")
        if self.intercepts is not None and len(self.intercepts) != self.n_classes:
            raise ValueError("intercepts length must match n_classes")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.weights is not None:
            payload["weights"] = [list(row) for row in self.weights]
        if self.intercepts is not None:
            payload["intercepts"] = list(self.intercepts)
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SyntheticClassificationStreamConfig:
        known = {field.name for field in fields(cls)}
        weights = value.get("weights")
        intercepts = value.get("intercepts")
        return cls(
            **{key: item for key, item in value.items() if key in known and key not in ("weights", "intercepts")},
            weights=tuple(tuple(float(entry) for entry in row) for row in weights) if weights is not None else None,
            intercepts=tuple(float(entry) for entry in intercepts) if intercepts is not None else None,
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode()).hexdigest()


@dataclass(frozen=True)
class ClassificationTrajectory:
    X: np.ndarray
    y: np.ndarray
    config: SyntheticClassificationStreamConfig
    seed: int
    metadata: list[dict[str, Any]] | None = None

    def __iter__(self) -> Iterator[tuple[np.ndarray, int]]:
        return iter(zip(self.X, self.y, strict=True))

    def save_npz(self, path: str | Path) -> None:
        payload: dict[str, Any] = {
            "X": np.ascontiguousarray(self.X, dtype=np.uint8),
            "y": np.ascontiguousarray(self.y, dtype=np.int64),
            "config_json": np.frombuffer(_canonical_json(self.config.to_dict()).encode(), dtype=np.uint8),
            "seed": np.asarray(self.seed, dtype=np.uint64),
            "generator_version": np.frombuffer(GENERATOR_VERSION.encode(), dtype=np.uint8),
        }
        if self.metadata is not None:
            payload["metadata_json"] = np.frombuffer(_canonical_json(self.metadata).encode(), dtype=np.uint8)
        np.savez(path, **payload)

    @classmethod
    def load_npz(cls, path: str | Path) -> ClassificationTrajectory:
        with np.load(path, allow_pickle=False) as archive:
            required = {"X", "y", "config_json", "seed", "generator_version"}
            if not required.issubset(archive.files):
                raise ValueError("trajectory artifact is missing required fields")
            version = bytes(archive["generator_version"].tolist()).decode()
            if version != GENERATOR_VERSION:
                raise ValueError(f"unsupported generator version: {version}")
            config = SyntheticClassificationStreamConfig.from_dict(json.loads(bytes(archive["config_json"].tolist()).decode()))
            X = np.asarray(archive["X"], dtype=np.uint8)
            y = np.asarray(archive["y"], dtype=np.int64)
            if X.ndim != 2 or X.shape[1] != config.n_features or y.shape != (len(X),):
                raise ValueError("trajectory feature or target shape is invalid")
            if np.any(y < 0) or np.any(y >= config.n_classes):
                raise ValueError("trajectory labels must lie in [0, n_classes)")
            metadata = json.loads(bytes(archive["metadata_json"].tolist()).decode()) if "metadata_json" in archive.files else None
            return cls(X, y, config, int(archive["seed"]), metadata)


class SyntheticDriftingClassificationStream(Iterator[tuple[np.ndarray, int]]):
    """Infinite paper-defined stream; every step emits then advances its state."""

    def __init__(self, config: SyntheticClassificationStreamConfig, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**128:
            raise ValueError("seed must be an integer in [0, 2**128)")
        self.config, self.seed, self.time = config, seed, 0
        stream_count = 5 + 4 * config.n_classes
        streams = np.random.SeedSequence(seed).spawn(stream_count)
        generators = [np.random.Generator(np.random.PCG64DXSM(stream)) for stream in streams]
        self._target_rng = generators[0]
        self._input_distribution_rng = generators[1]
        self._input_drift_rng = generators[2]
        self._input_sampling_rng = generators[3]
        self._noise_rng = generators[4]
        self._gate_distribution_rngs = []
        self._gate_drift_rngs = []
        self._gate_sampling_rngs = []
        self._intercept_rngs = []
        offset = 5
        for class_index in range(config.n_classes):
            base = offset + 4 * class_index
            self._gate_distribution_rngs.append(generators[base])
            self._gate_drift_rngs.append(generators[base + 1])
            self._gate_sampling_rngs.append(generators[base + 2])
            self._intercept_rngs.append(generators[base + 3])
        dgp_config = BooleanDgpConfig.from_classification_config(config)
        self.functions = [sample_function(self._target_rng, dgp_config) for _ in range(config.n_functions)]
        if config.weights is not None:
            self.weights = np.asarray(config.weights, dtype=float)
        else:
            self.weights = self._target_rng.uniform(
                config.w_min,
                config.w_max,
                (config.n_classes, config.n_functions),
            )
        self.input_dag = self._sample_dag(
            config.n_features,
            config.p_sample_min_x,
            config.p_sample_max_x,
            self._input_distribution_rng,
        )
        self.gate_dags = [
            self._sample_dag(
                config.n_functions,
                config.p_sample_min_g,
                config.p_sample_max_g,
                self._gate_distribution_rngs[class_index],
            )
            for class_index in range(config.n_classes)
        ]
        self.input_state = sample_ancestrally(self.input_dag, self._input_distribution_rng)
        self.gate_states = [
            sample_ancestrally(self.gate_dags[class_index], self._gate_distribution_rngs[class_index])
            for class_index in range(config.n_classes)
        ]
        if config.intercepts is not None:
            self.intercepts = [float(value) for value in config.intercepts]
        else:
            self.intercepts = [
                float(self._intercept_rngs[class_index].uniform(config.b_min, config.b_max))
                for class_index in range(config.n_classes)
            ]

    def __iter__(self) -> SyntheticDriftingClassificationStream:
        return self

    def __next__(self) -> tuple[np.ndarray, int]:
        return self.next_sample()

    def _sample_dag(self, width: int, p_sample_min: float, p_sample_max: float, rng: np.random.Generator) -> ConditionalDag:
        return sample_conditional_dag(
            rng,
            width=width,
            q_max=self.config.q_max,
            p_min=self.config.p_min,
            p_max=self.config.p_max,
            p_sample_min=p_sample_min,
            p_sample_max=p_sample_max,
        )

    def _partial_ancestral_sample(self, state: np.ndarray, dag: ConditionalDag, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        mask = rng.binomial(1, dag.node_sampling_probability, len(state)).astype(bool)
        next_state = state.copy()
        for node in dag.order:
            if mask[node]:
                next_state[node] = int(rng.random() < cpt_probability(dag, next_state, node))
        return next_state, mask

    def _class_score(
        self,
        class_index: int,
        features: np.ndarray,
        gate_state: np.ndarray,
        intercept: float,
    ) -> float:
        return intercept + float(
            sum(
                self.weights[class_index, function_index]
                * gate_state[function_index]
                * function.evaluate(features)
                for function_index, function in enumerate(self.functions)
            )
        )

    def next_sample(self, return_metadata: bool = False) -> tuple[np.ndarray, int] | tuple[np.ndarray, int, dict[str, Any]]:
        x = self.input_state.copy()
        gate_states = [gate_state.copy() for gate_state in self.gate_states]
        intercepts = list(self.intercepts)
        class_scores = [
            self._class_score(class_index, x, gate_states[class_index], intercepts[class_index])
            for class_index in range(self.config.n_classes)
        ]
        score_noise = [
            float(self._noise_rng.normal(0, self.config.noise_std)) for _ in range(self.config.n_classes)
        ]
        noisy_scores = [score + noise for score, noise in zip(class_scores, score_noise, strict=True)]
        label = int(np.argmax(noisy_scores))
        input_distribution_sampled = bool(self._input_drift_rng.random() < self.config.p_x)
        if input_distribution_sampled:
            self.input_dag = self._sample_dag(
                self.config.n_features,
                self.config.p_sample_min_x,
                self.config.p_sample_max_x,
                self._input_distribution_rng,
            )
        gate_distribution_sampled = []
        gate_sampling_masks = []
        intercept_sampled = []
        for class_index in range(self.config.n_classes):
            gate_distribution_sampled.append(bool(self._gate_drift_rngs[class_index].random() < self.config.p_g))
            if gate_distribution_sampled[class_index]:
                self.gate_dags[class_index] = self._sample_dag(
                    self.config.n_functions,
                    self.config.p_sample_min_g,
                    self.config.p_sample_max_g,
                    self._gate_distribution_rngs[class_index],
                )
            self.gate_states[class_index], gate_mask = self._partial_ancestral_sample(
                self.gate_states[class_index],
                self.gate_dags[class_index],
                self._gate_sampling_rngs[class_index],
            )
            gate_sampling_masks.append(gate_mask.tolist())
            intercept_sampled.append(bool(self._intercept_rngs[class_index].random() < self.config.p_b))
            if intercept_sampled[class_index]:
                self.intercepts[class_index] = float(
                    self._intercept_rngs[class_index].uniform(self.config.b_min, self.config.b_max)
                )
        self.input_state, input_sampling_mask = self._partial_ancestral_sample(
            self.input_state,
            self.input_dag,
            self._input_sampling_rng,
        )
        metadata = {
            "time": self.time,
            "class_scores": class_scores,
            "noisy_class_scores": noisy_scores,
            "score_noise": score_noise,
            "label": label,
            "intercepts": intercepts,
            "weights": self.weights.tolist(),
            "functions": [function.to_dict() for function in self.functions],
            "input_state": x.tolist(),
            "gate_states": [gate_state.tolist() for gate_state in gate_states],
            "input_distribution_sampling_indicator": input_distribution_sampled,
            "gate_distribution_sampling_indicator": any(gate_distribution_sampled),
            "gate_distribution_sampling_indicators": gate_distribution_sampled,
            "input_node_sampling_mask": input_sampling_mask.tolist(),
            "gate_node_sampling_masks": gate_sampling_masks,
            "intercept_sampling_indicator": any(intercept_sampled),
            "intercept_sampling_indicators": intercept_sampled,
            "input_distribution": self.input_dag.to_dict(),
            "gate_distributions": [gate_dag.to_dict() for gate_dag in self.gate_dags],
        }
        self.time += 1
        return (x, label, metadata) if return_metadata else (x, label)

    def get_state(self) -> dict[str, Any]:
        rngs = [
            self._target_rng,
            self._input_distribution_rng,
            self._input_drift_rng,
            self._input_sampling_rng,
            self._noise_rng,
        ]
        for class_index in range(self.config.n_classes):
            rngs.extend(
                [
                    self._gate_distribution_rngs[class_index],
                    self._gate_drift_rngs[class_index],
                    self._gate_sampling_rngs[class_index],
                    self._intercept_rngs[class_index],
                ]
            )
        return {
            "generator_version": GENERATOR_VERSION,
            "config_fingerprint": self.config.fingerprint,
            "time": self.time,
            "input_state": self.input_state.tolist(),
            "gate_states": [gate_state.tolist() for gate_state in self.gate_states],
            "input_distribution": self.input_dag.to_dict(),
            "gate_distributions": [gate_dag.to_dict() for gate_dag in self.gate_dags],
            "intercepts": list(self.intercepts),
            "weights": self.weights.tolist(),
            "functions": [function.to_dict() for function in self.functions],
            "rng_states": _jsonify([rng.bit_generator.state for rng in rngs]),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if state.get("generator_version") != GENERATOR_VERSION or state.get("config_fingerprint") != self.config.fingerprint:
            raise ValueError("incompatible generator state")
        input_state = np.asarray(state["input_state"], dtype=np.uint8)
        gate_states = [np.asarray(gate_state, dtype=np.uint8) for gate_state in state["gate_states"]]
        weights = np.asarray(state["weights"], dtype=float)
        functions = [FunctionSpec.from_dict(value) for value in state["functions"]]
        input_dag = ConditionalDag.from_dict(state["input_distribution"], self.config.n_features)
        gate_dags = [
            ConditionalDag.from_dict(gate_distribution, self.config.n_functions)
            for gate_distribution in state["gate_distributions"]
        ]
        intercepts = [float(value) for value in state["intercepts"]]
        time = state["time"]
        if (
            input_state.shape != (self.config.n_features,)
            or len(gate_states) != self.config.n_classes
            or any(gate_state.shape != (self.config.n_functions,) for gate_state in gate_states)
            or weights.shape != (self.config.n_classes, self.config.n_functions)
            or len(functions) != self.config.n_functions
            or len(gate_dags) != self.config.n_classes
            or len(intercepts) != self.config.n_classes
            or any(value not in (0, 1) for value in input_state)
            or any(value not in (0, 1) for gate_state in gate_states for value in gate_state)
            or not all(math.isfinite(value) for value in intercepts)
            or isinstance(time, bool)
            or not isinstance(time, int)
            or time < 0
        ):
            raise ValueError("invalid stream state")
        rng_states = state["rng_states"]
        rngs = [
            self._target_rng,
            self._input_distribution_rng,
            self._input_drift_rng,
            self._input_sampling_rng,
            self._noise_rng,
        ]
        for class_index in range(self.config.n_classes):
            rngs.extend(
                [
                    self._gate_distribution_rngs[class_index],
                    self._gate_drift_rngs[class_index],
                    self._gate_sampling_rngs[class_index],
                    self._intercept_rngs[class_index],
                ]
            )
        if not isinstance(rng_states, list) or len(rng_states) != len(rngs):
            raise ValueError("invalid RNG state")
        self.time, self.input_state = time, input_state
        self.gate_states, self.gate_dags = gate_states, gate_dags
        self.intercepts, self.weights, self.functions = intercepts, weights, functions
        self.input_dag = input_dag
        for rng, rng_state in zip(rngs, rng_states, strict=True):
            rng.bit_generator.state = rng_state


def generate_classification_trajectory(
    config: SyntheticClassificationStreamConfig,
    n_samples: int,
    seed: int,
    return_metadata: bool = False,
) -> ClassificationTrajectory:
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 0:
        raise ValueError("n_samples must be a non-negative integer")
    stream = SyntheticDriftingClassificationStream(config, seed)
    X = np.empty((n_samples, config.n_features), dtype=np.uint8)
    y = np.empty(n_samples, dtype=np.int64)
    metadata = [] if return_metadata else None
    for index in range(n_samples):
        if return_metadata:
            X[index], y[index], item_metadata = stream.next_sample(return_metadata=True)
            metadata.append(item_metadata)  # type: ignore[union-attr]
        else:
            X[index], y[index] = stream.next_sample()
    return ClassificationTrajectory(X, y, config, seed, metadata)
