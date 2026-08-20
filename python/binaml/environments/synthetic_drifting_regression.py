"""Reproducible synthetic binary regression stream with paper-defined drift."""

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

GENERATOR_VERSION = "7.0.0-numpy-pcg64dxsm-clause"


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonify(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


@dataclass(frozen=True)
class SyntheticStreamConfig:
    n_features: int
    n_functions: int
    q_max: int = 0
    p_min: float = 0.05
    p_max: float = 0.95
    p_x: float = 0.0
    p_sample_min_x: float = 1.0
    p_sample_max_x: float = 1.0
    p_g: float = 0.0
    p_sample_min_g: float = 1.0
    p_sample_max_g: float = 1.0
    min_term_degree: int = 1
    max_term_degree: int = 7
    p_negated_literal: float = 0.5
    w_min: float = -1.0
    w_max: float = 1.0
    b_min: float = 0.0
    b_max: float = 0.0
    p_b: float = 0.0
    noise_std: float = 0.1

    def __post_init__(self) -> None:
        integer_fields = (
            "n_features",
            "n_functions",
            "q_max",
            "min_term_degree",
            "max_term_degree",
        )
        if any(isinstance(getattr(self, field), bool) or not isinstance(getattr(self, field), int) for field in integer_fields):
            raise TypeError("integer configuration fields must be integers, not booleans")
        if self.n_features < 1 or self.n_functions < 1 or self.q_max < 0:
            raise ValueError("invalid dimensions")
        numeric = asdict(self)
        if not all(math.isfinite(float(value)) for key, value in numeric.items() if key not in integer_fields):
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
        if not 1 <= self.min_term_degree <= self.max_term_degree <= self.n_features:
            raise ValueError("invalid term degree range")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SyntheticStreamConfig:
        known = {field.name for field in fields(cls)}
        return cls(**{key: item for key, item in value.items() if key in known})

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode()).hexdigest()


@dataclass(frozen=True)
class Trajectory:
    X: np.ndarray
    y: np.ndarray
    config: SyntheticStreamConfig
    seed: int
    metadata: list[dict[str, Any]] | None = None

    def __iter__(self) -> Iterator[tuple[np.ndarray, float]]:
        return iter(zip(self.X, self.y, strict=True))

    def save_npz(self, path: str | Path) -> None:
        payload: dict[str, Any] = {
            "X": np.ascontiguousarray(self.X, dtype=np.uint8),
            "y": np.ascontiguousarray(self.y, dtype=np.float64),
            "config_json": np.frombuffer(_canonical_json(self.config.to_dict()).encode(), dtype=np.uint8),
            "seed": np.asarray(self.seed, dtype=np.uint64),
            "generator_version": np.frombuffer(GENERATOR_VERSION.encode(), dtype=np.uint8),
        }
        if self.metadata is not None:
            payload["metadata_json"] = np.frombuffer(_canonical_json(self.metadata).encode(), dtype=np.uint8)
        np.savez(path, **payload)

    @classmethod
    def load_npz(cls, path: str | Path) -> Trajectory:
        with np.load(path, allow_pickle=False) as archive:
            required = {"X", "y", "config_json", "seed", "generator_version"}
            if not required.issubset(archive.files):
                raise ValueError("trajectory artifact is missing required fields")
            version = bytes(archive["generator_version"].tolist()).decode()
            if version != GENERATOR_VERSION:
                raise ValueError(f"unsupported generator version: {version}")
            config = SyntheticStreamConfig.from_dict(json.loads(bytes(archive["config_json"].tolist()).decode()))
            X = np.asarray(archive["X"], dtype=np.uint8)
            y = np.asarray(archive["y"], dtype=np.float64)
            if X.ndim != 2 or X.shape[1] != config.n_features or y.shape != (len(X),):
                raise ValueError("trajectory feature or target shape is invalid")
            metadata = json.loads(bytes(archive["metadata_json"].tolist()).decode()) if "metadata_json" in archive.files else None
            return cls(X, y, config, int(archive["seed"]), metadata)


class SyntheticDriftingRegressionStream(Iterator[tuple[np.ndarray, float]]):
    """Infinite paper-defined stream; every step emits then advances its state."""

    def __init__(self, config: SyntheticStreamConfig, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**128:
            raise ValueError("seed must be an integer in [0, 2**128)")
        self.config, self.seed, self.time = config, seed, 0
        streams = np.random.SeedSequence(seed).spawn(9)
        self._target_rng, self._input_distribution_rng, self._gate_distribution_rng = (
            np.random.Generator(np.random.PCG64DXSM(stream)) for stream in streams[:3]
        )
        self._input_drift_rng, self._gate_drift_rng, self._input_sampling_rng = (
            np.random.Generator(np.random.PCG64DXSM(stream)) for stream in streams[3:6]
        )
        self._gate_sampling_rng, self._intercept_rng, self._noise_rng = (
            np.random.Generator(np.random.PCG64DXSM(stream)) for stream in streams[6:]
        )
        dgp_config = BooleanDgpConfig.from_regression_config(config)
        self.functions = [sample_function(self._target_rng, dgp_config) for _ in range(config.n_functions)]
        self.weights = self._target_rng.uniform(config.w_min, config.w_max, config.n_functions)
        self.intercept = float(self._target_rng.uniform(config.b_min, config.b_max))
        self.input_dag = self._sample_dag(config.n_features, config.p_sample_min_x, config.p_sample_max_x, self._input_distribution_rng)
        self.gate_dag = self._sample_dag(config.n_functions, config.p_sample_min_g, config.p_sample_max_g, self._gate_distribution_rng)
        self.input_state = sample_ancestrally(self.input_dag, self._input_distribution_rng)
        self.gate_state = sample_ancestrally(self.gate_dag, self._gate_distribution_rng)

    def __iter__(self) -> SyntheticDriftingRegressionStream:
        return self

    def __next__(self) -> tuple[np.ndarray, float]:
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

    def next_sample(self, return_metadata: bool = False) -> tuple[np.ndarray, float] | tuple[np.ndarray, float, dict[str, Any]]:
        x = self.input_state.copy()
        gate_state = self.gate_state.copy()
        intercept = self.intercept
        latent = intercept + float(
            sum(
                weight * gate * function.evaluate(x)
                for weight, gate, function in zip(self.weights, gate_state, self.functions, strict=True)
            )
        )
        noise = float(self._noise_rng.normal(0, self.config.noise_std))
        input_distribution_sampled = bool(self._input_drift_rng.random() < self.config.p_x)
        gate_distribution_sampled = bool(self._gate_drift_rng.random() < self.config.p_g)
        if input_distribution_sampled:
            self.input_dag = self._sample_dag(self.config.n_features, self.config.p_sample_min_x, self.config.p_sample_max_x, self._input_distribution_rng)
        if gate_distribution_sampled:
            self.gate_dag = self._sample_dag(self.config.n_functions, self.config.p_sample_min_g, self.config.p_sample_max_g, self._gate_distribution_rng)
        self.input_state, input_sampling_mask = self._partial_ancestral_sample(self.input_state, self.input_dag, self._input_sampling_rng)
        if self.config.p_g > 0:
            self.gate_state, gate_sampling_mask = self._partial_ancestral_sample(
                self.gate_state, self.gate_dag, self._gate_sampling_rng
            )
        else:
            gate_sampling_mask = np.zeros(len(self.gate_state), dtype=bool)
        intercept_sampled = bool(self._intercept_rng.random() < self.config.p_b)
        if intercept_sampled:
            self.intercept = float(self._intercept_rng.uniform(self.config.b_min, self.config.b_max))
        metadata = {
            "time": self.time,
            "latent_target": latent,
            "observed_target": latent + noise,
            "noise": noise,
            "intercept": intercept,
            "weights": self.weights.tolist(),
            "functions": [function.to_dict() for function in self.functions],
            "input_state": x.tolist(),
            "gate_state": gate_state.tolist(),
            "input_distribution_sampling_indicator": input_distribution_sampled,
            "gate_distribution_sampling_indicator": gate_distribution_sampled,
            "input_node_sampling_mask": input_sampling_mask.tolist(),
            "gate_node_sampling_mask": gate_sampling_mask.tolist(),
            "intercept_sampling_indicator": intercept_sampled,
            "input_distribution": self.input_dag.to_dict(),
            "gate_distribution": self.gate_dag.to_dict(),
        }
        self.time += 1
        return (x, latent + noise, metadata) if return_metadata else (x, latent + noise)

    def get_state(self) -> dict[str, Any]:
        rngs = (
            self._target_rng,
            self._input_distribution_rng,
            self._gate_distribution_rng,
            self._input_drift_rng,
            self._gate_drift_rng,
            self._input_sampling_rng,
            self._gate_sampling_rng,
            self._intercept_rng,
            self._noise_rng,
        )
        return {
            "generator_version": GENERATOR_VERSION,
            "config_fingerprint": self.config.fingerprint,
            "time": self.time,
            "input_state": self.input_state.tolist(),
            "gate_state": self.gate_state.tolist(),
            "input_distribution": self.input_dag.to_dict(),
            "gate_distribution": self.gate_dag.to_dict(),
            "intercept": self.intercept,
            "weights": self.weights.tolist(),
            "functions": [function.to_dict() for function in self.functions],
            "rng_states": _jsonify([rng.bit_generator.state for rng in rngs]),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if state.get("generator_version") != GENERATOR_VERSION or state.get("config_fingerprint") != self.config.fingerprint:
            raise ValueError("incompatible generator state")
        input_state = np.asarray(state["input_state"], dtype=np.uint8)
        gate_state = np.asarray(state["gate_state"], dtype=np.uint8)
        weights = np.asarray(state["weights"], dtype=float)
        functions = [FunctionSpec.from_dict(value) for value in state["functions"]]
        input_dag = ConditionalDag.from_dict(state["input_distribution"], self.config.n_features)
        gate_dag = ConditionalDag.from_dict(state["gate_distribution"], self.config.n_functions)
        intercept, time = float(state["intercept"]), state["time"]
        if (
            input_state.shape != (self.config.n_features,)
            or gate_state.shape != (self.config.n_functions,)
            or weights.shape != (self.config.n_functions,)
            or len(functions) != self.config.n_functions
            or any(value not in (0, 1) for value in input_state)
            or any(value not in (0, 1) for value in gate_state)
            or not math.isfinite(intercept)
            or isinstance(time, bool)
            or not isinstance(time, int)
            or time < 0
        ):
            raise ValueError("invalid stream state")
        rng_states = state["rng_states"]
        rngs = (
            self._target_rng,
            self._input_distribution_rng,
            self._gate_distribution_rng,
            self._input_drift_rng,
            self._gate_drift_rng,
            self._input_sampling_rng,
            self._gate_sampling_rng,
            self._intercept_rng,
            self._noise_rng,
        )
        if not isinstance(rng_states, list) or len(rng_states) != len(rngs):
            raise ValueError("invalid RNG state")
        self.time, self.input_state, self.gate_state = time, input_state, gate_state
        self.input_dag, self.gate_dag = input_dag, gate_dag
        self.intercept, self.weights, self.functions = intercept, weights, functions
        for rng, rng_state in zip(rngs, rng_states, strict=True):
            rng.bit_generator.state = rng_state


def generate_trajectory(config: SyntheticStreamConfig, n_samples: int, seed: int, return_metadata: bool = False) -> Trajectory:
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 0:
        raise ValueError("n_samples must be a non-negative integer")
    stream = SyntheticDriftingRegressionStream(config, seed)
    X = np.empty((n_samples, config.n_features), dtype=np.uint8)
    y = np.empty(n_samples, dtype=np.float64)
    metadata = [] if return_metadata else None
    for index in range(n_samples):
        if return_metadata:
            X[index], y[index], item_metadata = stream.next_sample(return_metadata=True)
            metadata.append(item_metadata)  # type: ignore[union-attr]
        else:
            X[index], y[index] = stream.next_sample()
    return Trajectory(X, y, config, seed, metadata)
