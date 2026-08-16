"""Shared plotting helpers for streaming benchmark artifacts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

DRIFT_TYPES = (
    ("input_distribution_sampling_indicator", "input distribution", "#2E7D32"),
    ("gate_distribution_sampling_indicator", "gate distribution", "#7B4F9E"),
    ("intercept_sampling_indicator", "intercept", "#9A4D21"),
)
LEGEND_OPTIONS = {"loc": "upper right", "framealpha": 0.95, "fontsize": 8, "ncol": 2}


def rolling_mean(values: np.ndarray, window: int = 100) -> np.ndarray:
    if len(values) == 0:
        return values
    window = min(window, len(values))
    valid = np.isfinite(values)
    totals = np.concatenate(([0.0], np.cumsum(np.where(valid, values, 0.0))))
    counts = np.concatenate(([0], np.cumsum(valid)))
    starts = np.maximum(np.arange(len(values)) + 1 - window, 0)
    return (totals[1:] - totals[starts]) / (counts[1:] - counts[starts])


def drift_handles() -> list[Line2D]:
    return [Line2D([], [], color=color, linestyle="dashed", label=label) for _, label, color in DRIFT_TYPES]


def draw_warmup_end(axis: plt.Axes, warmup_samples: int) -> None:
    if warmup_samples:
        axis.axvline(warmup_samples, color="#555555", linewidth=1.2, label="warmup end")


def draw_drift_events(axis: plt.Axes, trajectory) -> None:
    if trajectory.metadata is None:
        return
    for field, _, color in DRIFT_TYPES:
        event_times = [time for time, metadata in enumerate(trajectory.metadata) if metadata[field]]
        if event_times:
            axis.vlines(
                event_times,
                0,
                1,
                transform=axis.get_xaxis_transform(),
                color=color,
                linewidth=1,
                alpha=0.75,
                linestyles="dashed",
            )
