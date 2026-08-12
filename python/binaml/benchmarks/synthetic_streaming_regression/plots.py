"""Seaborn visualizations saved with streaming-regression run artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

from binaml.environments import Trajectory
from binaml.evaluation import PrequentialResult


def _rolling_mean(values: np.ndarray, window: int = 100) -> np.ndarray:
    if len(values) == 0:
        return values
    window = min(window, len(values))
    valid = np.isfinite(values)
    totals = np.concatenate(([0.0], np.cumsum(np.where(valid, values, 0.0))))
    counts = np.concatenate(([0], np.cumsum(valid)))
    starts = np.maximum(np.arange(len(values)) + 1 - window, 0)
    return (totals[1:] - totals[starts]) / (counts[1:] - counts[starts])


DRIFT_TYPES = (
    ("input_distribution_sampling_indicator", "input distribution", "#2E7D32"),
    ("gate_distribution_sampling_indicator", "gate distribution", "#7B4F9E"),
    ("intercept_sampling_indicator", "intercept", "#9A4D21"),
)
LEGEND_OPTIONS = {"loc": "upper right", "framealpha": 0.95, "fontsize": 8, "ncol": 2}


def _drift_handles() -> list[Line2D]:
    return [Line2D([], [], color=color, linestyle="dashed", label=label) for _, label, color in DRIFT_TYPES]


def _draw_warmup_end(axis: plt.Axes, warmup_samples: int) -> None:
    if warmup_samples:
        axis.axvline(warmup_samples, color="#555555", linewidth=1.2, label="warmup end")


def _draw_drift_events(axis: plt.Axes, trajectory: Trajectory) -> None:
    if trajectory.metadata is None:
        return
    for field, _, color in DRIFT_TYPES:
        event_times = [
            time
            for time, metadata in enumerate(trajectory.metadata)
            if metadata[field]
        ]
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


def write_rmse_plot(
    path: Path,
    trajectory: Trajectory,
    evaluations: dict[str, PrequentialResult],
    warmup_samples: int = 0,
) -> None:
    """Plot rolling RMSE for every model on one chart."""
    sns.set_theme(context="notebook", style="whitegrid", rc={"grid.alpha": 0.3})
    time = np.arange(len(trajectory.y))
    figure, axis = plt.subplots(figsize=(16, 5), layout="constrained")
    for (model_name, evaluation), color in zip(
        evaluations.items(), sns.color_palette("colorblind", n_colors=len(evaluations)), strict=True
    ):
        sns.lineplot(x=time, y=np.sqrt(_rolling_mean(evaluation.squared_errors)), ax=axis, color=color, label=model_name)
    _draw_drift_events(axis, trajectory)
    _draw_warmup_end(axis, warmup_samples)
    axis.set(title=f"Rolling RMSE (seed {trajectory.seed})", xlabel="sample", ylabel="last 100 samples")
    axis.legend(handles=[*axis.lines, *_drift_handles()], **LEGEND_OPTIONS)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def write_model_plot(
    path: Path,
    trajectory: Trajectory,
    model_name: str,
    evaluation: PrequentialResult,
    color: tuple[float, float, float],
    warmup_samples: int = 0,
) -> None:
    """Plot a model's trajectory and residual in one file."""
    sns.set_theme(context="notebook", style="whitegrid", rc={"grid.alpha": 0.3})
    time = np.arange(len(trajectory.y))

    figure, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True, layout="constrained")

    sns.lineplot(x=time, y=trajectory.y, ax=axes[0], color="#1F5A8A", label="observed target", alpha=0.8)
    sns.lineplot(x=time, y=evaluation.predictions, ax=axes[0], color=color, label="prediction", alpha=0.8)
    _draw_drift_events(axes[0], trajectory)
    _draw_warmup_end(axes[0], warmup_samples)
    axes[0].set(title=f"Trajectory — {model_name} (seed {trajectory.seed})", ylabel="target")
    axes[0].legend(handles=[*axes[0].lines, *_drift_handles()], **LEGEND_OPTIONS)

    sns.lineplot(x=time, y=evaluation.predictions - trajectory.y, ax=axes[1], color=color, label="residual", alpha=0.8)
    axes[1].axhline(0, color="#444444", linewidth=0.8, label="zero")
    _draw_drift_events(axes[1], trajectory)
    _draw_warmup_end(axes[1], warmup_samples)
    axes[1].set(title=f"Residual — {model_name}", xlabel="sample", ylabel="prediction − target")
    axes[1].legend(handles=[*axes[1].lines, *_drift_handles()], **LEGEND_OPTIONS)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def write_aggregate_scatter(path: Path, metrics: dict[str, dict[str, object]]) -> None:
    """Compare model quality and speed."""
    names = list(metrics["rmse"])
    times = [metrics["timing_seconds"][name]["total"]["average"] for name in names]  # type: ignore[index]
    rmses = [metrics["rmse"][name]["average"] for name in names]  # type: ignore[index]

    sns.set_theme(context="notebook", style="whitegrid", rc={"grid.alpha": 0.3})
    figure, axis = plt.subplots(figsize=(10, 6), layout="constrained")
    colors = sns.color_palette("colorblind", n_colors=len(names))
    axis.scatter(times, rmses, s=400, c=colors, alpha=0.8, edgecolors="#222222", linewidths=0.8)
    for name, time, rmse in zip(names, times, rmses, strict=True):
        axis.annotate(name, (time, rmse), xytext=(6, 6), textcoords="offset points")
    legend_handles = [
        Line2D(
            [],
            [],
            color=color,
            marker="o",
            linestyle="",
            markersize=8,
            label=f"{name}: RMSE {rmse:.3f}, time {time:.3f} s",
        )
        for name, time, rmse, color in zip(names, times, rmses, colors, strict=True)
    ]
    axis.set(
        title="Model comparison",
        xlabel="Total step time (seconds)",
        ylabel="RMSE",
    )
    axis.margins(x=0.08, y=0.1)
    axis.legend(handles=legend_handles, title="Metrics", loc="center left", bbox_to_anchor=(1.02, 0.5))
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


