"""Seaborn visualizations saved with streaming-classification run artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

from binaml.benchmarks.plots_common import (
    LEGEND_OPTIONS,
    draw_drift_events,
    draw_warmup_end,
    drift_handles,
    rolling_mean,
)
from binaml.environments import ClassificationTrajectory
from binaml.evaluation import PrequentialClassificationResult


def write_accuracy_plot(
    path: Path,
    trajectory: ClassificationTrajectory,
    evaluations: dict[str, PrequentialClassificationResult],
    warmup_samples: int = 0,
) -> None:
    sns.set_theme(context="notebook", style="whitegrid", rc={"grid.alpha": 0.3})
    time = np.arange(len(trajectory.y))
    figure, axis = plt.subplots(figsize=(16, 5), layout="constrained")
    for (model_name, evaluation), color in zip(
        evaluations.items(), sns.color_palette("colorblind", n_colors=len(evaluations)), strict=True
    ):
        sns.lineplot(
            x=time,
            y=rolling_mean(evaluation.correct.astype(np.float64)),
            ax=axis,
            color=color,
            label=model_name,
        )
    draw_drift_events(axis, trajectory)
    draw_warmup_end(axis, warmup_samples)
    axis.set(title=f"Rolling accuracy (seed {trajectory.seed})", xlabel="sample", ylabel="last 100 samples")
    axis.set_ylim(0.0, 1.0)
    axis.legend(handles=[*axis.lines, *drift_handles()], **LEGEND_OPTIONS)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def write_model_plot(
    path: Path,
    trajectory: ClassificationTrajectory,
    model_name: str,
    evaluation: PrequentialClassificationResult,
    color: tuple[float, float, float],
    warmup_samples: int = 0,
) -> None:
    sns.set_theme(context="notebook", style="whitegrid", rc={"grid.alpha": 0.3})
    time = np.arange(len(trajectory.y))

    figure, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True, layout="constrained")

    sns.lineplot(x=time, y=trajectory.y, ax=axes[0], color="#1F5A8A", label="label", alpha=0.8)
    sns.lineplot(x=time, y=evaluation.predictions, ax=axes[0], color=color, label="prediction", alpha=0.8)
    draw_drift_events(axes[0], trajectory)
    draw_warmup_end(axes[0], warmup_samples)
    axes[0].set(title=f"Trajectory — {model_name} (seed {trajectory.seed})", ylabel="class")
    axes[0].legend(handles=[*axes[0].lines, *drift_handles()], **LEGEND_OPTIONS)

    sns.lineplot(
        x=time,
        y=evaluation.correct.astype(np.float64),
        ax=axes[1],
        color=color,
        label="correct",
        alpha=0.8,
        drawstyle="steps-post",
    )
    draw_drift_events(axes[1], trajectory)
    draw_warmup_end(axes[1], warmup_samples)
    axes[1].set(title=f"Correctness — {model_name}", xlabel="sample", ylabel="1 if correct")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(handles=[*axes[1].lines, *drift_handles()], **LEGEND_OPTIONS)
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def write_aggregate_scatter(path: Path, metrics: dict[str, dict[str, object]]) -> None:
    names = list(metrics["accuracy"])
    times = [metrics["timing_seconds"][name]["total"]["average"] for name in names]  # type: ignore[index]
    accuracies = [metrics["accuracy"][name]["average"] for name in names]  # type: ignore[index]

    sns.set_theme(context="notebook", style="whitegrid", rc={"grid.alpha": 0.3})
    figure, axis = plt.subplots(figsize=(10, 6), layout="constrained")
    colors = sns.color_palette("colorblind", n_colors=len(names))
    axis.scatter(times, accuracies, s=400, c=colors, alpha=0.8, edgecolors="#222222", linewidths=0.8)
    for name, time, accuracy in zip(names, times, accuracies, strict=True):
        axis.annotate(name, (time, accuracy), xytext=(6, 6), textcoords="offset points")
    legend_handles = [
        Line2D(
            [],
            [],
            color=color,
            marker="o",
            linestyle="",
            markersize=8,
            label=f"{name}: accuracy {accuracy:.3f}, time {time:.3f} s",
        )
        for name, time, accuracy, color in zip(names, times, accuracies, colors, strict=True)
    ]
    axis.set(
        title="Model comparison",
        xlabel="Total step time (seconds)",
        ylabel="Accuracy",
    )
    axis.set_ylim(0.0, 1.0)
    axis.margins(x=0.08, y=0.1)
    axis.legend(handles=legend_handles, title="Metrics", loc="center left", bbox_to_anchor=(1.02, 0.5))
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
