"""Seaborn visualizations saved with streaming-regression run artifacts."""

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
from binaml.environments import Trajectory
from binaml.evaluation import PrequentialResult


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
        sns.lineplot(x=time, y=np.sqrt(rolling_mean(evaluation.squared_errors)), ax=axis, color=color, label=model_name)
    draw_drift_events(axis, trajectory)
    draw_warmup_end(axis, warmup_samples)
    axis.set(title=f"Rolling RMSE (seed {trajectory.seed})", xlabel="sample", ylabel="last 100 samples")
    axis.legend(handles=[*axis.lines, *drift_handles()], **LEGEND_OPTIONS)
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
    draw_drift_events(axes[0], trajectory)
    draw_warmup_end(axes[0], warmup_samples)
    axes[0].set(title=f"Trajectory — {model_name} (seed {trajectory.seed})", ylabel="target")
    axes[0].legend(handles=[*axes[0].lines, *drift_handles()], **LEGEND_OPTIONS)

    sns.lineplot(x=time, y=evaluation.predictions - trajectory.y, ax=axes[1], color=color, label="residual", alpha=0.8)
    axes[1].axhline(0, color="#444444", linewidth=0.8, label="zero")
    draw_drift_events(axes[1], trajectory)
    draw_warmup_end(axes[1], warmup_samples)
    axes[1].set(title=f"Residual — {model_name}", xlabel="sample", ylabel="prediction − target")
    axes[1].legend(handles=[*axes[1].lines, *drift_handles()], **LEGEND_OPTIONS)
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


def write_job_plots(
    output_dir: Path,
    source_argument: str,
    source_path: Path,
    completed: list[dict[str, object]],
    metrics: dict[str, dict[str, object]],
) -> None:
    from binaml.benchmarks.scenario import load_scenario, warmup_samples
    from binaml.environments import SyntheticStreamConfig, generate_trajectory
    from binaml.evaluation import EvaluationTiming, PrequentialResult

    records_by_seed: dict[int, dict[str, dict[str, object]]] = {}
    for job in completed:
        records_by_seed.setdefault(int(job["seed"]), {})[str(job["model"])] = job["result"]  # type: ignore[index]
    plots_dir = output_dir / "plots"
    plots_dir.mkdir()
    scenario = load_scenario(source_path) if source_argument == "--scenario" else None
    warmup = warmup_samples(scenario) if scenario is not None else 0
    for seed, records in records_by_seed.items():
        trajectory = (
            generate_trajectory(
                SyntheticStreamConfig.from_dict(scenario["environment"]),
                int(scenario["n_samples"]),
                seed,
                return_metadata=True,
            )
            if scenario is not None
            else Trajectory.load_npz(source_path)
        )
        evaluations = {
            name: PrequentialResult(
                np.asarray(record["predictions"]),
                trajectory.y,
                np.asarray(record["squared_errors"]),
                EvaluationTiming(**record["timing_seconds"]),  # type: ignore[arg-type]
            )
            for name, record in records.items()
        }
        write_rmse_plot(plots_dir / f"rmse_seed_{seed}.png", trajectory, evaluations, warmup)
        for (model_name, evaluation), color in zip(
            evaluations.items(), sns.color_palette("colorblind", n_colors=len(evaluations)), strict=True
        ):
            model_dir = plots_dir / model_name.replace("/", "_")
            model_dir.mkdir(exist_ok=True)
            write_model_plot(model_dir / f"seed_{seed}.png", trajectory, model_name, evaluation, color, warmup)
    write_aggregate_scatter(plots_dir / "model_comparison.png", metrics)
