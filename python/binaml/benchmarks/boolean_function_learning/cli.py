"""Boolean function learning benchmark orchestration helpers."""

from __future__ import annotations

from binaml.benchmarks.batch import run_batch_benchmark_cli

RUN_PREFIX = "boolean_function_learning"
JOB_MODULE = "binaml.benchmarks.boolean_function_learning.job"
TRACKED_ENV_KEYS = (
    "p_min",
    "p_max",
    "p_activation_min",
    "p_activation_max",
    "n_features",
    "min_truth_table_function_arity",
    "max_truth_table_function_arity",
)


def variant_label(variant: dict[str, object], variant_index: int) -> str:
    environment = variant.get("environment", {})
    if isinstance(environment, dict):
        parts = [f"{key}_{environment[key]}" for key in TRACKED_ENV_KEYS if key in environment]
        if parts:
            return "_".join(parts)
    p_noise = variant.get("p_noise")
    if p_noise is not None:
        return f"variant_{variant_index}_p_noise_{p_noise}"
    return f"variant_{variant_index}"


def summarize_variant(records: list[dict[str, object]]) -> dict[str, object]:
    from .evaluate import summarize_records

    by_learner: dict[str, list[dict[str, object]]] = {}
    for record in records:
        by_learner.setdefault(str(record["learner"]), []).append(record)
    return {learner: summarize_records(learner_records) for learner, learner_records in by_learner.items()}


def extract_variant_metrics(summaries: dict[str, object]) -> dict[str, object]:
    from .evaluate import extract_metrics

    if not isinstance(summaries, dict):
        raise TypeError("variant summaries must be an object")
    typed = {str(key): value for key, value in summaries.items() if isinstance(value, dict)}
    return extract_metrics(typed)


def main() -> None:
    run_batch_benchmark_cli(
        run_prefix=RUN_PREFIX,
        job_module=JOB_MODULE,
        entries_key="learners",
        default_entries=[],
        config_schema_version=1,
        summary_schema_version=1,
        metrics_schema_version=1,
        summarize_variant=summarize_variant,
        extract_metrics=extract_variant_metrics,
        variant_label=variant_label,
    )


if __name__ == "__main__":
    main()
