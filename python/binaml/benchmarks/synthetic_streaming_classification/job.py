"""One isolated model/seed streaming-classification benchmark job."""

from __future__ import annotations

from binaml.benchmarks.streaming import run_streaming_job_cli

from .benchmark import BENCHMARK


def main() -> None:
    run_streaming_job_cli(BENCHMARK)


if __name__ == "__main__":
    main()
