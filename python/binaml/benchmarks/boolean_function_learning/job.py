"""One isolated learner/seed boolean-function-learning benchmark job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from binaml.benchmarks._common import write_json_atomically
from binaml.benchmarks.scenario import load_scenario

from .evaluate import run_job


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--factory", required=True)
    parser.add_argument("--parameters-json", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scenario = load_scenario(args.scenario)
    parameters = json.loads(args.parameters_json)
    if not isinstance(parameters, dict):
        raise TypeError("learner parameters must be a JSON object")

    write_json_atomically(
        args.output,
        run_job(scenario, args.factory, parameters, args.seed),
    )


if __name__ == "__main__":
    main()
