"""Evaluate a run, checkpoint or exported model using its recorded structure."""

import argparse

from smartsom.api import EvaluationOptions, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--checkpoint", choices=("last", "best"), default="last")
    parser.add_argument("--seed", type=int, default=202)
    parser.add_argument("--replications", type=int, default=5)
    parser.add_argument("--baseline", action="append", default=[])
    args = parser.parse_args()

    options = EvaluationOptions(
        seed=args.seed,
        replications=args.replications,
        checkpoint=args.checkpoint,
        baselines=tuple(args.baseline),
    )
    result = evaluate(args.source, options)
    print(f"Evaluation: {result.run_dir}\nStatus: {result.status}")


if __name__ == "__main__":
    main()
