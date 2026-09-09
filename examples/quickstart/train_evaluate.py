"""Train then independently evaluate the selected model in one experiment."""

import argparse

from smartsom.api import load_preset, train_evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="marl_micro")
    parser.add_argument("--steps", type=int, default=4096)
    args = parser.parse_args()

    config = load_preset(args.preset)
    config.training.total_steps = args.steps
    config.evaluation.replications = 5
    config.evaluation.checkpoint = "last"
    result = train_evaluate(config)
    print(f"Experiment: {result.training.run_dir}")
    if result.evaluation is not None:
        print(f"Evaluation: {result.evaluation.status}")


if __name__ == "__main__":
    main()
