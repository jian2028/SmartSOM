"""Train a named preset; edit typed properties here for a Python experiment."""

import argparse

from smartsom.api import load_preset, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="marl_micro")
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    config = load_preset(args.preset)
    config.training.total_steps = args.steps
    config.training.steps_per_update = 256
    config.algorithm.learning_rate = 3e-4
    config.seed = args.seed
    config.runtime.num_envs = args.num_envs
    config.runtime.device = args.device
    result = train(config)
    print(f"Run: {result.run_dir}\nLast checkpoint: {result.last_checkpoint}")


if __name__ == "__main__":
    main()
