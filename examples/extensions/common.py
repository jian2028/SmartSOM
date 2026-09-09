"""Shared, explicitly registered feed-forward example for all learning backends."""

import argparse
import inspect
import json
from pathlib import Path

from torch import nn

from smartsom.api import EvaluationOptions, evaluate, load_preset, train_evaluate
from smartsom.config.extensions import (
    ActorCriticSpec,
    ExtensionRef,
    ExtensionSpec,
    NetworkBranch,
    NetworkSpec,
    RewardSpec,
)
from smartsom.experiments.audit import audit_run
from smartsom.learning.extensions import register_extension
from smartsom.learning.torch_extensions import FlattenEncoder

PRESETS = {
    "sb3.maskable_ppo": "sb3_micro",
    "rllib.ppo": "rllib_micro",
    "rllib.resource_ppo": "marl_micro",
}


class DemoEncoder(nn.Module):
    """A custom stateless module; parameters are ordinary checkpointed Torch state."""

    def __init__(self, parameters, space):
        super().__init__()
        if set(parameters) != {"width"}:
            raise ValueError("DemoEncoder requires exactly one width parameter")
        width = parameters["width"]
        if type(width) is not int or width < 1:
            raise ValueError("DemoEncoder width must be a positive integer")
        self.flatten = FlattenEncoder({}, space)
        self.output_size = width
        self.encoder = nn.Sequential(nn.Linear(space.flat_size, width), nn.Tanh())

    def forward(self, observations):
        return self.encoder(self.flatten(observations))


def register_demo_encoder():
    # Declaring the delegated implementation file makes its code part of identity.
    return register_extension(
        "torch_encoder",
        "example.public_ff",
        "1",
        DemoEncoder,
        supported_backends=tuple(PRESETS),
        source_files=(inspect.getsourcefile(FlattenEncoder),),
    )


def branch(width, head):
    return NetworkBranch(
        encoder=ExtensionRef(
            name="example.public_ff", version="1", parameters={"width": width}
        ),
        hidden_sizes=(head,),
        activation="tanh",
    )


def build_config(provider, output_root, *, steps=128, replications=1):
    """A short engineering demonstration, separate from the frozen research recipe."""
    if provider not in PRESETS or type(steps) is not int or steps < 32 or steps % 32:
        raise ValueError(
            "choose a supported provider and a positive multiple of 32 steps"
        )
    register_demo_encoder()
    config = load_preset(PRESETS[provider])
    config.training.total_steps = steps
    config.training.steps_per_update = 32
    config.algorithm.batch_size = 16
    config.algorithm.n_epochs = 2
    config.validation.every_updates = 2
    config.validation.replications = 1
    config.checkpointing.every_updates = 1
    config.evaluation.replications = replications
    config.evaluation.full_replay = True
    config.logging.progress = "off"
    config.logging.tensorboard = False
    config.output.root = str(Path(output_root).expanduser().resolve())
    config.output.name = f"extension-demo-{PRESETS[provider]}"
    resource = provider == "rllib.resource_ppo"
    config.algorithm.extensions = ExtensionSpec(
        observation=ExtensionRef(name="builtin.dict", version="1"),
        network=NetworkSpec(
            actor=branch(8, 8),
            critic=branch(12, 16),
            roles={
                "agv_policy": ActorCriticSpec(actor=branch(6, 8), critic=branch(10, 12))
            }
            if resource
            else {},
        ),
        reward=RewardSpec(
            team=ExtensionRef(
                name="builtin.reward_scale", version="1", parameters={"scale": 0.5}
            ),
            roles={
                "agv_policy": ExtensionRef(
                    name="builtin.reward_scale", version="1", parameters={"scale": 0.75}
                )
            }
            if resource
            else {},
        ),
    )
    return config


def main(provider):
    parser = argparse.ArgumentParser(
        description="Train or evaluate the registered SmartSOM extension example."
    )
    parser.add_argument("--output-root", default="runs/extension-examples")
    parser.add_argument(
        "--steps",
        type=int,
        default=128,
        help="demonstration environment steps, a positive multiple of 32",
    )
    parser.add_argument("--replications", type=int, default=1)
    parser.add_argument(
        "--checkpoint-source",
        help="evaluate an existing matching demo run/checkpoint instead of training",
    )
    args = parser.parse_args()
    register_demo_encoder()
    if args.checkpoint_source:
        evaluation = evaluate(
            args.checkpoint_source,
            EvaluationOptions(replications=args.replications),
            output_root=args.output_root,
        )
        training_directory = None
    else:
        config = build_config(
            provider, args.output_root, steps=args.steps, replications=args.replications
        )
        result = train_evaluate(config)
        evaluation = result.evaluation
        training_directory = str(result.training.run_dir)
    if evaluation is None:
        raise RuntimeError("training did not reach independent evaluation")
    audit = audit_run(evaluation.run_dir)
    print(
        json.dumps(
            {
                "training": training_directory,
                "evaluation": str(evaluation.run_dir),
                "status": evaluation.status,
                "completed": evaluation.completed,
                "failed": evaluation.failed,
                "engineering_failures": evaluation.engineering_failures,
                "audit": audit["status"],
            },
            indent=2,
        )
    )
    return 1 if evaluation.engineering_failures or audit["status"] == "failed" else 0
