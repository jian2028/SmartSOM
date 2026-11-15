"""Scientific world generation called by the official environment reset path."""

from dataclasses import dataclass

from smartsom.config.base_diagnostic import (
    BaseExperiment,
    BaseWorkload,
    generate_scenario,
    seed_for,
)
from smartsom.domain.factory_design import FactoryDesign


@dataclass(frozen=True)
class TrainingWorlds:
    experiment: BaseExperiment
    factory: FactoryDesign
    workload: BaseWorkload
    train_seed: int
    drain_ticks: int

    def sample(self, rng, episode_index):
        """Use the EnvRunner-seeded Gym RNG; no application sampling workers.

        Balanced cycling is per environment. Official checkpoint recovery does
        not promise preservation of this environment's RNG or partial episode.
        """
        ex = self.experiment
        index = (
            int(rng.integers(len(ex.training_conditions)))
            if ex.condition_sampling == "random"
            else episode_index % len(ex.training_conditions)
        )
        condition = ex.training_conditions[index]
        seed = seed_for(
            self.train_seed,
            ex.training_world_domain,
            *ex.training_world_prefix,
            int(rng.integers(0, 2**63)),
        )
        if ex.training_seed_modulus:
            seed %= ex.training_seed_modulus
        scenario = generate_scenario(
            *ex.inputs_for(condition, self.factory, self.workload),
            seed,
            condition,
            self.drain_ticks,
        )
        return scenario, {"condition": condition, "world_seed": seed}
