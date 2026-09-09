"""A final tensor transform before RLlib's PPO GAE/value-target connector."""

from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core.columns import Columns


class ScaleLearnerRewards(ConnectorV2):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def __call__(self, *, batch, **kwargs):
        return {
            module: {**values, Columns.REWARDS: values[Columns.REWARDS] * self.scale}
            for module, values in batch.items()
        }
