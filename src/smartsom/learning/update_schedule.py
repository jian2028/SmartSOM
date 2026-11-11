"""Per-network replay training clocks, independent of physical simulation time."""

from dataclasses import dataclass
from fractions import Fraction


@dataclass
class UpdateClock:
    inserted: int = 0
    pending: int = 0
    credit: Fraction = Fraction(0)

    def advance(self, count, settings, *, episode_end):
        previous = self.inserted
        self.inserted += count
        # The transition that reaches warmup is the first eligible insertion.
        eligible = max(
            0, self.inserted - max(previous, settings.warmup_transitions - 1)
        )
        self.pending += eligible
        if settings.update_amount == "ratio":
            self.credit += eligible * Fraction(str(settings.updates_per_transition))
        if self.inserted < settings.warmup_transitions:
            # Warmup supplies replay diversity, not a backlog of optimizer work.
            self.pending = 0
            self.credit = Fraction(0)
            return 0
        if settings.update_trigger == "episode":
            triggers = int(episode_end)
        else:
            triggers = self.pending // settings.update_every
        if not triggers:
            return 0
        if settings.update_trigger == "episode":
            self.pending = 0
        else:
            self.pending %= settings.update_every
        if settings.update_amount == "fixed":
            return triggers * settings.epochs
        updates = int(self.credit)
        self.credit -= updates
        return updates

    def epsilon(self, settings):
        if settings.exploration == "constant":
            return settings.epsilon
        fraction = min(1.0, self.inserted / settings.epsilon_decay_transitions)
        return settings.epsilon + fraction * (settings.epsilon_final - settings.epsilon)

    def state(self):
        return {
            "inserted": self.inserted,
            "pending": self.pending,
            "credit": (self.credit.numerator, self.credit.denominator),
        }

    @classmethod
    def restore(cls, state):
        return cls(state["inserted"], state["pending"], Fraction(*state["credit"]))
