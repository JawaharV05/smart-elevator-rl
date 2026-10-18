"""Seeded stochastic passenger-arrival generation."""

from __future__ import annotations

import random

from elevator_rl.environment.config import SimulationConfig


class PassengerGenerator:
    """Generates at most one passenger per time tick using a Bernoulli process.

    The simple probability model is easy to reason about and keeps traffic
    demand configurable.  Later phases can replace it with richer patterns.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self._rng = random.Random(config.seed)
        self._next_id = 0

    def generate(self, current_time: int) -> tuple[int, int, int] | None:
        """Return ``(id, source, destination)`` or ``None`` for no arrival."""
        if self._rng.random() >= self.config.arrival_probability:
            return None
        source = self._rng.randrange(self.config.num_floors)
        destination = self._rng.randrange(self.config.num_floors - 1)
        if destination >= source:
            destination += 1
        passenger_id = self._next_id
        self._next_id += 1
        return passenger_id, source, destination

