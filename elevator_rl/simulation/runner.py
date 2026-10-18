"""Execute any high-level scheduling policy on the shared simulator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from elevator_rl.environment.config import SimulationConfig
from elevator_rl.simulation.simulator import ElevatorSimulator


class Scheduler(Protocol):
    """Minimal interface shared by baseline and future RL schedulers."""

    name: str

    def select_target(self, simulator: ElevatorSimulator) -> int | None:
        """Return a requested target floor, or ``None`` to idle."""


def run_policy(
    config: SimulationConfig,
    scheduler: Scheduler,
    on_tick: Callable[[ElevatorSimulator], None] | None = None,
) -> ElevatorSimulator:
    """Run a policy until the configured duration and return its simulator.

    Every policy receives a fresh simulator configured with the supplied seed,
    making experiments comparable under the same passenger arrival scenario.

    ``on_tick`` (optional) is forwarded to every simulator movement so
    callers such as the animation can observe the building floor by floor.
    """
    simulator = ElevatorSimulator(config)
    while not simulator.is_finished:
        target = scheduler.select_target(simulator)
        if target is None:
            simulator.wait_one_tick(on_tick=on_tick)
        else:
            simulator.move_to(target, on_tick=on_tick)
    return simulator

