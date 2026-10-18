"""Configuration objects shared by the simulation and later experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters for one single-elevator simulation run.

    Floors are numbered from 0 through ``num_floors - 1``.  One simulation
    tick is also the time required to travel one floor.
    """

    num_floors: int = 10
    elevator_capacity: int = 6
    duration: int = 200
    arrival_probability: float = 0.12
    seed: int = 42
    completion_reward: float = 2.0
    waiting_penalty: float = 0.05
    movement_penalty: float = 0.02
    long_wait_penalty: float = 0.10
    long_wait_threshold: int = 15

    def __post_init__(self) -> None:
        if self.num_floors < 2:
            raise ValueError("num_floors must be at least 2")
        if self.elevator_capacity < 1:
            raise ValueError("elevator_capacity must be positive")
        if self.duration < 1:
            raise ValueError("duration must be positive")
        if not 0 <= self.arrival_probability <= 1:
            raise ValueError("arrival_probability must be between 0 and 1")
        if self.long_wait_threshold < 0:
            raise ValueError("long_wait_threshold cannot be negative")
