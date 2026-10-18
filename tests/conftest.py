from __future__ import annotations

from elevator_rl.environment.config import SimulationConfig
from elevator_rl.environment.passenger import Passenger
from elevator_rl.simulation.simulator import ElevatorSimulator


def make_simulator(**overrides: int | float) -> ElevatorSimulator:
    """Create a quiet, short simulator suitable for deterministic tests."""
    values: dict[str, int | float] = {
        "num_floors": 10,
        "elevator_capacity": 2,
        "duration": 30,
        "arrival_probability": 0.0,
        "seed": 123,
    }
    values.update(overrides)
    return ElevatorSimulator(SimulationConfig(**values))  # type: ignore[arg-type]


def add_waiting_passenger(
    simulator: ElevatorSimulator,
    passenger_id: int,
    source: int,
    destination: int,
    arrival_time: int = 0,
) -> Passenger:
    """Add a controlled request without using the random arrival process."""
    passenger = Passenger(passenger_id, arrival_time, source, destination)
    simulator.passengers.append(passenger)
    simulator.waiting_by_floor[source].append(passenger)
    return passenger

