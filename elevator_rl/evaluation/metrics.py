"""Policy-independent performance measures for completed elevator trips."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from elevator_rl.simulation.simulator import ElevatorSimulator


@dataclass(frozen=True)
class SimulationMetrics:
    algorithm: str
    duration: int
    generated_passengers: int
    served_passengers: int
    unserved_passengers: int
    average_waiting_time: float
    maximum_waiting_time: float
    average_journey_time: float
    throughput_per_tick: float
    elevator_travel_distance: int

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


def calculate_metrics(simulator: ElevatorSimulator, algorithm: str) -> SimulationMetrics:
    """Calculate metrics from completed trips only, except overall counts.

    This avoids presenting an unfinished passenger's current wait as their
    final service outcome.  The unserved count exposes any remaining demand.
    """
    completed = simulator.completed_passengers
    waiting_times = [passenger.waiting_time(simulator.current_time) for passenger in completed]
    journey_times = [
        passenger.journey_time()
        for passenger in completed
        if passenger.journey_time() is not None
    ]
    served = len(completed)
    duration = simulator.current_time

    return SimulationMetrics(
        algorithm=algorithm,
        duration=duration,
        generated_passengers=len(simulator.passengers),
        served_passengers=served,
        unserved_passengers=len(simulator.passengers) - served,
        average_waiting_time=sum(waiting_times) / served if served else 0.0,
        maximum_waiting_time=max(waiting_times, default=0.0),
        average_journey_time=sum(journey_times) / len(journey_times) if journey_times else 0.0,
        throughput_per_tick=served / duration if duration else 0.0,
        elevator_travel_distance=simulator.elevator.travel_distance,
    )

