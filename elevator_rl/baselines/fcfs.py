"""First-Come-First-Serve elevator scheduling."""

from __future__ import annotations

from elevator_rl.environment.passenger import PassengerStatus
from elevator_rl.simulation.simulator import ElevatorSimulator


class FCFSScheduler:
    """Select the floor associated with the oldest unfinished passenger.

    A waiting passenger requests their source floor; an onboard passenger
    requests their destination floor.  Ties are resolved by lower floor number
    to keep the policy deterministic.
    """

    name = "FCFS"

    def select_target(self, simulator: ElevatorSimulator) -> int | None:
        outstanding = [
            passenger
            for passenger in simulator.passengers
            if passenger.status != PassengerStatus.COMPLETED
        ]
        if not outstanding:
            return None

        for passenger in sorted(
            outstanding,
            key=lambda passenger: (passenger.arrival_time, passenger.passenger_id),
        ):
            if passenger.status == PassengerStatus.INSIDE:
                return passenger.destination_floor
            if simulator.elevator.remaining_capacity > 0:
                return passenger.source_floor
        return None
