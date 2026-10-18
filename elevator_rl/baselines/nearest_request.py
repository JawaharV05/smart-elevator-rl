"""Nearest-request elevator scheduling."""

from __future__ import annotations

from elevator_rl.simulation.simulator import ElevatorSimulator


class NearestRequestScheduler:
    """Select the pending pickup or drop-off floor nearest to the elevator."""

    name = "Nearest-Request"

    def select_target(self, simulator: ElevatorSimulator) -> int | None:
        # A pickup cannot currently be served by a full elevator. Excluding it
        # prevents repeatedly selecting the current pickup floor with no way to
        # make progress; drop-offs remain actionable at all times.
        pending_floors = {passenger.destination_floor for passenger in simulator.elevator.passengers}
        if simulator.elevator.remaining_capacity > 0:
            pending_floors.update(
                floor for floor, queue in simulator.waiting_by_floor.items() if queue
            )
        if not pending_floors:
            return None
        return min(pending_floors, key=lambda floor: (abs(floor - simulator.elevator.current_floor), floor))
