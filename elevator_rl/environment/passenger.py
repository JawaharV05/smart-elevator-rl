"""Passenger data model and lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PassengerStatus(str, Enum):
    WAITING = "waiting"
    INSIDE = "inside"
    COMPLETED = "completed"


@dataclass
class Passenger:
    """One request made by a passenger in the simulated building."""

    passenger_id: int
    arrival_time: int
    source_floor: int
    destination_floor: int
    status: PassengerStatus = PassengerStatus.WAITING
    board_time: int | None = None
    completion_time: int | None = None

    def __post_init__(self) -> None:
        if self.source_floor == self.destination_floor:
            raise ValueError("source and destination floors must differ")

    def waiting_time(self, current_time: int) -> int:
        """Return time spent waiting so far (or until boarding)."""
        end_time = self.board_time if self.board_time is not None else current_time
        return max(0, end_time - self.arrival_time)

    def journey_time(self) -> int | None:
        """Return time inside the elevator once the trip has completed."""
        if self.board_time is None or self.completion_time is None:
            return None
        return self.completion_time - self.board_time

