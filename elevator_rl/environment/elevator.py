"""State and operations of the single simulated elevator."""

from __future__ import annotations

from dataclasses import dataclass, field

from .passenger import Passenger, PassengerStatus


@dataclass
class Elevator:
    capacity: int
    current_floor: int = 0
    direction: int = 0  # -1 down, 0 idle, 1 up
    target_floor: int | None = None
    passengers: list[Passenger] = field(default_factory=list)
    travel_distance: int = 0

    @property
    def remaining_capacity(self) -> int:
        return self.capacity - len(self.passengers)

    def move_one_floor_toward(self, target_floor: int) -> None:
        """Move exactly one floor toward a valid target."""
        if target_floor == self.current_floor:
            self.direction = 0
            return
        self.direction = 1 if target_floor > self.current_floor else -1
        self.current_floor += self.direction
        self.travel_distance += 1

    def unload(self, current_time: int) -> list[Passenger]:
        """Let all passengers whose destination is the current floor exit."""
        exiting = [p for p in self.passengers if p.destination_floor == self.current_floor]
        for passenger in exiting:
            passenger.status = PassengerStatus.COMPLETED
            passenger.completion_time = current_time
        self.passengers = [p for p in self.passengers if p not in exiting]
        return exiting

    def board(self, waiting_passengers: list[Passenger], current_time: int) -> list[Passenger]:
        """Board waiting passengers in arrival order until capacity is reached."""
        boarded = waiting_passengers[: self.remaining_capacity]
        for passenger in boarded:
            passenger.status = PassengerStatus.INSIDE
            passenger.board_time = current_time
        self.passengers.extend(boarded)
        return boarded

