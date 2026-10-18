"""The shared, non-RL elevator simulator used by all later policies."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

from elevator_rl.environment.config import SimulationConfig
from elevator_rl.environment.elevator import Elevator
from elevator_rl.environment.passenger import Passenger, PassengerStatus
from elevator_rl.simulation.passenger_generator import PassengerGenerator


class ElevatorSimulator:
    """Simulate requests and a single elevator in discrete time.

    ``move_to`` is deliberately a high-level operation: it represents a
    scheduler choosing the next floor.  Each travelled floor advances time and
    can create new passenger arrivals.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.generator = PassengerGenerator(config)
        self.elevator = Elevator(capacity=config.elevator_capacity)
        self.current_time = 0
        self.passengers: list[Passenger] = []
        self.waiting_by_floor: dict[int, list[Passenger]] = defaultdict(list)

    @property
    def is_finished(self) -> bool:
        return self.current_time >= self.config.duration

    @property
    def completed_passengers(self) -> list[Passenger]:
        return [p for p in self.passengers if p.status == PassengerStatus.COMPLETED]

    def pending_floors(self) -> list[int]:
        """Floors with pickups or drop-offs required, in ascending order."""
        floors = {floor for floor, people in self.waiting_by_floor.items() if people}
        floors.update(p.destination_floor for p in self.elevator.passengers)
        return sorted(floors)

    def tick(self) -> Passenger | None:
        """Advance one time unit and possibly create one new passenger."""
        if self.is_finished:
            return None
        self.current_time += 1
        generated = self.generator.generate(self.current_time)
        if generated is None:
            return None
        passenger_id, source, destination = generated
        passenger = Passenger(passenger_id, self.current_time, source, destination)
        self.passengers.append(passenger)
        self.waiting_by_floor[source].append(passenger)
        return passenger

    def service_current_floor(self) -> tuple[list[Passenger], list[Passenger]]:
        """Drop off first, then board FIFO waiting passengers at this floor."""
        exited = self.elevator.unload(self.current_time)
        queue = self.waiting_by_floor[self.elevator.current_floor]
        boarded = self.elevator.board(queue, self.current_time)
        del queue[: len(boarded)]
        if not queue:
            self.waiting_by_floor.pop(self.elevator.current_floor, None)
        return exited, boarded

    def move_to(
        self,
        target_floor: int,
        on_tick: Callable[[ElevatorSimulator], None] | None = None,
    ) -> tuple[list[Passenger], list[Passenger]]:
        """Travel to a floor, processing new arrivals during the trip, then serve.

        Returns passengers who exited and boarded at the selected floor.

        ``on_tick`` (optional) is invoked after every travelled floor and once
        more after the door service at the destination, with the simulator as
        its only argument.  It is used by the animated visualization to capture
        the building state at live resolution.
        """
        if not 0 <= target_floor < self.config.num_floors:
            raise ValueError(f"target floor must be between 0 and {self.config.num_floors - 1}")
        self.elevator.target_floor = target_floor
        while self.elevator.current_floor != target_floor and not self.is_finished:
            self.elevator.move_one_floor_toward(target_floor)
            self.tick()
            if on_tick is not None:
                on_tick(self)
        if self.elevator.current_floor != target_floor:
            return [], []
        self.elevator.direction = 0
        self.elevator.target_floor = None
        served = self.service_current_floor()
        if on_tick is not None:
            on_tick(self)
        return served

    def wait_one_tick(
        self,
        on_tick: Callable[[ElevatorSimulator], None] | None = None,
    ) -> Passenger | None:
        """Keep the elevator idle for one time unit."""
        self.elevator.direction = 0
        result = self.tick()
        if on_tick is not None:
            on_tick(self)
        return result

    def summary(self) -> dict[str, float | int]:
        """Return basic Phase 1 metrics without imposing a scheduling policy."""
        completed = self.completed_passengers
        waits = [p.waiting_time(self.current_time) for p in completed]
        journeys = [p.journey_time() for p in completed if p.journey_time() is not None]
        return {
            "time": self.current_time,
            "generated": len(self.passengers),
            "completed": len(completed),
            "waiting_or_inside": len(self.passengers) - len(completed),
            "travel_distance": self.elevator.travel_distance,
            "average_waiting_time": sum(waits) / len(waits) if waits else 0.0,
            "average_journey_time": sum(journeys) / len(journeys) if journeys else 0.0,
        }

