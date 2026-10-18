from elevator_rl.environment.elevator import Elevator
from elevator_rl.environment.passenger import Passenger, PassengerStatus


def test_elevator_movement_updates_floor_direction_and_distance() -> None:
    elevator = Elevator(capacity=2, current_floor=4)

    elevator.move_one_floor_toward(7)
    elevator.move_one_floor_toward(2)

    assert elevator.current_floor == 4
    assert elevator.direction == -1
    assert elevator.travel_distance == 2


def test_boarding_respects_capacity_and_preserves_queue_order() -> None:
    elevator = Elevator(capacity=2)
    queue = [Passenger(index, 0, 0, 5) for index in range(3)]

    boarded = elevator.board(queue, current_time=4)

    assert [passenger.passenger_id for passenger in boarded] == [0, 1]
    assert elevator.remaining_capacity == 0
    assert all(passenger.status == PassengerStatus.INSIDE for passenger in boarded)
    assert all(passenger.board_time == 4 for passenger in boarded)
    assert queue[2].status == PassengerStatus.WAITING


def test_unload_completes_only_matching_destinations() -> None:
    elevator = Elevator(capacity=3, current_floor=6)
    arriving = Passenger(1, 0, 0, 6, PassengerStatus.INSIDE, board_time=2)
    continuing = Passenger(2, 0, 0, 7, PassengerStatus.INSIDE, board_time=2)
    elevator.passengers = [arriving, continuing]

    exited = elevator.unload(current_time=8)

    assert exited == [arriving]
    assert arriving.status == PassengerStatus.COMPLETED
    assert arriving.completion_time == 8
    assert elevator.passengers == [continuing]

