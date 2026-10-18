import pytest

from elevator_rl.environment.passenger import PassengerStatus

from .conftest import add_waiting_passenger, make_simulator


def test_service_boards_waiting_passengers_at_current_floor() -> None:
    simulator = make_simulator()
    passenger = add_waiting_passenger(simulator, 1, source=0, destination=4)

    exited, boarded = simulator.move_to(0)

    assert exited == []
    assert boarded == [passenger]
    assert passenger.status == PassengerStatus.INSIDE
    assert simulator.waiting_by_floor.get(0) is None
    assert simulator.pending_floors() == [4]


def test_move_to_delivers_passenger_and_tracks_wait_and_journey_time() -> None:
    simulator = make_simulator()
    passenger = add_waiting_passenger(simulator, 1, source=0, destination=3)
    simulator.move_to(0)

    exited, boarded = simulator.move_to(3)

    assert exited == [passenger]
    assert boarded == []
    assert passenger.status == PassengerStatus.COMPLETED
    assert passenger.waiting_time(simulator.current_time) == 0
    assert passenger.journey_time() == 3
    assert simulator.elevator.travel_distance == 3
    assert simulator.summary()["completed"] == 1


def test_simulator_leaves_excess_passengers_waiting_when_full() -> None:
    simulator = make_simulator(elevator_capacity=2)
    passengers = [add_waiting_passenger(simulator, index, 0, 5) for index in range(3)]

    _, boarded = simulator.move_to(0)

    assert boarded == passengers[:2]
    assert simulator.waiting_by_floor[0] == [passengers[2]]
    assert passengers[2].status == PassengerStatus.WAITING


def test_drop_off_happens_before_boarding_at_same_floor() -> None:
    simulator = make_simulator(elevator_capacity=1)
    inside = add_waiting_passenger(simulator, 1, 0, 2)
    simulator.move_to(0)
    waiting = add_waiting_passenger(simulator, 2, 2, 8)

    exited, boarded = simulator.move_to(2)

    assert exited == [inside]
    assert boarded == [waiting]
    assert inside.status == PassengerStatus.COMPLETED
    assert waiting.status == PassengerStatus.INSIDE


def test_pending_floors_includes_pickups_and_dropoffs_once() -> None:
    simulator = make_simulator()
    passenger = add_waiting_passenger(simulator, 1, 3, 7)
    simulator.elevator.board([passenger], current_time=0)
    simulator.waiting_by_floor.pop(3)
    add_waiting_passenger(simulator, 2, 4, 1)

    assert simulator.pending_floors() == [4, 7]


def test_episode_termination_prevents_more_time_advancement() -> None:
    simulator = make_simulator(duration=2)

    simulator.wait_one_tick()
    simulator.wait_one_tick()
    generated_after_end = simulator.wait_one_tick()

    assert simulator.is_finished
    assert simulator.current_time == 2
    assert generated_after_end is None


def test_invalid_target_floor_is_rejected() -> None:
    simulator = make_simulator()

    with pytest.raises(ValueError, match="target floor"):
        simulator.move_to(10)
