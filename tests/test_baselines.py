from elevator_rl.baselines.fcfs import FCFSScheduler
from elevator_rl.baselines.nearest_request import NearestRequestScheduler
from elevator_rl.environment.passenger import PassengerStatus
from elevator_rl.evaluation.metrics import calculate_metrics
from elevator_rl.simulation.runner import run_policy

from .conftest import add_waiting_passenger, make_simulator


def test_fcfs_selects_oldest_waiting_passengers_source_floor() -> None:
    simulator = make_simulator()
    add_waiting_passenger(simulator, 2, source=7, destination=1, arrival_time=5)
    add_waiting_passenger(simulator, 1, source=3, destination=8, arrival_time=2)

    assert FCFSScheduler().select_target(simulator) == 3


def test_fcfs_selects_oldest_onboard_passengers_destination() -> None:
    simulator = make_simulator()
    onboard = add_waiting_passenger(simulator, 1, source=0, destination=6, arrival_time=1)
    simulator.elevator.board([onboard], current_time=2)
    simulator.waiting_by_floor.pop(0)
    add_waiting_passenger(simulator, 2, source=4, destination=8, arrival_time=3)

    assert FCFSScheduler().select_target(simulator) == 6


def test_full_elevator_fcfs_prioritizes_dropoff_over_unboardable_pickup() -> None:
    simulator = make_simulator(elevator_capacity=1)
    onboard = add_waiting_passenger(simulator, 1, source=0, destination=8, arrival_time=5)
    simulator.elevator.board([onboard], current_time=5)
    simulator.waiting_by_floor.pop(0)
    add_waiting_passenger(simulator, 2, source=1, destination=3, arrival_time=1)

    assert FCFSScheduler().select_target(simulator) == 8


def test_nearest_request_selects_closest_pending_floor() -> None:
    simulator = make_simulator()
    simulator.elevator.current_floor = 5
    add_waiting_passenger(simulator, 1, source=2, destination=9)
    add_waiting_passenger(simulator, 2, source=7, destination=0)

    assert NearestRequestScheduler().select_target(simulator) == 7


def test_nearest_request_uses_dropoff_when_elevator_is_full() -> None:
    simulator = make_simulator(elevator_capacity=1)
    onboard = add_waiting_passenger(simulator, 1, source=0, destination=7)
    simulator.elevator.board([onboard], current_time=0)
    simulator.waiting_by_floor.pop(0)
    add_waiting_passenger(simulator, 2, source=1, destination=2)

    assert NearestRequestScheduler().select_target(simulator) == 7


def test_baselines_receive_identical_seeded_arrivals_and_report_metrics() -> None:
    config = make_simulator(duration=50, arrival_probability=0.3, seed=99).config
    fcfs_simulator = run_policy(config, FCFSScheduler())
    nearest_simulator = run_policy(config, NearestRequestScheduler())

    fcfs_arrivals = [(p.arrival_time, p.source_floor, p.destination_floor) for p in fcfs_simulator.passengers]
    nearest_arrivals = [(p.arrival_time, p.source_floor, p.destination_floor) for p in nearest_simulator.passengers]
    metrics = calculate_metrics(fcfs_simulator, "FCFS")

    assert fcfs_arrivals == nearest_arrivals
    assert metrics.algorithm == "FCFS"
    assert metrics.generated_passengers == len(fcfs_arrivals)
    assert metrics.served_passengers + metrics.unserved_passengers == metrics.generated_passengers
    assert metrics.elevator_travel_distance >= 0
