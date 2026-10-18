"""Tests for the `on_tick` tracing hook and the animation snapshot format."""

from __future__ import annotations

from .conftest import add_waiting_passenger, make_simulator


def test_snapshot_contains_animation_fields() -> None:
    simulator = make_simulator()
    add_waiting_passenger(simulator, 1, source=3, destination=7)
    simulator.elevator.board(simulator.waiting_by_floor.pop(3), current_time=0)

    from elevator_rl.visualization.animate_demo import snapshot

    snap = snapshot(simulator)

    assert snap["floor"] == 0
    assert snap["inside"] == 1
    assert snap["waiting"] == {}
    assert snap["generated"] == 1
    assert snap["travel_distance"] == 0


def test_move_to_reports_one_snapshot_per_travelled_floor_and_door_service() -> None:
    simulator = make_simulator()
    passenger = add_waiting_passenger(simulator, 1, source=0, destination=5)
    observed: list[tuple[int, int]] = []

    simulator.move_to(0, on_tick=lambda sim: observed.append((sim.current_time, len(sim.elevator.passengers))))
    simulator.move_to(5, on_tick=lambda sim: observed.append((sim.current_time, len(sim.elevator.passengers))))

    assert passenger.status.value == "completed"
    traveller_ticks = [entry for entry in observed if entry[0] >= 1]
    assert [time for time, _ in traveller_ticks] == [1, 2, 3, 4, 5, 5]
    assert traveller_ticks[-1][1] == 0