"""Create a clean 2D snapshot of the simulated elevator."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from elevator_rl.simulation.simulator import ElevatorSimulator


def save_snapshot(simulator: ElevatorSimulator, algorithm: str, output_path: Path) -> Path:
    """Draw floors, elevator, demand, and compact performance statistics."""
    floors = simulator.config.num_floors
    figure, axis = plt.subplots(figsize=(8, 8))
    for floor in range(floors):
        axis.hlines(floor, 0, 8, color="black", linewidth=0.5)
        waiting = len(simulator.waiting_by_floor.get(floor, []))
        axis.text(0.15, floor + 0.15, f"Floor {floor}", fontsize=9)
        axis.text(2.0, floor + 0.15, f"Waiting: {waiting}", fontsize=9)
    elevator_floor = simulator.elevator.current_floor
    axis.add_patch(Rectangle((4.0, elevator_floor), 1.2, 0.8, facecolor="#4C78A8", edgecolor="black"))
    axis.text(4.1, elevator_floor + 0.35, "ELEVATOR", color="white", fontsize=8)
    axis.text(5.6, elevator_floor + 0.15, f"Inside: {len(simulator.elevator.passengers)}", fontsize=9)
    summary = simulator.summary()
    axis.text(
        0.15,
        floors + 0.25,
        f"Algorithm: {algorithm} | Time: {simulator.current_time} | Target: {simulator.elevator.target_floor}",
        fontsize=10,
        fontweight="bold",
    )
    axis.text(
        0.15,
        floors + 0.65,
        f"Completed: {summary['completed']} | Distance: {summary['travel_distance']} | Avg wait: {summary['average_waiting_time']:.2f}",
        fontsize=9,
    )
    axis.set(xlim=(0, 8), ylim=(-0.2, floors + 1.1), xticks=[], yticks=[])
    axis.set_title("Intelligent Elevator Scheduling — Simulation Snapshot")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path
