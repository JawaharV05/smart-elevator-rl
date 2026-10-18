"""Animated side-by-side video of FCFS, Nearest-Request, and MaskablePPO.

All three policies replay the *same* seeded scenario, so the faculty demo can
watch the RL controller react to exactly the passengers the baselines saw.
Each time unit becomes one frame; every policy runs its own simulator but the
passenger arrivals depend only on the seed, keeping the three panels in sync.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle
from sb3_contrib import MaskablePPO

from elevator_rl.baselines.fcfs import FCFSScheduler
from elevator_rl.baselines.nearest_request import NearestRequestScheduler
from elevator_rl.environment.config import SimulationConfig
from elevator_rl.environment.elevator_env import ElevatorEnv
from elevator_rl.simulation.runner import run_policy
from elevator_rl.simulation.simulator import ElevatorSimulator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL_STYLE = (
    ("FCFS", FCFSScheduler().name, "#4C78A8"),
    ("Nearest-Request", NearestRequestScheduler().name, "#F58518"),
    ("PPO", "PPO", "#54A24B"),
)


def snapshot(simulator: ElevatorSimulator) -> dict[str, float | int | dict[int, int]]:
    """Compact, animation-friendly picture of the building at one moment."""
    summary = simulator.summary()
    return {
        "floor": simulator.elevator.current_floor,
        "direction": simulator.elevator.direction,
        "target": simulator.elevator.target_floor,
        "inside": len(simulator.elevator.passengers),
        "waiting": {floor: len(queue) for floor, queue in simulator.waiting_by_floor.items()},
        "completed": summary["completed"],
        "generated": summary["generated"],
        "average_waiting_time": summary["average_waiting_time"],
        "travel_distance": summary["travel_distance"],
    }


def _record_time(trace: dict[int, dict[str, Any]]) -> Callable[[ElevatorSimulator], None]:
    def record(simulator: ElevatorSimulator) -> None:
        trace[simulator.current_time] = snapshot(simulator)

    return record


def collect_trace(policy: str, model_path: Path, config: SimulationConfig) -> dict[int, dict[str, Any]]:
    """Run one policy on the scenario and return one snapshot per time unit."""
    trace: dict[int, dict[str, Any]] = {}
    on_tick = _record_time(trace)
    if policy == "PPO":
        environment = ElevatorEnv(config)
        model = MaskablePPO.load(model_path)
        observation, _ = environment.reset(seed=config.seed)
        simulator = environment.simulator
        while not simulator.is_finished:
            serviceable = environment._serviceable_floors()
            if serviceable:
                action, _ = model.predict(observation, deterministic=True, action_masks=environment.action_masks())
                simulator.move_to(int(action), on_tick=on_tick)
            else:
                simulator.wait_one_tick(on_tick=on_tick)
            observation = environment._observation()
    else:
        scheduler = FCFSScheduler() if policy == "FCFS" else NearestRequestScheduler()
        run_policy(config, scheduler, on_tick=on_tick)
    return trace


def _draw_panel(
    axis: plt.Axes,
    trace: dict[int, dict[str, Any]],
    time: int,
    config: SimulationConfig,
    algorithm: str,
    color: str,
) -> None:
    snap = trace.get(time, trace[1])
    floors = config.num_floors
    waiting_counts = snap["waiting"] if isinstance(snap["waiting"], dict) else {}
    for floor in range(floors):
        axis.hlines(floor, 0, 8, color="0.55", linewidth=0.6)
        waiting = waiting_counts.get(floor, 0)
        if waiting:
            axis.plot(3.6, floor + 0.5, marker="o", color="#C85250", markersize=4 + waiting * 2)
            axis.text(4.1, floor + 0.55, f"{waiting}", color="#C85250", fontsize=8)
    elevator_floor = int(snap["floor"])
    axis.add_patch(
        Rectangle((5.0, elevator_floor), 1.2, 0.8, facecolor=color, edgecolor="black", zorder=3)
    )
    axis.text(5.15, elevator_floor + 0.35, f"{snap['inside']}", color="white", fontsize=9, zorder=4)
    direction = int(snap["direction"])
    arrow = {1: "\u25b2", -1: "\u25bc", 0: "\u25a0"}[direction]
    axis.text(6.5, elevator_floor + 0.35, arrow, fontsize=11, zorder=4)
    axis.set_xlim(0, 8)
    axis.set_ylim(-0.4, floors + 2.2)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(algorithm, color=color, fontsize=12, fontweight="bold")
    axis.text(
        4.0,
        floors + 1.4,
        f"t={time:03d}   served {snap['completed']}/{snap['generated']}",
        ha="center",
        fontsize=10,
    )
    axis.text(
        4.0,
        floors + 0.9,
        f"avg wait {snap['average_waiting_time']:.1f}   dist {snap['travel_distance']}",
        ha="center",
        fontsize=9,
        color="0.25",
    )


def render_animation(
    traces: dict[str, dict[int, dict[str, Any]]],
    config: SimulationConfig,
    output_path: Path,
    fps: int = 8,
    title: str = "Same hall call, three controllers",
) -> Path:
    """Compose the per-algorithm traces into a three-panel GIF."""
    figure, axes = plt.subplots(
        1, 3, figsize=(15, 6), sharey=True, sharex=True, gridspec_kw={"wspace": 0.05}
    )
    figure.suptitle(title, fontsize=13, fontweight="bold")
    time_steps = list(range(1, config.duration + 1))

    def update(time: int) -> None:
        for axis, (algorithm, _, color) in zip(axes, PANEL_STYLE):
            axis.clear()
            _draw_panel(axis, traces[algorithm], time, config, algorithm, color)

    animation = FuncAnimation(figure, update, frames=time_steps, interval=1000 / fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an animated comparison of elevator policies.")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--arrival", type=float, default=0.35, help="Passenger arrival probability (0.35 = peak).")
    parser.add_argument("--duration", type=int, default=100)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "ppo_elevator.zip")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "elevator_animated_demo.gif")
    arguments = parser.parse_args()

    config = SimulationConfig(
        seed=arguments.seed, duration=arguments.duration, arrival_probability=arguments.arrival
    )
    labels = {algorithm: scheduler_name for algorithm, scheduler_name, _ in PANEL_STYLE}
    traces = {
        algorithm: collect_trace(algorithm, arguments.model, config) for algorithm, _ in labels.items()
    }
    output = render_animation(traces, config, arguments.output, fps=arguments.fps)
    print(f"Animation saved: {output}")


if __name__ == "__main__":
    main()