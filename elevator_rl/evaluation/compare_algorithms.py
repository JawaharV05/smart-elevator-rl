"""Multi-seed traffic experiments and report-quality comparison plots."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from elevator_rl.agents.evaluate_ppo import run_ppo_episode
from elevator_rl.baselines.fcfs import FCFSScheduler
from elevator_rl.baselines.nearest_request import NearestRequestScheduler
from elevator_rl.environment.config import SimulationConfig
from elevator_rl.evaluation.metrics import calculate_metrics
from elevator_rl.simulation.runner import run_policy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAFFIC_LEVELS = {"Low": 0.08, "Normal": 0.20, "Peak": 0.35}


def run_comparison(model_path: Path, seeds: list[int], duration: int = 150) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all policies on identical seeds across three traffic levels."""
    rows: list[dict[str, object]] = []
    schedulers = (FCFSScheduler(), NearestRequestScheduler())
    for traffic, arrival_probability in TRAFFIC_LEVELS.items():
        for seed in seeds:
            config = SimulationConfig(seed=seed, duration=duration, arrival_probability=arrival_probability)
            for scheduler in schedulers:
                metrics = calculate_metrics(run_policy(config, scheduler), scheduler.name)
                row = asdict(metrics)
                row.update({"traffic": traffic, "seed": seed, "total_reward": None})
                rows.append(row)
            reward, metrics = run_ppo_episode(model_path, config)
            row = asdict(metrics)
            row.update({"traffic": traffic, "seed": seed, "total_reward": reward})
            rows.append(row)
    detailed = pd.DataFrame(rows)
    numeric = detailed.select_dtypes(include="number").columns.tolist()
    aggregate = detailed.groupby(["traffic", "algorithm"], as_index=False)[numeric].mean()
    return detailed, aggregate


def _bar_plot(data: pd.DataFrame, metric: str, title: str, ylabel: str, output: Path) -> None:
    pivot = data.pivot(index="traffic", columns="algorithm", values=metric).reindex(list(TRAFFIC_LEVELS))
    axis = pivot.plot(kind="bar", figsize=(8, 5), color=["#4C78A8", "#F58518", "#54A24B"])
    axis.set_title(title)
    axis.set_xlabel("Traffic level")
    axis.set_ylabel(ylabel)
    axis.legend(title="Algorithm")
    axis.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def create_plots(aggregate: pd.DataFrame, reward_history: Path, output_dir: Path) -> list[Path]:
    """Create the requested report figures in a single results directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    if reward_history.exists():
        history = pd.read_csv(reward_history)
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(history["timestep"], history["episode_reward"], marker="o", color="#4C78A8")
        axis.set(title="PPO training reward", xlabel="Training decision", ylabel="Episode reward")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        path = output_dir / "ppo_training_reward.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        outputs.append(path)
    for metric, title, ylabel, filename in (
        ("average_waiting_time", "Average waiting time", "Ticks", "average_waiting_time.png"),
        ("maximum_waiting_time", "Maximum waiting time", "Ticks", "maximum_waiting_time.png"),
        ("elevator_travel_distance", "Elevator travel distance", "Floors", "travel_distance.png"),
        ("throughput_per_tick", "Passenger throughput", "Completed passengers per tick", "throughput.png"),
        ("served_passengers", "Performance under traffic levels", "Passengers served", "traffic_performance.png"),
    ):
        path = output_dir / filename
        _bar_plot(aggregate, metric, title, ylabel, path)
        outputs.append(path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PPO with FCFS and Nearest-Request.")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "ppo_elevator.zip")
    parser.add_argument("--seeds", default="101,102,103,104,105", help="Comma-separated unseen evaluation seeds.")
    parser.add_argument("--duration", type=int, default=150)
    arguments = parser.parse_args()
    seeds = [int(seed.strip()) for seed in arguments.seeds.split(",") if seed.strip()]
    detailed, aggregate = run_comparison(arguments.model, seeds, arguments.duration)
    results = PROJECT_ROOT / "results"
    results.mkdir(exist_ok=True)
    detailed.to_csv(results / "comparison_detailed.csv", index=False)
    aggregate.to_csv(results / "comparison_summary.csv", index=False)
    reward_history = results / f"{arguments.model.stem}_rewards.csv"
    create_plots(aggregate, reward_history, results)
    print("Comparison complete")
    print(aggregate.to_string(index=False))
    print(f"CSV and plots saved in: {results}")


if __name__ == "__main__":
    main()
