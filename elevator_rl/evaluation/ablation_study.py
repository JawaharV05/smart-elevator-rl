"""Ablation study: measure the contribution of each design component.

Each variant retrains the policy with exactly one component switched off, then
every variant is evaluated on the *same* seeds so the effect of that component
is isolated.  A random (non-learning) policy and the two rule-based baselines
are included as reference points.

Run with::

    python -m elevator_rl.evaluation.ablation_study --timesteps 40000 --seeds 101,102,103
"""

from __future__ import annotations

import argparse
import random
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from elevator_rl.agents.evaluate_ppo import run_ppo_episode
from elevator_rl.agents.train_ppo import build_model
from elevator_rl.baselines.fcfs import FCFSScheduler
from elevator_rl.baselines.nearest_request import NearestRequestScheduler
from elevator_rl.environment.config import SimulationConfig
from elevator_rl.evaluation.metrics import calculate_metrics
from elevator_rl.simulation.runner import run_policy
from elevator_rl.simulation.simulator import ElevatorSimulator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAFFIC_LEVELS = {"Low": 0.08, "Normal": 0.20, "Peak": 0.35}


class RandomScheduler:
    """Pick a uniformly random serviceable floor — no learning involved."""

    name = "Random"

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def select_target(self, simulator: ElevatorSimulator) -> int | None:
        floors = simulator.pending_floors()
        if not floors:
            return None
        return self._rng.choice(floors)


def train_variant(
    name: str,
    *,
    total_timesteps: int,
    seed: int = 42,
    use_action_masking: bool = True,
    arrival_rate_range: tuple[float, float] | None = (0.08, 0.35),
    fixed_episode_seed: int | None = None,
    config_overrides: dict[str, float] | None = None,
    force: bool = False,
) -> Path:
    """Train (or reuse) one ablation variant and return its model path."""
    model_path = PROJECT_ROOT / "models" / "ablations" / f"{name}.zip"
    if model_path.exists() and not force:
        print(f"  reusing cached model: {model_path.name}")
        return model_path

    config = SimulationConfig(seed=seed, duration=150, arrival_probability=0.20)
    if config_overrides:
        config = replace(config, **config_overrides)
    model = build_model(
        config,
        seed,
        arrival_rate_range,
        use_action_masking,
        fixed_episode_seed,
    )
    model.learn(total_timesteps=total_timesteps)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    return model_path


def evaluate_model(model_path: Path, seeds: list[int], duration: int = 150) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for traffic, arrival in TRAFFIC_LEVELS.items():
        for seed in seeds:
            config = SimulationConfig(seed=seed, duration=duration, arrival_probability=arrival)
            reward, metrics = run_ppo_episode(model_path, config)
            row = metrics.to_dict()
            row.update({"traffic": traffic, "seed": seed, "total_reward": reward})
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_scheduler(scheduler, seeds: list[int], duration: int = 150) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for traffic, arrival in TRAFFIC_LEVELS.items():
        for seed in seeds:
            config = SimulationConfig(seed=seed, duration=duration, arrival_probability=arrival)
            metrics = calculate_metrics(run_policy(config, scheduler), scheduler.name)
            row = metrics.to_dict()
            row.update({"traffic": traffic, "seed": seed, "total_reward": float("nan")})
            rows.append(row)
    return pd.DataFrame(rows)


# name -> overrides applied to that variant (everything else stays at default)
VARIANTS: dict[str, dict[str, object]] = {
    "full": {},
    "no_masking": {"use_action_masking": False},
    "no_long_wait": {"config_overrides": {"long_wait_penalty": 0.0}},
    "no_movement": {"config_overrides": {"movement_penalty": 0.0}},
    "no_waiting": {"config_overrides": {"waiting_penalty": 0.0}},
    "no_variation": {"arrival_rate_range": None, "fixed_episode_seed": 42},
}


def run_ablation(
    total_timesteps: int,
    seeds: list[int],
    duration: int = 150,
    force: bool = False,
    variants: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train/evaluate every variant and return (detailed, summary) tables."""
    selected = variants or list(VARIANTS)
    frames: list[pd.DataFrame] = []
    for name in selected:
        kwargs = VARIANTS[name]
        print(f"[{name}] training/evaluating ...")
        model_path = train_variant(name, total_timesteps=total_timesteps, force=force, **kwargs)
        frame = evaluate_model(model_path, seeds, duration)
        frame["variant"] = name
        frames.append(frame)

    if variants is None:
        for scheduler in (RandomScheduler(), FCFSScheduler(), NearestRequestScheduler()):
            frame = evaluate_scheduler(scheduler, seeds, duration)
            frame["variant"] = scheduler.name
            frames.append(frame)

    detailed = pd.concat(frames, ignore_index=True)
    numeric = detailed.select_dtypes(include="number").columns.tolist()
    summary = detailed.groupby(["traffic", "variant"], as_index=False)[numeric].mean()
    return detailed, summary


def create_plots(summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Bar charts comparing variants on the key metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    order = list(VARIANTS) + ["Random", "FCFS", "Nearest-Request"]
    colors = plt.get_cmap("tab10")

    outputs: list[Path] = []
    for metric, title, ylabel, filename in (
        ("total_reward", "Average episode reward", "Reward", "ablation_reward.png"),
        ("average_waiting_time", "Average waiting time", "Ticks", "ablation_waiting_time.png"),
        ("unserved_passengers", "Unserved passengers", "Passengers", "ablation_unserved.png"),
    ):
        pivot = summary.pivot(index="traffic", columns="variant", values=metric).reindex(list(TRAFFIC_LEVELS))
        pivot = pivot[[column for column in order if column in pivot.columns]]
        figure, axis = plt.subplots(figsize=(11, 5))
        pivot.plot(kind="bar", ax=axis, color=[colors(i) for i in range(len(pivot.columns))])
        axis.set_title(title)
        axis.set_xlabel("Traffic level")
        axis.set_ylabel(ylabel)
        axis.legend(title="Variant", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        path = output_dir / filename
        figure.savefig(path, dpi=180)
        plt.close(figure)
        outputs.append(path)
    return outputs


def collect_partials() -> None:
    """Combine per-variant partial CSVs into the final tables and plots."""
    results = PROJECT_ROOT / "results"
    partials = sorted(results.glob("ablation_partial_*.csv"))
    if not partials:
        print("No partial files found.")
        return
    detailed = pd.concat([pd.read_csv(path) for path in partials], ignore_index=True)
    numeric = detailed.select_dtypes(include="number").columns.tolist()
    summary = detailed.groupby(["traffic", "variant"], as_index=False)[numeric].mean()
    detailed.to_csv(results / "ablation_detailed.csv", index=False)
    summary.to_csv(results / "ablation_summary.csv", index=False)
    create_plots(summary, results)
    print("\nAblation summary (mean over seeds):")
    print(summary.to_string(index=False))
    print(f"\nCombined {len(partials)} partial files.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ablation study.")
    parser.add_argument("--timesteps", type=int, default=40_000, help="Training decisions per variant.")
    parser.add_argument("--seeds", default="101,102,103", help="Comma-separated evaluation seeds.")
    parser.add_argument("--duration", type=int, default=150)
    parser.add_argument("--force", action="store_true", help="Retrain even if a cached model exists.")
    parser.add_argument("--only", default=None, help="Comma-separated variant names (parallel mode).")
    parser.add_argument("--collect", action="store_true", help="Combine partial results and plot.")
    arguments = parser.parse_args()

    if arguments.collect:
        collect_partials()
        return

    seeds = [int(seed.strip()) for seed in arguments.seeds.split(",") if seed.strip()]

    if arguments.only:
        variants = [name.strip() for name in arguments.only.split(",") if name.strip()]
        results = PROJECT_ROOT / "results"
        results.mkdir(exist_ok=True)
        if variants == ["baselines"]:
            frames = []
            for scheduler in (RandomScheduler(), FCFSScheduler(), NearestRequestScheduler()):
                frame = evaluate_scheduler(scheduler, seeds, arguments.duration)
                frame["variant"] = scheduler.name
                frames.append(frame)
            pd.concat(frames, ignore_index=True).to_csv(
                results / "ablation_partial_baselines.csv", index=False
            )
            print("Saved partial results for: baselines")
            return
        detailed, _ = run_ablation(arguments.timesteps, seeds, arguments.duration, arguments.force, variants)
        for name in variants:
            detailed[detailed["variant"] == name].to_csv(
                results / f"ablation_partial_{name}.csv", index=False
            )
        print(f"Saved partial results for: {', '.join(variants)}")
        return

    detailed, summary = run_ablation(arguments.timesteps, seeds, arguments.duration, arguments.force)

    results = PROJECT_ROOT / "results"
    results.mkdir(exist_ok=True)
    detailed.to_csv(results / "ablation_detailed.csv", index=False)
    summary.to_csv(results / "ablation_summary.csv", index=False)
    create_plots(summary, results)

    print("\nAblation summary (mean over seeds):")
    print(summary.to_string(index=False))
    print(f"\nSaved: results/ablation_detailed.csv, results/ablation_summary.csv, ablation_*.png")


if __name__ == "__main__":
    main()