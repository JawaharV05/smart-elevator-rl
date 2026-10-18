"""Short, reproducible PPO training runs for the elevator environment."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from elevator_rl.environment.config import SimulationConfig
from elevator_rl.environment.elevator_env import ElevatorEnv


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class EpisodeRewardCallback(BaseCallback):
    """Store completed-episode rewards in a compact CSV training history."""

    def __init__(self, output_path: Path) -> None:
        super().__init__()
        self.output_path = output_path
        self.history: list[dict[str, float | int]] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is not None:
                self.history.append(
                    {
                        "timestep": self.num_timesteps,
                        "episode_reward": float(episode["r"]),
                        "episode_length": int(episode["l"]),
                    }
                )
        return True

    def _on_training_end(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["timestep", "episode_reward", "episode_length"])
            writer.writeheader()
            writer.writerows(self.history)


def build_model(
    config: SimulationConfig,
    seed: int,
    arrival_rate_range: tuple[float, float] | None = None,
    use_action_masking: bool = True,
    fixed_episode_seed: int | None = None,
) -> MaskablePPO:
    """Create MaskablePPO; optionally vary traffic demand during training.

    ``use_action_masking`` and ``fixed_episode_seed`` are toggles used by the
    ablation study to switch off a single design component at a time.
    """
    core = ElevatorEnv(config)
    core.arrival_rate_range = arrival_rate_range
    core.use_action_masking = use_action_masking
    core.fixed_episode_seed = fixed_episode_seed
    environment = ActionMasker(
        Monitor(core),
        lambda env: env.get_wrapper_attr("action_masks")(),
    )
    return MaskablePPO(
        policy="MlpPolicy",
        env=environment,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=5,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [64, 64]},
        seed=seed,
        verbose=0,
        device="cpu",
    )


def train_smoke(
    total_timesteps: int = 2_000,
    seed: int = 42,
    run_name: str = "ppo_smoke",
    model_path: Path | None = None,
    reward_history_path: Path | None = None,
    arrival_rate_range: tuple[float, float] | None = (0.08, 0.35),
    checkpoint_frequency: int = 1_000,
) -> tuple[MaskablePPO, list[dict[str, float | int]]]:
    """Train MaskablePPO and save model, checkpoints, and reward history.

    ``run_name`` names the saved artifacts: ``models/{run_name}.zip``,
    ``results/{run_name}_rewards.csv`` and checkpoint prefix ``{run_name}``.
    Short runs (default 2,000 decisions) validate PPO integration; longer runs
    train a usable policy.
    """
    if total_timesteps < 128:
        raise ValueError("total_timesteps must be at least 128")
    model_path = model_path or PROJECT_ROOT / "models" / f"{run_name}.zip"
    reward_history_path = reward_history_path or PROJECT_ROOT / "results" / f"{run_name}_rewards.csv"
    config = SimulationConfig(seed=seed, duration=150, arrival_probability=0.20)
    model = build_model(config, seed, arrival_rate_range)
    reward_callback = EpisodeRewardCallback(reward_history_path)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_frequency,
        save_path=str(PROJECT_ROOT / "models" / "checkpoints"),
        name_prefix=run_name,
    )
    model.learn(total_timesteps=total_timesteps, callback=[reward_callback, checkpoint_callback])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    return model, reward_callback.history


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PPO training for the elevator environment.")
    parser.add_argument("--timesteps", type=int, default=2_000, help="Training decisions; minimum 128.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for simulation and PPO.")
    parser.add_argument(
        "--arrival-min",
        type=float,
        default=0.08,
        help="Lowest per-episode arrival probability during training.",
    )
    parser.add_argument(
        "--arrival-max",
        type=float,
        default=0.35,
        help="Highest per-episode arrival probability during training.",
    )
    parser.add_argument(
        "--no-traffic-variation",
        action="store_true",
        help="Fix the arrival probability instead of sampling it per episode.",
    )
    parser.add_argument(
        "--name",
        default="ppo_smoke",
        help="Run name that names the saved model, reward history, and checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5_000,
        help="Save a checkpoint every N training decisions.",
    )
    arguments = parser.parse_args()

    arrival_rate_range = None if arguments.no_traffic_variation else (arguments.arrival_min, arguments.arrival_max)
    model, history = train_smoke(
        total_timesteps=arguments.timesteps,
        seed=arguments.seed,
        run_name=arguments.name,
        arrival_rate_range=arrival_rate_range,
        checkpoint_frequency=arguments.checkpoint_every,
    )
    print("PPO training complete")
    print(f"training decisions: {arguments.timesteps}")
    print(f"completed training episodes: {len(history)}")
    print(f"per-episode seed variation: enabled with base seed {arguments.seed}")
    print(f"per-episode traffic variation: {arrival_rate_range}")
    if history:
        print(f"last episode reward: {history[-1]['episode_reward']:.3f}")
    model_path = PROJECT_ROOT / "models" / f"{arguments.name}.zip"
    reward_history_path = PROJECT_ROOT / "results" / f"{arguments.name}_rewards.csv"
    print(f"model saved: {model_path}")
    print(f"reward history saved: {reward_history_path}")
    del model


if __name__ == "__main__":
    main()
