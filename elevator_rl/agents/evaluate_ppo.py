"""Small post-training evaluation helper for PPO smoke runs."""

from __future__ import annotations

from pathlib import Path

from sb3_contrib import MaskablePPO

from elevator_rl.environment.config import SimulationConfig
from elevator_rl.environment.elevator_env import ElevatorEnv
from elevator_rl.evaluation.metrics import SimulationMetrics, calculate_metrics


def evaluate_model(model_path: Path, seed: int = 10_042) -> tuple[float, SimulationMetrics]:
    """Run one unseen seeded episode and return reward plus common metrics."""
    environment = ElevatorEnv(SimulationConfig(seed=seed, duration=100, arrival_probability=0.20))
    model = MaskablePPO.load(model_path)
    observation, _ = environment.reset(seed=seed)
    episode_reward = 0.0
    terminated = False
    while not terminated:
        action, _ = model.predict(observation, deterministic=True, action_masks=environment.action_masks())
        observation, reward, terminated, _, _ = environment.step(int(action))
        episode_reward += reward
    return episode_reward, calculate_metrics(environment.simulator, "PPO-smoke")


def run_ppo_episode(model_path: Path, config: SimulationConfig) -> tuple[float, SimulationMetrics]:
    """Evaluate a saved MaskablePPO policy under any configured traffic scenario."""
    environment = ElevatorEnv(config)
    model = MaskablePPO.load(model_path)
    observation, _ = environment.reset(seed=config.seed)
    total_reward = 0.0
    terminated = False
    while not terminated:
        action, _ = model.predict(observation, deterministic=True, action_masks=environment.action_masks())
        observation, reward, terminated, _, _ = environment.step(int(action))
        total_reward += reward
    return total_reward, calculate_metrics(environment.simulator, "PPO")
