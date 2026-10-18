from elevator_rl.agents.evaluate_ppo import evaluate_model
from elevator_rl.agents.train_ppo import PROJECT_ROOT, train_smoke


def test_ppo_can_train_save_load_and_evaluate() -> None:
    # Use ignored project artifact paths rather than pytest's system temp area,
    # which may be permission-restricted on managed Windows installations.
    model_path = PROJECT_ROOT / "models" / "test_ppo_smoke.zip"
    history_path = PROJECT_ROOT / "results" / "test_ppo_smoke_rewards.csv"

    _, history = train_smoke(
        total_timesteps=128,
        seed=4,
        model_path=model_path,
        reward_history_path=history_path,
    )
    reward, metrics = evaluate_model(model_path, seed=1004)

    assert model_path.exists()
    assert history_path.exists()
    assert isinstance(reward, float)
    assert metrics.algorithm == "PPO-smoke"
    assert metrics.duration == 100
    assert history
