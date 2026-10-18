import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from elevator_rl.environment.config import SimulationConfig
from elevator_rl.environment.elevator_env import ElevatorEnv
from elevator_rl.environment.passenger import Passenger


def make_env(**overrides: int | float) -> ElevatorEnv:
    values: dict[str, int | float] = {
        "duration": 20,
        "arrival_probability": 0.0,
        "seed": 12,
        "elevator_capacity": 2,
    }
    values.update(overrides)
    return ElevatorEnv(SimulationConfig(**values))  # type: ignore[arg-type]


def add_request(env: ElevatorEnv, source: int, destination: int, passenger_id: int = 0) -> Passenger:
    passenger = Passenger(passenger_id, env.simulator.current_time, source, destination)
    env.simulator.passengers.append(passenger)
    env.simulator.waiting_by_floor[source].append(passenger)
    return passenger


def test_environment_passes_gymnasium_checker() -> None:
    check_env(make_env(), skip_render_check=True)


def test_observation_is_fixed_size_normalized_and_in_space() -> None:
    env = make_env()
    observation, info = env.reset(seed=5)

    assert observation.shape == (36,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert info["time"] == 0


def test_valid_floor_action_services_requested_floor() -> None:
    env = make_env()
    env.reset()
    passenger = add_request(env, source=0, destination=4)

    observation, reward, terminated, truncated, info = env.step(0)

    assert env.observation_space.contains(observation)
    assert passenger in env.simulator.elevator.passengers
    assert info["executed_target"] == 0
    assert reward == 0.0
    assert not terminated
    assert not truncated


def test_action_mask_marks_only_serviceable_floors() -> None:
    env = make_env()
    env.reset()
    add_request(env, source=3, destination=8)

    mask = env.action_masks()

    assert mask.dtype == bool
    assert mask[3]  # pickup at the waiting passenger's source floor
    assert not mask[8]  # not yet serviceable: the passenger has not boarded
    assert not mask[0]
    assert not mask[9]


def test_mask_includes_onboard_destinations_after_boarding() -> None:
    env = make_env()
    env.reset()
    add_request(env, source=3, destination=8)

    env.step(3)

    mask = env.action_masks()
    assert mask[8]
    assert not mask[3]  # the waiting queue at floor 3 is now empty


def test_action_mask_is_all_true_when_no_requests() -> None:
    env = make_env()
    env.reset()

    mask = env.action_masks()

    assert mask.all()


def test_action_masking_can_be_disabled_for_ablation() -> None:
    env = make_env()
    env.use_action_masking = False
    env.reset()
    add_request(env, source=3, destination=8)

    assert env.action_masks().all()

    # an unrequested floor is still executable when masking is off
    _, _, _, _, info = env.step(9)

    assert info["executed_target"] == 9


def test_unrequested_action_with_pending_requests_raises() -> None:
    env = make_env()
    env.reset()
    add_request(env, source=3, destination=8)

    with pytest.raises(ValueError, match="currently requested floor"):
        env.step(9)


def test_invalid_action_is_never_executed() -> None:
    env = make_env()
    env.reset()
    add_request(env, source=3, destination=8)

    mask = env.action_masks()
    action = int(np.argwhere(mask).ravel()[0])

    _, _, _, _, info = env.step(action)

    assert info["executed_target"] == action


def test_completion_reward_and_movement_penalty_are_reported() -> None:
    env = make_env()
    env.reset()
    passenger = add_request(env, source=0, destination=2)
    env.step(0)

    _, reward, _, _, info = env.step(2)

    assert passenger.completion_time == 2
    assert info["reward_components"]["completion"] == 2.0
    assert info["reward_components"]["movement"] == -0.04
    assert reward == pytest.approx(1.96)


def test_no_requests_advances_one_tick() -> None:
    env = make_env()
    env.reset()

    _, _, _, _, info = env.step(7)

    assert info["executed_target"] is None
    assert info["time"] == 1


def test_episode_terminates_at_configured_duration() -> None:
    env = make_env(duration=2)
    env.reset()

    _, _, terminated, truncated, _ = env.step(0)
    _, _, terminated, truncated, _ = env.step(0)

    assert terminated
    assert not truncated
    with pytest.raises(RuntimeError, match="call reset"):
        env.step(0)


def test_reset_without_seed_advances_episode_seed() -> None:
    env = make_env(seed=7)

    env.reset()
    assert env.config.seed == 7

    env.reset()
    assert env.config.seed == 8

    env.reset(seed=100)  # explicit seed is honoured and does not consume the counter
    assert env.config.seed == 100

    env.reset()
    assert env.config.seed == 9


def test_successive_resets_yield_different_arrival_scenarios() -> None:
    env = make_env(arrival_probability=0.4, duration=30)

    traces = []
    for _ in range(3):
        env.reset()
        terminated = False
        while not terminated:
            mask = env.action_masks()
            action = int(np.argwhere(mask).ravel()[0])
            _, _, terminated, _, _ = env.step(action)
        traces.append(
            tuple((p.arrival_time, p.source_floor, p.destination_floor) for p in env.simulator.passengers)
        )

    assert len(set(traces)) == 3


def test_traffic_variation_samples_arrival_probability_in_range() -> None:
    env = make_env(seed=3)
    env.arrival_rate_range = (0.10, 0.30)

    values = set()
    for _ in range(20):
        env.reset()
        values.add(round(env.config.arrival_probability, 4))

    assert values
    assert all(0.10 <= value <= 0.30 for value in values)
    assert len(values) > 1  # the traffic level changes between episodes
