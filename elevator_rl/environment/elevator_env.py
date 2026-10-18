"""Gymnasium environment for high-level elevator scheduling with PPO."""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np

from .config import SimulationConfig
from .passenger import PassengerStatus
from elevator_rl.simulation.simulator import ElevatorSimulator


class ElevatorEnv(gym.Env[np.ndarray, int]):
    """Select a target floor while the simulator handles travel and service.

    Action ``i`` selects floor ``i``.  Because the number of currently requested
    floors changes over time, the environment exposes ``action_masks()`` for
    MaskablePPO: a boolean array marking exactly the floors that are serviceable
    right now.  Masked (invalid) floors are never chosen by the policy, so the
    action the agent picks is always executed.  When no requests exist the mask
    is all-True and the elevator idles for one tick to allow new arrivals.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: SimulationConfig | None = None) -> None:
        super().__init__()
        self.base_config = config or SimulationConfig()
        self.config = self.base_config
        self.simulator = ElevatorSimulator(self.config)
        self.action_space = gym.spaces.Discrete(self.config.num_floors)

        # 4 elevator features + 3 vectors of floor demand + time and pending demand.
        observation_size = 4 + (3 * self.config.num_floors) + 2
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(observation_size,),
            dtype=np.float32,
        )
        # Training-time scenario variation: unsets to a single, repeated trace.
        self.arrival_rate_range: tuple[float, float] | None = None
        self._episode_seed_index = 0
        # When False every floor is masked valid, so the policy may pick any
        # floor (used by the ablation study to measure the value of masking).
        self.use_action_masking: bool = True
        # When set, every training episode reuses this exact scenario (used by
        # the ablation study to measure the value of scenario variation).
        self.fixed_episode_seed: int | None = None

    def _total_accumulated_wait(self) -> int:
        """Total wait already accrued by all passengers, including boarded ones."""
        return sum(passenger.waiting_time(self.simulator.current_time) for passenger in self.simulator.passengers)

    def _observation(self) -> np.ndarray:
        elevator = self.simulator.elevator
        floors = self.config.num_floors
        capacity = self.config.elevator_capacity
        waiting_by_source = np.zeros(floors, dtype=np.float32)
        waiting_by_destination = np.zeros(floors, dtype=np.float32)
        onboard_by_destination = np.zeros(floors, dtype=np.float32)

        for floor, queue in self.simulator.waiting_by_floor.items():
            waiting_by_source[floor] = min(len(queue) / capacity, 1.0)
            for passenger in queue:
                waiting_by_destination[passenger.destination_floor] += 1.0 / capacity
        for passenger in elevator.passengers:
            onboard_by_destination[passenger.destination_floor] += 1.0 / capacity

        direction_normalized = (elevator.direction + 1) / 2
        pending = sum(
            passenger.status != PassengerStatus.COMPLETED
            for passenger in self.simulator.passengers
        )
        observation = np.concatenate(
            (
                np.array(
                    [
                        elevator.current_floor / (floors - 1),
                        direction_normalized,
                        len(elevator.passengers) / capacity,
                        elevator.remaining_capacity / capacity,
                    ],
                    dtype=np.float32,
                ),
                np.clip(waiting_by_source, 0.0, 1.0),
                np.clip(waiting_by_destination, 0.0, 1.0),
                np.clip(onboard_by_destination, 0.0, 1.0),
                np.array(
                    [
                        self.simulator.current_time / self.config.duration,
                        min(pending / (floors * capacity), 1.0),
                    ],
                    dtype=np.float32,
                ),
            )
        )
        return observation.astype(np.float32)

    def _serviceable_floors(self) -> list[int]:
        """Return targets that can make progress with the current capacity."""
        floors = {passenger.destination_floor for passenger in self.simulator.elevator.passengers}
        if self.simulator.elevator.remaining_capacity > 0:
            floors.update(floor for floor, queue in self.simulator.waiting_by_floor.items() if queue)
        return sorted(floors)

    def action_masks(self) -> np.ndarray:
        """Return a boolean mask marking currently serviceable target floors.

        Valid targets are onboard destination floors and, while capacity
        remains, source floors with waiting passengers.  When nothing is pending
        every floor is marked valid so the policy still produces an action; the
        environment simply idles during the step.
        """
        if not self.use_action_masking:
            return np.ones(self.action_space.n, dtype=bool)
        mask = np.zeros(self.action_space.n, dtype=bool)
        for floor in self._serviceable_floors():
            mask[floor] = True
        if not mask.any():
            mask[:] = True
        return mask

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a fresh episode.

        An explicit ``seed`` is honoured as-is and produces a deterministic
        scenario (used by evaluation).  When called without a seed, as PPO does
        at the start of each training episode, the environment advances an
        internal counter so that every episode sees a *different* arrival trace
        instead of replaying one scenario forever.  If ``arrival_rate_range`` is
        set, the traffic intensity is also sampled inside that range per
        episode, training the policy to handle varied demand.
        """
        super().reset(seed=seed)
        del options  # Reserved by Gymnasium; no reset options are needed yet.
        if seed is None:
            if self.fixed_episode_seed is not None:
                seed = self.fixed_episode_seed
            else:
                seed = self.base_config.seed + self._episode_seed_index
                self._episode_seed_index += 1
        config = replace(self.base_config, seed=seed)
        if self.arrival_rate_range is not None:
            low, high = self.arrival_rate_range
            arrival_rate = random.Random(seed).uniform(low, high)
            config = replace(config, arrival_probability=arrival_rate)
        self.config = config
        self.simulator = ElevatorSimulator(config)
        return self._observation(), {"time": 0}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one high-level target selection and return its accumulated reward."""
        if not self.action_space.contains(action):
            raise ValueError(f"action must be an integer floor from 0 to {self.config.num_floors - 1}")
        if self.simulator.is_finished:
            raise RuntimeError("episode is finished; call reset() before step()")

        wait_before = self._total_accumulated_wait()
        completed_before = len(self.simulator.completed_passengers)
        distance_before = self.simulator.elevator.travel_distance
        requested_floors = self._serviceable_floors()
        executed_target: int | None

        if requested_floors:
            if self.use_action_masking and action not in requested_floors:
                raise ValueError(f"action {action} is not a currently requested floor")
            executed_target = action
            self.simulator.move_to(executed_target)
        else:
            executed_target = None
            self.simulator.wait_one_tick()

        waiting_increment = self._total_accumulated_wait() - wait_before
        completed_now = len(self.simulator.completed_passengers) - completed_before
        distance_now = self.simulator.elevator.travel_distance - distance_before
        long_waiting_count = sum(
            passenger.status == PassengerStatus.WAITING
            and passenger.waiting_time(self.simulator.current_time) > self.config.long_wait_threshold
            for passenger in self.simulator.passengers
        )
        components = {
            "completion": self.config.completion_reward * completed_now,
            "waiting": -self.config.waiting_penalty * waiting_increment,
            "movement": -self.config.movement_penalty * distance_now,
            "long_wait": -self.config.long_wait_penalty * long_waiting_count,
        }
        reward = float(sum(components.values()))
        terminated = self.simulator.is_finished
        info: dict[str, Any] = {
            "selected_floor": int(action),
            "executed_target": executed_target,
            "time": self.simulator.current_time,
            "reward_components": components,
        }
        return self._observation(), reward, terminated, False, info
