# Intelligent Elevator Scheduling using Reinforcement Learning

This is a beginner-friendly academic project that compares rule-based elevator
schedulers with a PPO reinforcement-learning controller.

## Current progress: Phases 6 and 7 complete

The project contains a tested, deterministic single-elevator simulation core,
plus two rule-based schedulers: First-Come-First-Serve (FCFS) and
Nearest-Request. The simulator is exposed as a Gymnasium environment and
trained with MaskablePPO. A trained model, evaluation comparison, and an
animated side-by-side demo GIF are produced by the commands below.

## Run the baseline demonstration

```powershell
python main.py
```

The simulator uses discrete time. Moving one floor consumes one tick, and
passenger arrivals can occur during travel. At an arrived floor, passengers
exit before waiting passengers board, in first-arrival order. The command runs
FCFS and Nearest-Request using the same seed, duration, and arrival rate.

## Baseline policies

- **FCFS:** serves the oldest unfinished passenger. It first visits the
  passenger's source floor, then their destination after they board.
- **Nearest-Request:** serves the pending pickup or drop-off floor nearest to
  the elevator's current position.

The reported metrics include passenger waiting and journey times, served and
unserved passengers, throughput, and elevator travel distance.

## Reinforcement-learning environment

`ElevatorEnv` uses a fixed `Discrete(10)` action space: action `0` targets
floor 0 through action `9` targeting floor 9. Because the set of *currently*
requested floors changes over time, the environment exposes `action_masks()`:
a boolean mask marking exactly the floors that are serviceable right now
(onboard destinations, plus pickup floors while capacity remains). Training
uses `MaskablePPO` from `sb3_contrib`, so masked (invalid) floors are never
chosen and the action the agent picks is always executed. When no requests
exist the mask is all-True and the environment waits for one tick.

The normalized 36-value observation contains elevator position, direction,
load, remaining capacity, waiting passenger counts by source floor, waiting
destination demand, onboard destination demand, elapsed-time fraction, and
total pending demand.

The step reward is configurable in `SimulationConfig`:

```text
+ 2.00 × completed passengers
- 0.05 × newly accrued passenger waiting time
- 0.02 × floors travelled
- 0.10 × passengers waiting beyond the threshold
```

## Run automated tests

```powershell
python -m pytest
```

The suite checks seeded passenger generation, movement, boarding, exits,
capacity handling, baseline decisions, Gymnasium compatibility, normalized
observations, action masking, reward components, and episode termination.

## Run PPO smoke training

```powershell
python -m elevator_rl.agents.train_ppo --timesteps 2000 --seed 42
```

This brief run verifies that MaskablePPO (action-masked PPO from `sb3_contrib`)
can collect observations, choose only valid floor targets, learn from rewards,
save checkpoints, save a final model, and record episode reward history. It is
deliberately too short to establish PPO performance.

Outputs are saved to:

- `models/ppo_smoke.zip` — final trained smoke-test model
- `models/checkpoints/` — periodic checkpoints for longer runs
- `results/ppo_smoke_rewards.csv` — episode reward history

The MaskablePPO configuration uses an MLP policy with two 64-unit layers,
learning rate `0.0003`, 128-decision rollouts, and a fixed seed. These are
sensible starting settings, not tuned final hyperparameters.

Each training episode draws a different arrival scenario: the seed advances
per episode (never one repeated trace), and the passenger arrival probability
is sampled uniformly in `[0.08, 0.35]` so the policy trains across low, normal,
and peak demand. Disable that with `--no-traffic-variation`.

## Full experiment and final outputs

Train a longer PPO model, then compare it with the two baselines across unseen
seeds and low, normal, and peak traffic:

```powershell
python -m elevator_rl.agents.train_ppo --timesteps 100000 --seed 42 --name ppo_elevator
python -m elevator_rl.evaluation.compare_algorithms --seeds 101,102,103,104,105
```

The comparison writes detailed and average CSV reports plus six PNG figures to
`results/`: PPO reward history, average and maximum wait, travel distance,
throughput, and traffic-level performance. The evaluation seeds are separate
from the training seed, and `compare_algorithms` loads the run named
`ppo_elevator` by default.

## Animated demonstration

Replay one seeded scenario under all three controllers at once and save a
three-panel GIF (`results/elevator_animated_demo.gif`):

```powershell
python -m elevator_rl.visualization.animate_demo --seed 101 --arrival 0.35 --duration 100 --fps 8
```

Every time unit becomes one frame. FCFS (blue), Nearest-Request (orange), and
PPO (green) each run their own simulator against the *same* passenger arrivals:
red dots at the floor line mark waiting passengers, the car shows how many
passengers it is carrying, and the arrow shows travel direction. Each panel
tracks served passengers, average wait, and travel distance so the faculty demo
can watch the learned policy respond to identical demand.

## Ablation study

Measure how much each design choice contributes by retraining the policy with
one component switched off at a time, then evaluating every variant on the same
seeds:

```powershell
python -m elevator_rl.evaluation.ablation_study --timesteps 40000 --seeds 101,102,103
```

Variants: `full`, `no_masking`, `no_long_wait`, `no_movement`, `no_waiting`, and
`no_variation` (trained on a single fixed scenario). A random policy and the two
rule-based baselines are included as references. Results are written to
`results/ablation_detailed.csv`, `results/ablation_summary.csv`, and three bar
charts (`ablation_reward.png`, `ablation_waiting_time.png`,
`ablation_unserved.png`).
