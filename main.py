"""Run the Phase 3 rule-based scheduler demonstration."""

from elevator_rl.baselines.fcfs import FCFSScheduler
from elevator_rl.baselines.nearest_request import NearestRequestScheduler
from elevator_rl.environment.config import SimulationConfig
from elevator_rl.evaluation.metrics import calculate_metrics
from elevator_rl.simulation.runner import run_policy


def main() -> None:
    config = SimulationConfig(seed=7, duration=100, arrival_probability=0.20)
    for scheduler in (FCFSScheduler(), NearestRequestScheduler()):
        simulator = run_policy(config, scheduler)
        metrics = calculate_metrics(simulator, scheduler.name)
        print(f"\n{scheduler.name} baseline")
        for name, value in metrics.to_dict().items():
            print(f"{name}: {value}")


if __name__ == "__main__":
    main()
