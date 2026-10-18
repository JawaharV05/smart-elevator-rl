from elevator_rl.environment.config import SimulationConfig
from elevator_rl.simulation.passenger_generator import PassengerGenerator


def test_generator_is_reproducible_for_same_seed() -> None:
    config = SimulationConfig(seed=41, arrival_probability=0.6)
    first = PassengerGenerator(config)
    second = PassengerGenerator(config)

    first_sequence = [first.generate(tick) for tick in range(1, 25)]
    second_sequence = [second.generate(tick) for tick in range(1, 25)]

    assert first_sequence == second_sequence


def test_generator_creates_valid_distinct_floors() -> None:
    generator = PassengerGenerator(SimulationConfig(seed=3, arrival_probability=1.0))

    arrivals = [generator.generate(tick) for tick in range(1, 10)]

    assert [arrival[0] for arrival in arrivals if arrival is not None] == list(range(9))
    assert all(
        arrival is not None
        and 0 <= arrival[1] < 10
        and 0 <= arrival[2] < 10
        and arrival[1] != arrival[2]
        for arrival in arrivals
    )

