"""Тесты моделей транспортного потока."""

import numpy as np
import pytest

from traffic_flow import SNFS, NaSch
from traffic_flow.experiments import fundamental_diagram, space_time


def run(model, steps):
    for _ in range(steps):
        model.step()
    return model


@pytest.mark.parametrize("make", [
    lambda rng: NaSch(200, 5, 0.5, 0.5, rng),
    lambda rng: SNFS(200, 3, 0.5, 0.5, 0.1, rng),
])
@pytest.mark.parametrize("density", [0.05, 0.3, 0.7, 1.0])
def test_no_collisions_and_car_count_preserved(make, density):
    model = make(np.random.default_rng(1))
    model.place_cars(density)
    n = model.n_cars
    for _ in range(300):
        model.step()
        # Каждая клетка занята не более чем одной машиной.
        assert np.unique(model.positions).size == n
        assert np.all((model.velocities >= 0) & (model.velocities <= model.v_max))


@pytest.mark.parametrize("make", [
    lambda rng: NaSch(100, 5, 0.5, 1.0, rng),
    lambda rng: SNFS(100, 3, 0.5, 0.5, 0.1, rng),
])
def test_cars_never_overtake(make):
    # Порядок машин на кольце сохраняется: между машиной и её лидером
    # расстояние всегда положительно и не больше длины кольца.
    model = make(np.random.default_rng(2))
    model.place_cars(0.3)
    for _ in range(200):
        before = model.positions.copy()
        model.step()
        travelled = (model.positions - before) % model.length
        np.testing.assert_array_equal(travelled, model.velocities)
        assert np.all(model.gaps() >= 0)


def test_deterministic_nasch_reaches_theoretical_max_flow():
    # Без случайного торможения максимум потока — v_max / (v_max + 1)
    # при плотности 1 / (v_max + 1).
    fd = fundamental_diagram(
        lambda rng: NaSch(600, 5, 0.5, human_ratio=0.0, rng=rng),
        np.array([1 / 6]), warmup=600, measure=100,
    )
    assert fd.flows[0] == pytest.approx(5 / 6, abs=0.01)


def test_deterministic_nasch_high_density_flow_is_one_minus_rho():
    fd = fundamental_diagram(
        lambda rng: NaSch(500, 5, 0.5, human_ratio=0.0, rng=rng),
        np.array([0.5]), warmup=1000, measure=100,
    )
    assert fd.flows[0] == pytest.approx(0.5, abs=0.01)


def test_human_drivers_reduce_flow():
    flows = [
        fundamental_diagram(
            lambda rng, h=h: NaSch(500, 5, 0.5, h, rng), np.array([0.2]), 500, 200
        ).flows[0]
        for h in (0.0, 1.0)
    ]
    assert flows[1] < flows[0]


def test_anticipation_increases_flow_without_slow_start():
    flows = [
        fundamental_diagram(
            lambda rng, r=r: SNFS(500, 3, q=0.0, r=r, p_brake=0.1, rng=rng),
            np.array([0.4]), 500, 200,
        ).flows[0]
        for r in (0.0, 1.0)
    ]
    assert flows[1] > flows[0] + 0.1


def test_space_time_marks_every_car():
    model = NaSch(100, 5, 0.3, 1.0, np.random.default_rng(3))
    model.place_cars(0.25)
    diagram = space_time(model, 50)
    assert diagram.shape == (50, 100)
    assert np.all((diagram >= 0).sum(axis=1) == model.n_cars)


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        NaSch(100, 5, p_slow=1.5)
    with pytest.raises(ValueError):
        SNFS(100, 3, r=-0.1)
