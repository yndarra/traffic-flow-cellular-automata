"""Численные эксперименты: фундаментальные и пространственно-временные диаграммы."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .models import RingRoad

# Фабрика модели: по генератору случайных чисел возвращает новую модель.
# Так одна и та же функция эксперимента работает и с NaSch, и с S-NFS.
ModelFactory = Callable[[np.random.Generator], RingRoad]


@dataclass
class FundamentalDiagram:
    """Зависимость потока и средней скорости от плотности."""

    densities: np.ndarray
    flows: np.ndarray
    velocities: np.ndarray


def fundamental_diagram(
    make_model: ModelFactory,
    densities: np.ndarray,
    warmup: int = 1000,
    measure: int = 500,
    seed: int | None = 0,
    repeats: int = 1,
) -> FundamentalDiagram:
    """Построить фундаментальную диаграмму J(ρ).

    Для каждой плотности машины расставляются случайно, затем ``warmup`` шагов
    система выходит на установившийся режим, и ещё ``measure`` шагов поток
    усредняется по времени. Если ``repeats > 1``, результат дополнительно
    усредняется по нескольким независимым начальным расстановкам — это важно
    для смешанного потока, где одна медленная машина может «задать темп»
    всей колонне за ней.
    """
    rng = np.random.default_rng(seed)
    flows = np.zeros(len(densities))
    velocities = np.zeros(len(densities))
    for k, density in enumerate(densities):
        for _ in range(repeats):
            model = make_model(rng)
            model.place_cars(density)
            for _ in range(warmup):
                model.step()
            v_sum = 0.0
            for _ in range(measure):
                model.step()
                v_sum += model.mean_velocity()
            velocities[k] += v_sum / measure / repeats
            flows[k] += model.density * v_sum / measure / repeats
    return FundamentalDiagram(np.asarray(densities), flows, velocities)


def space_time(model: RingRoad, steps: int) -> np.ndarray:
    """Пространственно-временная (x–t) диаграмма.

    Возвращает матрицу ``steps × length``: в клетке — скорость машины, которая
    занимает её в данный момент, или ``-1``, если клетка пустая.
    """
    diagram = np.full((steps, model.length), -1, dtype=np.int8)
    for t in range(steps):
        model.step()
        diagram[t, model.positions] = model.velocities
    return diagram


def velocity_histogram(diagram: np.ndarray, v_max: int) -> np.ndarray:
    """Доля машин с каждой скоростью 0..v_max по x–t диаграмме."""
    counts = np.bincount(diagram[diagram >= 0].ravel(), minlength=v_max + 1)
    return counts / counts.sum()
