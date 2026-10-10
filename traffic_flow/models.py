"""Клеточные автоматы транспортного потока на кольцевой дороге.

Дорога — кольцо из ``L`` клеток, в каждой клетке не больше одной машины.
Скорость машины — целое число клеток за шаг от 0 до ``v_max``. Машины не
обгоняют друг друга, поэтому их порядок на кольце не меняется: лидер машины
``i`` — всегда машина ``i + 1`` (по модулю ``N``). Это позволяет хранить
координаты в фиксированном порядке и обновлять все машины сразу, векторно.

Реализованы две модели:

* :class:`NaSch` — модель Нагеля–Шрекенберга (1992) со смешанным потоком:
  часть машин ведут люди (случайное торможение), остальные — автопилот
  (детерминированное движение);
* :class:`SNFS` — модель S-NFS, стохастическое обобщение модели
  Нисинари–Фукуи–Шадшнайдера (Sakai, Nishinari, Iida, 2006) с эффектами
  медленного старта и упреждения.
"""

from __future__ import annotations

import numpy as np


class RingRoad:
    """Общая часть моделей: машины на кольцевой дороге.

    Параметры
    ---------
    length:
        Длина кольца в клетках.
    v_max:
        Максимальная скорость (клеток за шаг).
    rng:
        Генератор случайных чисел (для воспроизводимости — с фиксированным seed).
    """

    def __init__(self, length: int, v_max: int, rng: np.random.Generator | None = None) -> None:
        if length < 1 or v_max < 1:
            raise ValueError("Длина дороги и максимальная скорость должны быть положительными")
        self.length = length
        self.v_max = v_max
        self.rng = rng if rng is not None else np.random.default_rng()
        self.positions = np.zeros(0, dtype=np.int64)
        self.velocities = np.zeros(0, dtype=np.int64)
        self.time = 0

    @property
    def n_cars(self) -> int:
        return self.positions.size

    @property
    def density(self) -> float:
        """Плотность — доля занятых клеток."""
        return self.n_cars / self.length

    def place_cars(self, density: float) -> None:
        """Расставить машины случайно с заданной плотностью.

        Число машин округляется, но на дороге всегда есть хотя бы одна машина.
        Начальная скорость — максимальная: первый же шаг затормозит тех,
        кому не хватает дистанции.
        """
        n = min(self.length, max(1, round(density * self.length)))
        cells = self.rng.choice(self.length, size=n, replace=False)
        self.positions = np.sort(cells).astype(np.int64)
        self.velocities = np.full(n, self.v_max, dtype=np.int64)
        self.time = 0

    def gaps(self, ahead: int = 1) -> np.ndarray:
        """Число клеток между машиной и ``ahead``-й машиной впереди за вычетом
        клеток, занятых промежуточными машинами: ``x_{i+S} − x_i − S``.

        При ``ahead = 1`` — обычная дистанция до лидера. Остаток от деления
        корректно обрабатывает «перешагивание» через начало кольца.
        """
        leader = np.roll(self.positions, -ahead)
        return (leader - self.positions - ahead) % self.length

    def move(self, velocities: np.ndarray) -> None:
        """Сдвинуть все машины на их скорости."""
        self.velocities = velocities
        self.positions = (self.positions + velocities) % self.length
        self.time += 1

    def mean_velocity(self) -> float:
        return float(self.velocities.mean())

    def flow(self) -> float:
        """Поток J = ρ·⟨v⟩ — машин через сечение дороги за шаг."""
        return self.density * self.mean_velocity()

    def step(self) -> None:  # pragma: no cover - реализуется в наследниках
        raise NotImplementedError


class NaSch(RingRoad):
    """Модель Нагеля–Шрекенберга со смешанным потоком.

    Правила (применяются ко всем машинам одновременно):

    1. **Ускорение:** ``v ← min(v + 1, v_max)``.
    2. **Торможение:** ``v ← min(v, gap)`` — не въехать в лидера.
    3. **Случайное торможение:** ``v ← max(v − 1, 0)`` с вероятностью
       ``p_slow``, *только для машин под управлением человека*.
    4. **Движение:** ``x ← x + v``.

    Параметры
    ---------
    p_slow:
        Вероятность случайного торможения водителя-человека.
    human_ratio:
        Доля машин, которыми управляют люди (RHDV — ratio of human-driven
        vehicles). Остальные — автономные: без случайного торможения.
        При ``human_ratio = 1`` получается классическая модель NaSch, при
        ``human_ratio = 0`` — детерминированная модель (правило 184 при
        ``v_max = 1``).
    """

    def __init__(
        self,
        length: int,
        v_max: int = 5,
        p_slow: float = 0.5,
        human_ratio: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(length, v_max, rng)
        if not (0.0 <= p_slow <= 1.0 and 0.0 <= human_ratio <= 1.0):
            raise ValueError("Вероятности должны лежать в отрезке [0, 1]")
        self.p_slow = p_slow
        self.human_ratio = human_ratio
        self.is_human = np.zeros(0, dtype=bool)

    def place_cars(self, density: float) -> None:
        super().place_cars(density)
        # Тип водителя назначается каждой машине один раз и не меняется.
        self.is_human = self.rng.random(self.n_cars) < self.human_ratio

    def step(self) -> None:
        v = np.minimum(self.velocities + 1, self.v_max)          # 1. ускорение
        v = np.minimum(v, self.gaps())                           # 2. торможение
        dawdle = self.is_human & (self.rng.random(self.n_cars) < self.p_slow)
        v = np.where(dawdle, np.maximum(v - 1, 0), v)            # 3. случайное торможение
        self.move(v)                                             # 4. движение


class SNFS(RingRoad):
    """Стохастическая модель S-NFS (Sakai, Nishinari, Iida, 2006).

    Правила для каждой машины ``i`` (``x`` — координаты в текущий момент,
    ``x'`` — координаты на предыдущем шаге):

    1. **Ускорение:** ``v ← min(v + 1, v_max)``.
    2. **Медленный старт** (с вероятностью ``q``):
       ``v ← min(v, x'_{i+1} − x'_i − 1)`` — водитель реагирует на дистанцию
       *прошлого* шага, то есть с запаздыванием трогается с места.
    3. **Упреждение** (quick start): с вероятностью ``r`` водитель смотрит на
       ``S = 2`` машины вперёд, иначе ``S = 1``:
       ``v ← min(v, x_{i+S} − x_i − S)``. Он рассчитывает, что машина впереди
       тоже поедет, и держит меньшую дистанцию.
    4. **Случайное торможение** с вероятностью ``p_brake``: ``v ← max(v − 1, 0)``.
       В статье Сакаи и др. эта вероятность записана как ``1 − p``.
    5. **Предотвращение столкновений:**
       ``v_i ← min(v_i, x_{i+1} − x_i − 1 + v_{i+1})``, где ``v_{i+1}`` — уже
       вычисленная по правилам 1–4 скорость лидера.
    6. **Движение:** ``x ← x + v``.
    """

    def __init__(
        self,
        length: int,
        v_max: int = 3,
        q: float = 1.0,
        r: float = 0.5,
        p_brake: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(length, v_max, rng)
        if not all(0.0 <= value <= 1.0 for value in (q, r, p_brake)):
            raise ValueError("Вероятности должны лежать в отрезке [0, 1]")
        self.q = q
        self.r = r
        self.p_brake = p_brake
        self.prev_positions = np.zeros(0, dtype=np.int64)

    def place_cars(self, density: float) -> None:
        super().place_cars(density)
        self.prev_positions = self.positions.copy()

    def _prev_gaps(self) -> np.ndarray:
        """Дистанция до лидера на предыдущем шаге (для медленного старта)."""
        leader = np.roll(self.prev_positions, -1)
        return (leader - self.prev_positions - 1) % self.length

    def step(self) -> None:
        n = self.n_cars
        rand = self.rng.random((3, n))
        gap1 = self.gaps(1)

        v = np.minimum(self.velocities + 1, self.v_max)                  # 1. ускорение
        slow_start = rand[0] < self.q
        v = np.where(slow_start, np.minimum(v, self._prev_gaps()), v)    # 2. медленный старт
        look_two = rand[1] < self.r
        v = np.where(look_two, np.minimum(v, self.gaps(2)), np.minimum(v, gap1))  # 3. упреждение
        brake = rand[2] < self.p_brake
        v = np.where(brake, np.maximum(v - 1, 0), v)                     # 4. случайное торможение
        # 5. Без столкновений. Если лидер сам притормозил на этом правиле,
        # ограничение для его последователя тоже меняется, поэтому применяем
        # правило до неподвижной точки (скорости только уменьшаются, так что
        # цикл всегда завершается, обычно за 1–2 итерации).
        while True:
            limited = np.minimum(v, gap1 + np.roll(v, -1))
            if np.array_equal(limited, v):
                break
            v = limited

        self.prev_positions = self.positions.copy()
        self.move(v)                                                     # 6. движение
