"""Точка входа: запускает эксперименты и сохраняет графики в папку ``images/``.

Примеры::

    python main.py            # все эксперименты
    python main.py nasch      # фундаментальные диаграммы NaSch
    python main.py xt         # x–t диаграммы и распределение скоростей
    python main.py snfs       # фундаментальные диаграммы S-NFS
    python main.py ring       # анимация кольцевой дороги
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from traffic_flow import SNFS, NaSch
from traffic_flow import experiments as exp
from traffic_flow import visualization as vis

ROAD_LENGTH = 1000      # длина кольца, клеток
WARMUP, MEASURE = 2000, 500
DENSITIES = np.round(np.linspace(0.01, 1.0, 100), 3)

# Параметры смешанного потока NaSch.
NASCH_V_MAX = 5
NASCH_P_SLOW = 0.8
HUMAN_RATIOS = (0.0, 0.3, 0.7, 1.0)

# Параметры S-NFS.
SNFS_V_MAX = 3
SNFS_P_BRAKE = 0.1


def run_nasch(out: Path, seed: int) -> None:
    """Как доля водителей-людей влияет на пропускную способность дороги."""
    diagrams = {}
    for ratio in HUMAN_RATIOS:
        fd = exp.fundamental_diagram(
            lambda rng, ratio=ratio: NaSch(ROAD_LENGTH, NASCH_V_MAX, NASCH_P_SLOW, ratio, rng),
            DENSITIES, WARMUP, MEASURE, seed, repeats=3,
        )
        diagrams[f"RHDV = {ratio:g}"] = fd
        k = int(np.argmax(fd.flows))
        print(f"  RHDV = {ratio:<4g} J_max = {fd.flows[k]:.3f} при ρ = {fd.densities[k]:.2f}")
    vis.plot_fundamental(
        diagrams,
        f"NaSch, mixed traffic: v_max = {NASCH_V_MAX}, p_slow = {NASCH_P_SLOW}",
        out / "nasch_fundamental.png",
    )


def run_space_time(out: Path, seed: int) -> None:
    """x–t диаграммы и распределение скоростей при плотности 0.3."""
    density, steps = 0.3, 600
    diagrams, hists = {}, {}
    for ratio in (0.0, 0.7):
        model = NaSch(ROAD_LENGTH, NASCH_V_MAX, NASCH_P_SLOW, ratio, np.random.default_rng(seed))
        model.place_cars(density)
        for _ in range(WARMUP):
            model.step()
        name = f"RHDV = {ratio:g}, ρ = {density}"
        diagrams[name] = exp.space_time(model, steps)
        hists[name] = exp.velocity_histogram(diagrams[name], NASCH_V_MAX)
        mean_v = float(np.dot(np.arange(NASCH_V_MAX + 1), hists[name]))
        print(f"  {name}: средняя скорость {mean_v:.2f}, "
              f"стоят {hists[name][0] * 100:.1f}% машин")
    vis.plot_space_time(diagrams, NASCH_V_MAX, out / "nasch_space_time.png")
    vis.plot_velocity_histograms(hists, out / "nasch_velocities.png")


def run_snfs(out: Path, seed: int) -> None:
    """S-NFS: влияние медленного старта (q) и упреждения (r)."""
    diagrams = {}
    for q, r in ((1.0, 0.5), (1.0, 1.0), (0.0, 0.0), (0.0, 1.0)):
        fd = exp.fundamental_diagram(
            lambda rng, q=q, r=r: SNFS(ROAD_LENGTH, SNFS_V_MAX, q, r, SNFS_P_BRAKE, rng),
            DENSITIES, WARMUP, MEASURE, seed,
        )
        diagrams[f"q = {q:g}, r = {r:g}"] = fd
        k = int(np.argmax(fd.flows))
        print(f"  q = {q:g}, r = {r:<4g} J_max = {fd.flows[k]:.3f} при ρ = {fd.densities[k]:.2f}")
    vis.plot_fundamental(
        diagrams,
        f"S-NFS: v_max = {SNFS_V_MAX}, p_brake = {SNFS_P_BRAKE}",
        out / "snfs_fundamental.png",
    )


def run_ring(out: Path, seed: int) -> None:
    """Анимация: спонтанные пробки на кольце из 200 клеток."""
    model = NaSch(200, v_max=5, p_slow=0.3, human_ratio=1.0, rng=np.random.default_rng(seed))
    model.place_cars(0.2)
    for _ in range(100):
        model.step()
    vis.save_ring_animation(model, frames=150, path=out / "ring_road.gif")


def main() -> None:
    parser = argparse.ArgumentParser(description="Traffic-flow cellular automata (NaSch, S-NFS)")
    parser.add_argument(
        "experiment",
        nargs="?",
        default="all",
        choices=["all", "nasch", "xt", "snfs", "ring"],
        help="какой эксперимент запустить (по умолчанию все)",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed генератора случайных чисел")
    parser.add_argument("--out", type=Path, default=Path("images"), help="папка для картинок")
    args = parser.parse_args()

    tasks = {
        "nasch": ("NaSch: фундаментальные диаграммы", run_nasch),
        "xt": ("NaSch: x–t диаграммы", run_space_time),
        "snfs": ("S-NFS: фундаментальные диаграммы", run_snfs),
        "ring": ("Анимация кольцевой дороги", run_ring),
    }
    selected = tasks if args.experiment == "all" else {args.experiment: tasks[args.experiment]}
    for title, task in selected.values():
        started = time.perf_counter()
        print(f"{title}...")
        task(args.out, args.seed)
        print(f"  готово за {time.perf_counter() - started:.1f} с")
    print(f"Картинки сохранены в {args.out.resolve()}")


if __name__ == "__main__":
    main()
