"""Графики и анимации для моделей транспортного потока."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap

from .experiments import FundamentalDiagram
from .models import RingRoad

# Скорость 0 — красный (стоит в пробке), максимальная — зелёный.
SPEED_COLORS = ["#d62728", "#ff7f0e", "#f2c12e", "#9acd32", "#3cb371", "#1b7a3e"]


def speed_cmap(v_max: int) -> ListedColormap:
    """Дискретная палитра на ``v_max + 1`` скоростей от красного к зелёному."""
    idx = np.linspace(0, len(SPEED_COLORS) - 1, v_max + 1).round().astype(int)
    return ListedColormap([SPEED_COLORS[i] for i in idx])


def plot_fundamental(
    diagrams: dict[str, FundamentalDiagram],
    title: str,
    path: Path,
) -> None:
    """Фундаментальные диаграммы: поток J(ρ) и средняя скорость v(ρ)."""
    fig, (ax_flow, ax_vel) = plt.subplots(1, 2, figsize=(13, 4.8))
    for name, fd in diagrams.items():
        ax_flow.plot(fd.densities, fd.flows, ".", ms=4, label=name)
        ax_vel.plot(fd.densities, fd.velocities, ".", ms=4, label=name)
    ax_flow.set(xlabel="Density ρ, cars/cell", ylabel="Flow J, cars/step", title="Flow")
    ax_vel.set(xlabel="Density ρ, cars/cell", ylabel="Mean velocity, cells/step", title="Velocity")
    for ax in (ax_flow, ax_vel):
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path)


def plot_space_time(diagrams: dict[str, np.ndarray], v_max: int, path: Path) -> None:
    """x–t диаграммы: время идёт вниз, машины едут вправо.

    Пробки видны как красные полосы, которые сползают *влево* — волна
    торможения распространяется против движения машин.
    """
    fig, axes = plt.subplots(1, len(diagrams), figsize=(7 * len(diagrams), 7), squeeze=False)
    cmap = speed_cmap(v_max)
    cmap.set_bad("white")  # пустые клетки
    for ax, (name, diagram) in zip(axes[0], diagrams.items(), strict=True):
        masked = np.ma.masked_less(diagram, 0)
        im = ax.imshow(
            masked, cmap=cmap, vmin=-0.5, vmax=v_max + 0.5,
            aspect="auto", interpolation="nearest",
        )
        ax.set(xlabel="Position, cell", ylabel="Time, step", title=name)
    cbar = fig.colorbar(im, ax=axes[0].tolist(), ticks=range(v_max + 1), fraction=0.03)
    cbar.set_label("Velocity, cells/step")
    _save(fig, path)


def plot_velocity_histograms(hists: dict[str, np.ndarray], path: Path) -> None:
    """Распределение машин по скоростям для нескольких сценариев."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / len(hists)
    for k, (name, hist) in enumerate(hists.items()):
        speeds = np.arange(hist.size)
        ax.bar(speeds + (k - (len(hists) - 1) / 2) * width, hist, width, label=name)
    ax.set(xlabel="Velocity, cells/step", ylabel="Share of cars", title="Velocity distribution")
    ax.set_xticks(np.arange(next(iter(hists.values())).size))
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    _save(fig, path)


def save_ring_animation(model: RingRoad, frames: int, path: Path, fps: int = 12) -> None:
    """Анимация машин на кольцевой дороге (вид сверху).

    Каждая машина — точка на окружности, цвет — скорость. Хорошо видно, как
    пробка возникает «из ничего» и ползёт назад, против движения.
    """
    angle = 2 * np.pi * np.arange(model.length) / model.length
    cmap = speed_cmap(model.v_max)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.set_aspect("equal")
    ax.axis("off")
    # Сама дорога — серое кольцо.
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="#bbbbbb", lw=10, zorder=0))
    dots = ax.scatter([], [], s=18, zorder=1)
    label = ax.text(0, 0, "", ha="center", va="center", fontsize=11)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)

    def update(_frame: int):
        model.step()
        # Машины едут против часовой стрелки, начиная сверху.
        phi = np.pi / 2 + angle[model.positions]
        dots.set_offsets(np.column_stack([np.cos(phi), np.sin(phi)]))
        dots.set_color(cmap(model.velocities / model.v_max))
        label.set_text(f"t = {model.time}\n⟨v⟩ = {model.mean_velocity():.2f}")
        return dots, label

    animation = FuncAnimation(fig, update, frames=frames, blit=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)


def _save(fig: plt.Figure, path: Path) -> None:
    """Сохранить фигуру и закрыть её."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
