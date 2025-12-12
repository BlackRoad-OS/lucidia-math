"""Prime pattern exploration and visualization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from .frameworks import select_backend


@dataclass
class PrimeVisualizer:
    output_dir: Path

    def save_fig(self, fig: plt.Figure, name: str) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        png = self.output_dir / f"{name}.png"
        svg = self.output_dir / f"{name}.svg"
        fig.savefig(png)
        fig.savefig(svg)
        plt.close(fig)


def ulam_spiral(size: int, *, backend: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Generate an Ulam spiral and mask of prime numbers."""

    if backend is not None and backend != "numpy":
        # Prime detection relies on SymPy; use numpy for now but validate name.
        select_backend(backend)
    grid = np.zeros((size, size), dtype=int)
    x = y = size // 2
    dx, dy = 0, -1
    for n in range(1, size * size + 1):
        if -size // 2 <= x < size // 2 and -size // 2 <= y < size // 2:
            grid[y + size // 2, x + size // 2] = n
        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1 - y):
            dx, dy = -dy, dx
        x, y = x + dx, y + dy
    prime_mask = np.vectorize(sp.isprime)(grid).astype(bool)
    return grid, prime_mask


def plot_ulam(grid: np.ndarray, mask: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.imshow(mask, cmap="Greys")
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def residue_grid(mod: int, size: int = 100, *, backend: str | None = None) -> np.ndarray:
    """Compute a modular residue grid.

    Parameters
    ----------
    mod:
        The modulus used for the residue computation.
    size:
        Total number of integers to include. ``size`` must be a perfect
        square so that the numbers can be reshaped into a square grid.

    Raises
    ------
    ValueError
        If ``size`` is not a perfect square.
    """

    backend_cfg = select_backend(backend)
    xp = backend_cfg.array_module
    numbers = xp.arange(1, size + 1)
    side = int(np.sqrt(size))
    if side * side != size:
        raise ValueError("size must be a perfect square")
    grid = xp.reshape(numbers, (side, side)) % mod
    return np.asarray(grid)


def plot_residue(grid: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap="viridis")
    ax.set_xticks([])
    ax.set_yticks([])
    return fig


def fourier_prime_gaps(limit: int, *, backend: str | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return prime gaps and their Fourier transform magnitude."""

    primes = list(sp.primerange(2, limit))
    backend_cfg = select_backend(backend)
    xp = backend_cfg.array_module
    gaps = xp.diff(xp.asarray(primes, dtype=float))
    fft = xp.abs(xp.fft.fft(gaps))
    return np.asarray(gaps, dtype=float), np.asarray(fft, dtype=float)


def plot_fourier(gaps: np.ndarray, fft: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(2, 1, figsize=(6, 6))
    ax[0].plot(gaps)
    ax[0].set_title("Prime gaps")
    ax[1].plot(fft)
    ax[1].set_title("FFT magnitude")
    return fig


def residue_class_counts(
    mod: int, limit: int, *, backend: str | None = None
) -> np.ndarray:
    """Count primes that land in each residue class modulo ``mod``.

    Parameters
    ----------
    mod:
        The modulus for the residue classes.
    limit:
        Upper bound (exclusive) for prime generation.
    backend:
        Optional array backend name. Defaults to the first available backend.

    Returns
    -------
    np.ndarray
        One-dimensional array with counts for each residue class ``0..mod-1``.
    """

    backend_cfg = select_backend(backend)
    xp = backend_cfg.array_module
    counts = xp.zeros(mod, dtype=int)
    for prime in sp.primerange(2, limit):
        counts[int(prime % mod)] += 1
    return np.asarray(counts)


def plot_residue_class_counts(counts: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    classes = np.arange(len(counts))
    ax.bar(classes, counts, color="slateblue")
    ax.set_xlabel("Residue class")
    ax.set_ylabel("Prime count")
    ax.set_title("Prime residues modulo n")
    return fig


def prime_density_profile(
    limit: int, window: int = 1000, *, backend: str | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute empirical and predicted prime densities across intervals.

    The function walks the integer line in windows and reports the observed
    density of primes along with the prime number theorem approximation
    using the logarithmic integral :math:`Li(x)`.

    Parameters
    ----------
    limit:
        Upper bound (exclusive) for sampling.
    window:
        Width of each sampling window. The final window is clipped to ``limit``.
    backend:
        Optional array backend name. Defaults to the first available backend.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        ``x`` sample positions, empirical densities, and predicted densities.
    """

    backend_cfg = select_backend(backend)
    xp = backend_cfg.array_module
    starts = xp.arange(2, limit, window)
    empirical: list[float] = []
    predicted: list[float] = []

    for start in starts:
        end = min(int(start + window), limit)
        primes = list(sp.primerange(int(start), end))
        count = len(primes)
        empirical.append(count / (end - int(start)))
        predicted.append(float(sp.li(end) - sp.li(int(start))) / (end - int(start)))

    centers = np.asarray(starts + window / 2.0, dtype=float)
    return np.asarray(centers), np.asarray(empirical), np.asarray(predicted)


def plot_density_profile(
    x: np.ndarray, empirical: np.ndarray, predicted: np.ndarray
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, empirical, label="empirical", marker="o", linestyle="-")
    ax.plot(x, predicted, label="PNT estimate", linestyle="--")
    ax.set_xlabel("Sample center")
    ax.set_ylabel("Prime density")
    ax.set_title("Prime density vs. logarithmic integral estimate")
    ax.legend()
    return fig
