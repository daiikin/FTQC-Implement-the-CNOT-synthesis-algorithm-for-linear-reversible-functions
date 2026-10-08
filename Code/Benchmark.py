"""
Benchmark reproducing Figure 4 of Patel, Markov & Hayes (quant-ph/0302002).

For each wire count n we synthesize `trials` random non-singular GF(2) matrices
with both the Patel-Markov-Hayes algorithm (Algorithm 1) and standard Gaussian
elimination, then plot the average circuit length (gate count) versus n.  This
mirrors the paper's experiment, including their heuristic m = round((log2 n)/2).
"""

from __future__ import annotations

import time
import numpy as np
import matplotlib
import os

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cnot_synth import (
    random_invertible_gf2,
    gaussian_synth,
    patel_markov_synth,
    verify,
)


def run_benchmark(
    n_values, trials: int = 100, seed: int = 12345, verify_each: bool = True
):
    rng = np.random.default_rng(seed)

    avg_gauss, avg_pmh = [], []
    avg_t_gauss, avg_t_pmh = [], []

    for n in n_values:
        g_lengths, p_lengths = [], []
        g_times, p_times = [], []

        for _ in range(trials):
            A = random_invertible_gf2(n, rng)

            t0 = time.perf_counter()
            gg = gaussian_synth(A)
            t1 = time.perf_counter()
            gp = patel_markov_synth(A)  # m defaults to round((log2 n)/2)
            t2 = time.perf_counter()

            if verify_each:
                assert verify(A, gg), f"Gaussian verify failed at n={n}"
                assert verify(A, gp), f"Patel-Markov verify failed at n={n}"

            g_lengths.append(len(gg))
            p_lengths.append(len(gp))
            g_times.append(t1 - t0)
            p_times.append(t2 - t1)

        avg_gauss.append(np.mean(g_lengths))
        avg_pmh.append(np.mean(p_lengths))
        avg_t_gauss.append(np.mean(g_times))
        avg_t_pmh.append(np.mean(p_times))

        print(
            f"n={n:3d}  Gaussian avg={avg_gauss[-1]:8.1f}  "
            f"PMH avg={avg_pmh[-1]:8.1f}  "
            f"improvement={100*(1-avg_pmh[-1]/avg_gauss[-1]):5.1f}%"
        )

    return {
        "n": list(n_values),
        "gauss_len": avg_gauss,
        "pmh_len": avg_pmh,
        "gauss_time": avg_t_gauss,
        "pmh_time": avg_t_pmh,
    }


def plot_results(res, outdir=None):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))
    n = res["n"]

    # --- Figure 1: full circuit length (the Figure 4 reproduction) ---
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(n, res["pmh_len"], "o-", color="#1f77b4", ms=4,
             label="Patel-Markov-Hayes")
    ax1.plot(n, res["gauss_len"], "s-", color="#d62728", ms=4,
             label="Gaussian elimination")
    ax1.set_xlabel("wires (n)")
    ax1.set_ylabel("circuit length (avg gates)")
    ax1.set_title("Average circuit length vs. n")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    path1 = os.path.join(outdir, "benchmark_length.png")
    fig1.savefig(path1, dpi=130)
    print(f"Saved plot to {path1}")

    # --- Figure 2: crossover region zoom ---
    small = [i for i, v in enumerate(n) if v <= 24]
    ns = [n[i] for i in small]
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(ns, [res["pmh_len"][i] for i in small], "o-", color="#1f77b4",
             ms=4, label="Patel-Markov-Hayes")
    ax2.plot(ns, [res["gauss_len"][i] for i in small], "s-", color="#d62728",
             ms=4, label="Gaussian elimination")
    ax2.set_xlabel("wires (n)")
    ax2.set_ylabel("circuit length (avg gates)")
    ax2.set_title("Crossover region (small n)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    path2 = os.path.join(outdir, "benchmark_crossover.png")
    fig2.savefig(path2, dpi=130)
    print(f"Saved plot to {path2}")


def plot_ratio(res, outdir=None):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(__file__))
    n = np.array(res["n"], dtype=float)
    ratio = np.array(res["gauss_len"]) / np.array(res["pmh_len"])

    # Fit only where log n is meaningful and PMH actually differs from
    # Gaussian (the two coincide at very small n, where m collapses to 1).
    mask = n >= 8
    nf, rf = n[mask], ratio[mask]

    # Model: ratio ~ a * log2(n) + b  (b absorbs PMH's lower-order terms)
    logn = np.log2(nf)
    a, b = np.polyfit(logn, rf, 1)
    fit = a * np.log2(n) + b

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(n, ratio, "o", color="#2ca02c", ms=5,
            label="Gaussian / PMH ratio")
    ax.plot(n, fit, "-", color="#9467bd", lw=2,
            label=fr"Logarithmic fit")
    ax.axhline(1.0, color="grey", ls=":", lw=1)
    ax.set_xlabel("wires (n)")
    ax.set_ylabel("gate-count ratio")
    ax.set_title("Gaussian / PMH gate-count ratio vs. logarithmic fit")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "benchmark_ratio.png")
    fig.savefig(path, dpi=130)
    print(f"Saved plot to {path}")
    print(f"Log fit: ratio ≈ {a:.3f} * log2(n) + {b:.3f}")


if __name__ == "__main__":
    n_values = list(range(2, 81, 2))
    res = run_benchmark(n_values, trials=100)
    plot_results(res)
    plot_ratio(res)