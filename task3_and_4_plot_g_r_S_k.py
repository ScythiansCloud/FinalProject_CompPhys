from __future__ import annotations
"""
Task3+4: Plot g(r), compute S(k), plot S(k), Plot global g(r) peaks vs Cs, Coordination number curves
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# settings modules -----------------------------------------------------------
import settings.settings_task3 as settings_std
import settings.settings_task3_10 as settings_10
from utilities.update import compute_S_of_k_from_gr
from utilities.utils import create_output_directory

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
DATA_DIR = Path("g_r")      # directory with g_r_Cs*.txt
CS_LIST  = [10, 100, 333, 666, 1000]
K_POINTS = 400
K_MAX_MULT = 20.0            # k_max = K_MAX_MULT * k_min

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def settings_for_cs(Cs: float):
    """Return the correct settings module for this Cs."""
    return settings_10 if int(round(Cs)) == 10 else settings_std


def load_gr_file(data_dir: Path, Cs: float) -> np.ndarray:
    """Load one-column g(r) file for the given Cs."""
    path = data_dir / f"g_r_Cs{int(round(Cs))}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing g(r) file: {path}")
    g = np.loadtxt(path, dtype=float)
    return np.asarray(g, dtype=float).ravel()


def r_from_dr(n_bins: int, dr: float) -> np.ndarray:
    return (np.arange(n_bins) + 0.5) * dr


def k_grid(L: float, k_max_mult: float = K_MAX_MULT, k_points: int = K_POINTS) -> np.ndarray:
    k_min = 2 * np.pi / (L / 2.0)
    return np.concatenate(([0.0], np.linspace(k_min, k_max_mult * k_min, k_points)))

# ----------------------------------------------------------------------------
# load all g(r) + associated settings
# ----------------------------------------------------------------------------

def load_all(data_dir: Path, cs_list):
    gr = {}
    rr = {}
    settings_by_cs = {}
    for Cs in cs_list:
        smod = settings_for_cs(Cs)
        smod.init(Cs)
        settings_by_cs[Cs] = smod
        g = load_gr_file(data_dir, Cs)
        r = r_from_dr(len(g), smod.dr)
        gr[Cs] = g
        rr[Cs] = r
    return gr, rr, settings_by_cs

# ----------------------------------------------------------------------------
# plotting helpers
# ----------------------------------------------------------------------------

def plot_gr_individual(gr, rr, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for Cs, g in gr.items():
        r = rr[Cs]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(r, g)
        ax.set(xlabel="r / sigma", ylabel="g(r)", title=f"g(r) – Cs={Cs}")
        fig.savefig(out_dir / f"g_r_Cs{int(Cs)}.png", dpi=300)
        plt.close(fig)


def plot_gr_combined(gr, rr, out_dir: Path):
    if not gr:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for Cs in sorted(gr):
        ax.plot(rr[Cs], gr[Cs], label=f"Cs={Cs}")
    ax.set(xlabel="r / sigma", ylabel="g(r)")
    ax.legend()
    fig.savefig(out_dir / "g_r_all.png", dpi=300)
    plt.close(fig)

# ----------------------------------------------------------------------------
# S(k) computation (with warning suppression)
# ----------------------------------------------------------------------------

def compute_sk(gr, settings_by_cs):
    """Compute S(k) arrays for all Cs; return dicts Sk, kk."""
    Sk = {}
    kk = {}
    for Cs, g in gr.items():
        smod = settings_by_cs[Cs]
        dr = smod.dr
        L = smod.L
        rho = getattr(smod, "rho", None) or smod.N / (L ** 3)
        k = k_grid(L)
        with np.errstate(divide='ignore', invalid='ignore'):
            S_k = compute_S_of_k_from_gr(g, dr, rho, k)
        Sk[Cs] = S_k
        kk[Cs] = k
    return Sk, kk


def plot_sk_individual(Sk, kk, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for Cs, S_k in Sk.items():
        k = kk[Cs]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(k[1:], S_k[1:])  # skip k=0 for scale
        ax.set(xlabel="k", ylabel="S(k)", title=f"S(k) – Cs={Cs}")
        fig.savefig(out_dir / f"S_k_Cs{int(Cs)}.png", dpi=300)
        plt.close(fig)


def plot_sk_combined(Sk, kk, out_dir: Path, *, atol=1e-10, rtol=1e-7):
    if not Sk:
        return
    k0 = next(iter(kk.values()))
    if not all(np.allclose(k0, k_i, atol=atol, rtol=rtol) for k_i in list(kk.values())[1:]):
        print("[Task3+4] Info: skipping combined S(k) plot (k grids differ).")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for Cs in sorted(Sk):
        ax.plot(kk[Cs][1:], Sk[Cs][1:], label=f"Cs={Cs}")
        # if you want to include k=0 in the plot, remove [1:]
    ax.set(xlabel="k", ylabel="S(k)")
    ax.legend()
    fig.savefig(out_dir / "S_k_all.png", dpi=300)
    plt.close(fig)

# =============================================================================
# === Functionality 1: global g(r) maxima vs Cs ==============================
# =============================================================================

def compute_gr_global_maxima(gr, rr):
    Cs_vals = []
    r_max_vals = []
    g_max_vals = []
    for Cs, g in gr.items():
        idx = int(np.argmax(g))
        Cs_vals.append(Cs)
        r_max_vals.append(rr[Cs][idx])
        g_max_vals.append(g[idx])
    order = np.argsort(Cs_vals)
    Cs_vals = np.asarray(Cs_vals)[order]
    r_max_vals = np.asarray(r_max_vals)[order]
    g_max_vals = np.asarray(g_max_vals)[order]
    return Cs_vals, r_max_vals, g_max_vals


def plot_gr_global_maxima(Cs_vals, r_max_vals, g_max_vals, out_dir: Path):
    # r_peak vs Cs
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(Cs_vals, r_max_vals, marker='o')
    ax.set(xlabel="Cs", ylabel="r_max", title="g(r) peak position vs Cs")
    fig.savefig(out_dir / "g_r_peak_r_vs_Cs.png", dpi=300)
    plt.close(fig)
    # g_peak vs Cs
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(Cs_vals, g_max_vals, marker='o')
    ax.set(xlabel="Cs", ylabel="g_max", title="g(r) peak height vs Cs")
    fig.savefig(out_dir / "g_r_peak_g_vs_Cs.png", dpi=300)
    plt.close(fig)

# =============================================================================
# === Functionality 2: coordination number curves ============================
# =============================================================================

def coordination_number_curve(g, r, rho):
    dr = np.gradient(r)
    integrand = g * (r**2) * dr
    cn = 4.0 * np.pi * rho * np.cumsum(integrand)
    return cn


def compute_all_coordination_numbers(gr, rr, settings_by_cs):
    cn = {}
    for Cs, g in gr.items():
        smod = settings_by_cs[Cs]
        rho = getattr(smod, "rho", None) or smod.N / (smod.L ** 3)
        cn[Cs] = coordination_number_curve(g, rr[Cs], rho)
    return cn


def plot_cn_individual(cn, rr, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for Cs, cn_arr in cn.items():
        r = rr[Cs]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(r, cn_arr)
        ax.set(xlabel="r / sigma", ylabel="coordination number", title=f"CN(r) – Cs={Cs}")
        fig.savefig(out_dir / f"CN_Cs{int(Cs)}.png", dpi=300)
        plt.close(fig)


def plot_cn_combined(cn, rr, out_dir: Path):
    if not cn:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for Cs in sorted(cn):
        ax.plot(rr[Cs], cn[Cs], label=f"Cs={Cs}")
    ax.set(xlabel="r / sigma", ylabel="coordination number")
    ax.legend()
    fig.savefig(out_dir / "CN_all.png", dpi=300)
    plt.close(fig)

# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    data_dir = DATA_DIR
    if not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")
    out_dir = create_output_directory()

    print(f"[Task3+4] Reading g(r) for Cs values: {CS_LIST}")
    gr, rr, settings_by_cs = load_all(data_dir, CS_LIST)

    print("[Task3+4] Plotting g(r)...")
    plot_gr_individual(gr, rr, out_dir)
    plot_gr_combined(gr, rr, out_dir)

    print("[Task3+4] Computing S(k)...")
    Sk, kk = compute_sk(gr, settings_by_cs)
    plot_sk_individual(Sk, kk, out_dir)
    plot_sk_combined(Sk, kk, out_dir)

    # === Functionality 1 ===
    Cs_vals, r_max_vals, g_max_vals = compute_gr_global_maxima(gr, rr)
    plot_gr_global_maxima(Cs_vals, r_max_vals, g_max_vals, out_dir)

    # === Functionality 2 ===
    cn = compute_all_coordination_numbers(gr, rr, settings_by_cs)
    plot_cn_individual(cn, rr, out_dir)
    plot_cn_combined(cn, rr, out_dir)

    print(f"[Task3+4] Done. Plots written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
