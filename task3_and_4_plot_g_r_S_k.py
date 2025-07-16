from __future__ import annotations
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
    return settings_10 if int(round(Cs)) == 10 else settings_std


def load_gr_file(data_dir: Path, Cs: float) -> np.ndarray:
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
# plotting: g(r)
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
# S(k)
# ----------------------------------------------------------------------------

def compute_sk(gr, settings_by_cs):
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
    ax.set(xlabel="k", ylabel="S(k)")
    ax.legend()
    fig.savefig(out_dir / "S_k_all.png", dpi=300)
    plt.close(fig)

# ----------------------------------------------------------------------------
# global g(r) maxima vs Cs
# ----------------------------------------------------------------------------

def compute_gr_global_maxima(gr, rr):
    Cs_vals = []
    r_max_vals = []
    g_max_vals = []
    for Cs, g in gr.items():
        i = int(np.argmax(g))
        Cs_vals.append(Cs)
        r_max_vals.append(rr[Cs][i])
        g_max_vals.append(g[i])
    o = np.argsort(Cs_vals)
    return np.asarray(Cs_vals)[o], np.asarray(r_max_vals)[o], np.asarray(g_max_vals)[o]


def plot_gr_global_maxima(Cs_vals, r_max_vals, g_max_vals, out_dir: Path):
    # --- combined 2-panel figure ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.plot(Cs_vals, r_max_vals, marker='o')
    ax.set(xlabel="Cs", ylabel="r_max", title="Peak position")
    ax = axes[1]
    ax.plot(Cs_vals, g_max_vals, marker='o')
    ax.set(xlabel="Cs", ylabel="g_max", title="Peak height")
    fig.tight_layout()
    fig.savefig(out_dir / "g_r_peaks_vs_Cs.png", dpi=300)
    plt.close(fig)

    # --- keep the original separate figures (unchanged behavior) ---
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(Cs_vals, r_max_vals, marker='o')
    ax.set(xlabel="Cs", ylabel="r_max")
    fig.savefig(out_dir / "g_r_peak_r_vs_Cs.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(Cs_vals, g_max_vals, marker='o')
    ax.set(xlabel="Cs", ylabel="g_max")
    fig.savefig(out_dir / "g_r_peak_g_vs_Cs.png", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------
# coordination number
# ----------------------------------------------------------------------------

def _first_shell_cutoff_r(r: np.ndarray, g: np.ndarray, *, min_start: int = 1) -> float:
    g = np.asarray(g)
    r = np.asarray(r)
    if len(g) < 3:
        return float(r[-1])
    peak_idx = int(np.argmax(g))
    start = min(len(g) - 2, max(peak_idx + min_start, 1))
    for i in range(start, len(g) - 1):
        if g[i] <= g[i - 1] and g[i] <= g[i + 1]:
            return float(r[i])
    if start < len(g):
        j_rel = int(np.argmin(g[start:]))
        return float(r[start + j_rel])
    return float(r[-1])


def coordination_number_first_shell(g: np.ndarray, r: np.ndarray, rho: float) -> tuple[float, float]:
    r1 = _first_shell_cutoff_r(r, g)
    mask = r <= r1
    integrand = g[mask] * (r[mask]**2)
    n1 = 4.0 * np.pi * rho * np.trapz(integrand, r[mask])
    return r1, n1


def write_coordination_numbers_first_shell(gr: dict, rr: dict, settings_by_cs: dict,
                                           out_dir: Path,
                                           fname: str = "coordination_numbers.txt"):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / fname
    Cs_vals = []
    r1_vals = []
    n1_vals = []
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# n1 = 4*pi*rho * int_0^{r1} g(r) r^2 dr   (r1 = first min after main peak)\n")
        fh.write("# Cs   r1   n1\n")
        for Cs in sorted(gr):
            smod = settings_by_cs[Cs]
            rho = getattr(smod, "rho", None) or smod.N / (smod.L ** 3)
            r1, n1 = coordination_number_first_shell(gr[Cs], rr[Cs], rho)
            fh.write(f"{Cs:10.4g}  {r1:16.8e}  {n1:16.8e}\n")
            Cs_vals.append(Cs)
            r1_vals.append(r1)
            n1_vals.append(n1)
    print(f"[Task3+4] coordination numbers written: {path}")
    # plot n1 vs Cs
    Cs_arr = np.asarray(Cs_vals, dtype=float)
    n1_arr = np.asarray(n1_vals, dtype=float)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(Cs_arr, n1_arr, marker='o')
    ax.set(xlabel="Cs", ylabel="coordination number n1")
    fig.savefig(out_dir / "coordination_number_vs_Cs.png", dpi=300)
    plt.close(fig)
    return Cs_arr, np.asarray(r1_vals, dtype=float), n1_arr

# =============================================================================
# isothermal compressibility
# =============================================================================

def write_compressibilities(Sk: dict, settings_by_cs: dict, kk: dict, out_dir: Path, fname: str = "compressibilities.txt"):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / fname
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# isothermal compressibility kappa_T = S(0) / (rho * kB * T)\n")
        fh.write("# Cs   S0   rho   kBT   kappa_T\n")
        for Cs in sorted(Sk):
            smod = settings_by_cs[Cs]
            rho = getattr(smod, "rho", None) or smod.N / (smod.L ** 3)
            if hasattr(smod, "kBT"):
                kBT = float(smod.kBT)
            elif hasattr(smod, "kB") and hasattr(smod, "T"):
                kBT = float(smod.kB) * float(smod.T)
            else:
                raise AttributeError(f"No kBT (or kB,T) in settings for Cs={Cs}")
            S0 = float(Sk[Cs][0])
            kappa = S0 / (rho * kBT)
            fh.write(f"{Cs:10.4g}  {S0:16.8e}  {rho:16.8e}  {kBT:16.8e}  {kappa:16.8e}\n")
    print(f"[Task3+4] compressibilities written: {path}")

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

    # Global g(r) peak summary
    Cs_vals, r_max_vals, g_max_vals = compute_gr_global_maxima(gr, rr)
    plot_gr_global_maxima(Cs_vals, r_max_vals, g_max_vals, out_dir)

    # First-shell coordination numbers (scalar + plot)
    Cs_cn, r1_cn, n1_cn = write_coordination_numbers_first_shell(gr, rr, settings_by_cs, out_dir)

    # Isothermal compressibilities
    write_compressibilities(Sk, settings_by_cs, kk, out_dir)

    print(f"[Task3+4] Done. Results written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
