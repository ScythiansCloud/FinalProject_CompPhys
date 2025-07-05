# Project 4 – Simulating Colloidal Flocculation

*Due 18 July 2025*

Tick a box once the task is finished.

## 0 General Setup

* [x] Create Git repository & virtual environment
* [x] Install dependencies (`numpy`, `scipy`, `matplotlib`, etc.)
* [x] Design folder structure (`src/`, `data/`, `analysis/`, `figures/`, `docs/`)
* [x] Set up CI & unit‑test scaffold

## 1 Implementation (Code)

* [x] 3‑D periodic simulation box (L³, density ρ)
* [ ] DLVO pair potential (Eq. 2)
* [x] Langevin integrator (Algorithm 1, Gaussian noise)
* [ ] Trajectory output every `Nsave` steps
* [ ] CLI or config file for parameters
* [ ] Unit tests for force, energy, integrator

## 2 Validation (Ideal‑Gas MSD)

* [ ] Disable interactions
* [ ] Simulate N = 256, ρ = 0.5 σ⁻³, Δt = 10⁻³ τₗd, 200 000 steps
* [ ] Save frames every 10 Δt
* [ ] Compute MSD (Eq. 4)
* [ ] Plot MSD vs t (log‑log)
* [ ] Identify ballistic & diffusive regimes
* [ ] Extract D; compare with kBT/ξ
* [ ] Conclude on integrator correctness

## 3 Production Runs (DLVO)

Common parameters: N = 343, ρ = 0.05 σ⁻³, Δt = 10⁻² τₗd, total = 200 000 steps, save = 10 Δt
For each salt concentration:

* [ ] Initialise system, **Cs = 10 σ⁻³**
* [ ] Run to completion & store trajectory
* [ ] Initialise system, **Cs = 100 σ⁻³**
* [ ] Run to completion & store trajectory
* [ ] Initialise system, **Cs = 333 σ⁻³**
* [ ] Run to completion & store trajectory
* [ ] Initialise system, **Cs = 666 σ⁻³**
* [ ] Run to completion & store trajectory
* [ ] Initialise system, **Cs = 1000 σ⁻³**
* [ ] Run to completion & store trajectory

*(mark each salt once the run + trajectory backup is done)*

## 4 Analysis

### 4a Radial Distribution Function g(r)

* [ ] Compute g(r) for every Cs
* [ ] Plot g(r) curves
* [ ] Determine first‑peak height & coordination number
* [ ] Plot those metrics vs Cs
* [ ] Calculate cluster‑size distribution
* [ ] Track cluster‑size vs time & discuss

### 4b Structure Factor S(k)

* [ ] Compute S(k) from g(r) or FFT
* [ ] Plot S(k) for each Cs
* [ ] Estimate compressibility (k → 0)
* [ ] Plot compressibility vs Cs & interpret

### 4c (Opt.) Diffusion vs Salt

* [ ] Compute MSD at low Cs (10)
* [ ] Compare with free‑particle MSD
* [ ] Hypothesise high‑salt/high‑density behaviour

## 5 Real‑World Context

* [ ] Gather DLVO parameters from literature (A, Z)
* [ ] Summarise typical systems & length scales
* [ ] Distinguish Agglomeration/Coagulation/Flocculation
* [ ] List real‑world applications & examples

## 6 Report & Submission

* [ ] Generate all plots & save to `figures/`
* [ ] Write report (LaTeX) including methods, validation, results, discussion
* [ ] Proofread & finalise PDF
* [ ] Tag final git commit & push
* [ ] Submit code & report by **18 July 2025**

---

*All tasks derive from the project brief in the supplied PDF.*
