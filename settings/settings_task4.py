from __future__ import annotations

# Histogram bin width for g(r)
dr: float = 0.02          # in simulation length units (≈ σ)

# Structure‑factor resolution
kmax: float = 20.0        # in multiples of first reciprocal‑lattice vector 2π/L
nk:   int   = 200         # number of k‑points between 0 and kmax

# Snapshot stride originally used when writing trajectories – required to
# convert “saved‑snapshot index” → real MD time in the MSD plot.
everyN: int = 10
