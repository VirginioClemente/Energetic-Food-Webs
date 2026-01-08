# Energetic Food-Web Dynamics

This repository contains code to reconstruct energy fluxes in empirical food webs
and analyse the associated population dynamics and stability properties.

The code accompanies the paper:

> **G. V. Clemente et al.**  
> *Maximum entropy networks predict fluctuations and stability of food web energetics* 

The notebooks and scripts here implement the mass–balance energetic food-web
model and the dynamical analysis used in the paper, including the computation of
entropy, resilience, reactivity and fluctuation–response relations.

---

## Repository structure

- `Mass_Balance.py`  
  Core class to solve the **mass-balance equations** and reconstruct the energy
  flux matrix between trophic groups, given topology, biomasses, metabolic
  rates, efficiencies and self-limitation.

- `Dynamic_equations.py`  
  Functions to:
  - solve the **population dynamics** at equilibrium,
  - compute the **Jacobian matrix** of the system,
  - derive stability metrics such as **maximum eigenvalue**, **numerical
    abscissa**, non-normality (Frobenius–eigenvalue difference), and **total
    flux**.

- `Results.ipynb`  
  Jupyter notebook that reproduces the main results and figures presented in the
  paper, including:
  - entropy of the maximum-entropy ensemble of flux networks,
  - relationships between entropy and dynamical stability metrics,
  - comparison between intrinsic fluctuations and responses to drought
    perturbations,
  - control vs drought (C vs P) analyses.

(Additional data files / Excel sheets are expected in the same structure used in
the paper.)

---
