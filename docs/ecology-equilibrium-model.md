# Ecological Equilibrium Reference Model

## Purpose

This document defines a **reduced mathematical reference model** for the lake
ecology in `tuiquarium`.

The goal is not to replace the simulation's agent-based and genome-driven
mechanics. The goal is to provide a **macroscopic source of truth** for:

- stable and unstable ecological regimes
- nutrient-balanced parameter sets
- calibration targets for a "realistic thriving lake"
- deliberate alternative equilibria, such as clear-water, mixed, or
  phytoplankton-dominated states

This should replace blind parameter search with a model-based workflow:

1. define a target ecological regime mathematically
2. solve for a positive equilibrium
3. verify local stability via the Jacobian
4. map that solution back into simulation calibration and founder defaults

The reduced model is intended to describe the **mean-field ecology** of the
simulation. The full simulation remains stochastic, spatial, and evolutionary.

## Why This Makes Sense

`tuiquarium` already contains the ingredients of a tractable ecological model:

- Monod-style nutrient limitation for producers
- light attenuation by water and phytoplankton
- explicit dissolved and sediment nutrient pools
- detrital recycling
- allometric consumer maintenance
- crowding-limited consumer reproduction
- separate rooted macrophyte and phytoplankton pathways

Those are standard ingredients for a reduced lake model. The current simulation
implements them in a distributed, entity-level way. A reduced equation system
lets us reason about the same ecology at the biomass-pool level.

This gives two benefits:

1. **Scientific clarity**
   We can state what "stable ecology" means as equations and inequalities,
   instead of as a collection of ad hoc code constants.

2. **Calibration leverage**
   We can solve for equilibria directly and tune the simulation toward them,
   instead of exploring parameter space by repeated trial runs alone.

## Scope And Limits

This reference model should be treated as the source of truth for
**macro-ecology**, not for micro-behavior.

It should govern:

- target nutrient ranges
- target biomass ratios
- target coexistence regimes
- desired stable and unstable equilibria
- calibration defaults for founder webs and recovery behavior

It should not be treated as the exact source of truth for:

- individual movement
- local predation events
- species morphology
- genome-level trait distributions
- short-term stochastic transients

The correct interpretation is:

- the reduced model defines the ecological envelope
- the full simulation should fluctuate around that envelope

## State Variables

Use the following reduced state vector:

- $M(t)$: rooted macrophyte biomass
- $A(t)$: phytoplankton biomass
- $D(t)$: detrital organic matter / labile detritus
- $G(t)$: small grazer / filter-feeder consumer biomass
- $B(t)$: benthic detritivore consumer biomass
- $H(t)$: higher-order mobile omnivore / predator biomass
- $N(t)$: dissolved nitrogen
- $P(t)$: dissolved phosphorus
- $S_N(t)$: sediment nitrogen
- $S_P(t)$: sediment phosphorus

These pools are the minimum useful decomposition for the current simulation.

- $M$ corresponds to the rooted producer ECS population
- $A$ corresponds to `phytoplankton_load`
- $D$ corresponds to detritus entities plus pending labile detritus
- $G$, $B$, and $H$ are consumer guilds, not fixed species
- $N$, $P$, $S_N$, and $S_P$ map directly to the nutrient pool

## Forcing And Limitation Terms

### Light

Let:

- $I_0(t)$ be surface light from the day/night and weather cycle
- $z_M$ be the effective benthic depth seen by macrophytes
- $z_A$ be the effective pelagic depth seen by phytoplankton

Use Beer-Lambert attenuation:

$$
I_M = I_0 \exp\!\left(-(k_w z_M + k_A A + k_C C_M)\right)
$$

$$
I_A = I_0 \exp\!\left(-(k_w z_A + k_A A)\right)
$$

Where:

- $k_w$ is water attenuation
- $k_A$ is phytoplankton shading strength
- $C_M$ is a canopy proxy from rooted macrophytes

For a first reduced model:

$$
C_M = c_M M
$$

If we later want more realism, $C_M$ can be replaced by a saturating canopy term.

### Resource limitation

Use Monod limitation:

$$
L_I^M = \frac{I_M}{K_I^M + I_M}
$$

$$
L_N^M = \frac{N + \rho_N^M S_N}{K_N^M + N + \rho_N^M S_N}
$$

$$
L_P^M = \frac{P + \rho_P^M S_P}{K_P^M + P + \rho_P^M S_P}
$$

$$
L_I^A = \frac{I_A}{K_I^A + I_A}
$$

$$
L_N^A = \frac{N}{K_N^A + N}
$$

$$
L_P^A = \frac{P}{K_P^A + P}
$$

The effective producer limitation terms are:

$$
U_M = \min(L_I^M, L_N^M, L_P^M)
$$

$$
U_A = \min(L_I^A, L_N^A, L_P^A)
$$

This matches the simulation's current Monod-style structure and rooted nutrient
access from the sediment pools.

## Reference ODE System

### Rooted macrophytes

$$
\frac{dM}{dt}
= \mu_M U_M M
- m_M M
- \tau_M M
- g_{MG} G M
$$

Where:

- $\mu_M$ is macrophyte gross growth
- $m_M$ is maintenance / respiration
- $\tau_M$ is turnover / senescence
- $g_{MG}$ is grazing pressure from $G$

$\tau_M$ may be made stress-dependent:

$$
\tau_M = \tau_{M0} + \tau_{MS}(1 - U_M)^2
$$

That mirrors the simulation's current resource-stress turnover logic.

### Phytoplankton

$$
\frac{dA}{dt}
= \mu_A U_A A
- m_A A
- \tau_A A
- c_A A^2
- g_{AG} G A
$$

Where:

- $c_A A^2$ is self-shading / bloom instability / density-dependent loss
- $g_{AG}$ is pelagic grazing by $G$

This is the correct reduced representation of the current `phytoplankton_load`
pathway.

### Detritus

$$
\frac{dD}{dt}
= \alpha_M (m_M M + \tau_M M + g_{MG} G M_{\mathrm{unassimilated}})
+ \alpha_A (m_A A + \tau_A A + c_A A^2)
+ \alpha_G m_G G
+ \alpha_B m_B B
+ \alpha_H m_H H
- r_D D
- g_{DG} G D
- g_{DB} B D
$$

This pool is essential. Without it, the simulation cannot sustain a realistic
benthic food path.

### Grazer / filter-feeder guild

$$
\frac{dG}{dt}
= \varepsilon_{MG} g_{MG} G M
+ \varepsilon_{AG} g_{AG} G A
+ \varepsilon_{DG} g_{DG} G D
- m_G G
- q_G G (G + B + H)
- p_{HG} H G
$$

This guild should represent the consumers that currently rely on producer
grazing and small-body pelagic feeding.

### Benthic detritivore guild

$$
\frac{dB}{dt}
= \varepsilon_{DB} g_{DB} B D
- m_B B
- q_B B (G + B + H)
- p_{HB} H B
$$

This guild is needed if we want the detrital loop to matter as a stabilizer.

### Predator / omnivore guild

$$
\frac{dH}{dt}
= \varepsilon_{HG} p_{HG} H G
+ \varepsilon_{HB} p_{HB} H B
- m_H H
- q_H H (G + B + H)
$$

This guild can be omitted in a first solver if we only want a two-fauna
equilibrium. It should remain in the reference model if the goal is stable
consumer diversity.

### Dissolved nitrogen

$$
\frac{dN}{dt}
= L_N
+ \rho_{SN} S_N
+ F_N(N)
+ \eta_N r_D D
- u_{NM} \mu_M U_M M
- u_{NA} \mu_A U_A A
- \xi_N N
$$

Where:

- $L_N$ is slow external loading
- $\rho_{SN} S_N$ is sediment release
- $F_N(N)$ is low-N fixation / low-N recovery
- $\eta_N r_D D$ is remineralization from detritus
- $\xi_N N$ is export

### Dissolved phosphorus

$$
\frac{dP}{dt}
= L_P
+ \rho_{SP} S_P
+ \eta_P r_D D
- u_{PM} \mu_M U_M M
- u_{PA} \mu_A U_A A
- B_P(P)
- \xi_P P
$$

Where:

- $B_P(P)$ is high-P burial/export

A direct analogue to the simulation is:

$$
B_P(P) = b_P \max(0, P - P_{\mathrm{target}})
$$

### Sediment nitrogen

$$
\frac{dS_N}{dt}
= (1 - \eta_N) r_D D
+ \sigma_{NM} \tau_M M
- \rho_{SN} S_N
$$

### Sediment phosphorus

$$
\frac{dS_P}{dt}
= (1 - \eta_P) r_D D
+ \sigma_{PM} \tau_M M
+ B_P(P)
- \rho_{SP} S_P
$$

## Healthy-Lake Equilibrium Conditions

Let:

$$
x = [M, A, D, G, B, H, N, P, S_N, S_P]^T
$$

A candidate equilibrium $x^*$ must satisfy:

$$
F(x^*; \theta) = 0
$$

with:

- all state variables nonnegative
- at least one positive fauna pool
- positive rooted macrophyte biomass
- bounded dissolved nutrient pools

For a **clear-water, macrophyte-dominated thriving lake**, require:

$$
M^* > 0,\quad
A^* > 0,\quad
G^* > 0,\quad
B^* > 0,\quad
N^* > 0,\quad
P^* > 0
$$

and the regime constraints:

$$
M^* > A^*
$$

$$
0.01 < \frac{G^* + B^* + H^*}{M^*} < 0.35
$$

$$
0.1 < \frac{A^*}{M^*} < 0.8
$$

The exact inequalities can be tuned, but the principle is:

- plants dominate biomass
- phytoplankton remains present but not bloom-dominant
- fauna remains present but does not overwhelm primary production

## Local Stability Criterion

The equilibrium is locally stable if the Jacobian:

$$
J = \left.\frac{dF}{dx}\right|_{x = x^*}
$$

has:

$$
\operatorname{Re}(\lambda_i(J)) < 0 \quad \text{for all } i
$$

This should be the formal definition of "stable ecology" for calibration work.

The solver workflow should be:

1. solve $F(x)=0$ under positivity constraints
2. compute $J(x^*)$
3. reject equilibria with any eigenvalue of nonnegative real part
4. classify the accepted equilibrium as:
   - clear-water macrophyte regime
   - mixed regime
   - turbid phytoplankton regime
   - low-fauna remnant regime

## First Practical Solver Form

The full $10$-variable system is the right conceptual reference, but the first
practical solver for this repo should use a **reduced clear-water system**
instead of solving everything at once.

Use these unknowns:

$$
y = [M, A, D, G, B, N, P]^T
$$

and make two simplifying assumptions for the first implementation:

1. $H^* = 0$ in the default target regime
   This does not mean the full simulation cannot evolve predators. It means the
   first equilibrium solver should target a stable producer-grazer-detritivore
   lake before adding a third consumer trophic level.

2. Sediment pools are quasi-steady
   Solve $S_N^*$ and $S_P^*$ algebraically from the slower sediment balances and
   substitute them into the nutrient-limitation terms.

That gives:

$$
S_N^* = \frac{(1-\eta_N) r_D D^* + \sigma_{NM} \tau_M M^*}{\rho_{SN}}
$$

$$
S_P^* = \frac{(1-\eta_P) r_D D^* + \sigma_{PM} \tau_M M^* + B_P(P^*)}{\rho_{SP}}
$$

This is the right first step because it matches the current code structure:

- predators are not yet the main stability problem
- dissolved/sediment nutrient exchange is already coarse-grained
- the hardest current failure modes are producer overshoot, nutrient drift, and
  weak consumer recruitment

## Default Realism Target

For the default lake, the solver should search only inside a **clear-water,
macrophyte-dominated coexistence regime**.

Use these admissibility bounds for the first solver:

$$
M^* > 0,\quad A^* > 0,\quad G^* > 0,\quad B^* > 0,\quad D^* > 0
$$

$$
2 \le \frac{M^*}{A^*} \le 8
$$

$$
0.03 \le \frac{G^*}{M^*} \le 0.14
$$

$$
0.02 \le \frac{B^*}{M^*} \le 0.10
$$

$$
0.05 \le \frac{D^*}{M^*} \le 0.35
$$

$$
2 \le N^* \le 12
$$

$$
0.3 \le P^* \le 2.5
$$

These ranges are intentionally narrower than the earlier generic regime bounds.
They are meant to encode the actual project goal:

- rooted macrophytes dominate visible biomass
- phytoplankton remains present but does not control the lake
- consumers remain persistent but clearly subordinate to producer biomass
- nutrient pools stay low to moderate rather than blooming into eutrophic
  saturation

## Positive-Equilibrium Relations

For a positive equilibrium in the reduced system, the main balance equations can
be rearranged into directly interpretable conditions.

### Rooted macrophytes

If $M^* > 0$, then:

$$
\mu_M U_M^* = m_M + \tau_M^* + g_{MG} G^*
$$

Interpretation: rooted producer growth must exactly balance maintenance,
turnover, and grazer losses.

### Phytoplankton

If $A^* > 0$, then:

$$
\mu_A U_A^* = m_A + \tau_A + c_A A^* + g_{AG} G^*
$$

Interpretation: pelagic growth must balance intrinsic loss, self-limitation,
and grazer filtering.

### Grazer guild

If $G^* > 0$ and $H^* = 0$, then:

$$
\varepsilon_{MG} g_{MG} M^*
+ \varepsilon_{AG} g_{AG} A^*
+ \varepsilon_{DG} g_{DG} D^*
= m_G + q_G(G^* + B^*)
$$

Interpretation: effective assimilated intake from plants, phytoplankton, and
detritus must exactly balance maintenance and crowding loss.

### Benthic detritivore guild

If $B^* > 0$ and $H^* = 0$, then:

$$
\varepsilon_{DB} g_{DB} D^*
= m_B + q_B(G^* + B^*)
$$

Interpretation: the detrital pool must be large enough to sustain benthic
consumers after maintenance and crowding.

### Dissolved nitrogen

At equilibrium:

$$
u_{NM}\mu_M U_M^* M^*
+ u_{NA}\mu_A U_A^* A^*
= L_N + \rho_{SN}S_N^* + F_N(N^*) + \eta_N r_D D^* - \xi_N N^*
$$

Interpretation: total biological nitrogen uptake must match total nitrogen
loading, release, fixation, remineralization, and export.

### Dissolved phosphorus

At equilibrium:

$$
u_{PM}\mu_M U_M^* M^*
+ u_{PA}\mu_A U_A^* A^*
= L_P + \rho_{SP}S_P^* + \eta_P r_D D^* - B_P(P^*) - \xi_P P^*
$$

Interpretation: phosphorus uptake must match loading and recycling after burial
and export are accounted for.

These equalities are the core of the calibration problem. A parameter set is
not acceptable unless it admits a positive solution satisfying all of them
inside the realism target band.

## Solver Objective

In practice, the first implementation should solve a constrained residual
problem rather than relying on an exact symbolic solve.

Let:

$$
R(y;\theta) =
\begin{bmatrix}
R_M \\
R_A \\
R_D \\
R_G \\
R_B \\
R_N \\
R_P
\end{bmatrix}
$$
$$
\Phi(y;\theta) =
w_R \lVert R(y;\theta) \rVert_2^2
+ w_+ \, \Pi_{\mathrm{positivity}}(y)
+ w_T \, \Pi_{\mathrm{target}}(y)
+ w_S \, \Pi_{\mathrm{stability}}(y)
$$

Where:

- $\Pi_{\mathrm{positivity}}$ penalizes negative state values
- $\Pi_{\mathrm{target}}$ penalizes leaving the realism target ranges above
- $\Pi_{\mathrm{stability}}$ penalizes equilibria whose Jacobian has
  nonnegative real-part eigenvalues

This is still numerical optimization, but it is **not random search**. It is a
constrained solve against a stated ecological theory.

## How This Should Guide Simulation Calibration

The reduced solution should be used in three ways.

### 1. Initialize toward equilibrium

Founder-web defaults should be placed near the solved regime:

- initial dissolved and sediment nutrients near $(N^*, P^*, S_N^*, S_P^*)$
- initial phytoplankton near $A^*$
- founder producer biomass near $M^*$ scaled to the visible tank
- founder consumer biomass split between $G^*$ and $B^*$

This should remove the current pattern where startup behavior depends heavily on
ad hoc founder energy and detritus values.

### 2. Constrain calibration constants

Simulation constants should be fitted so that their **effective pooled
behavior** is compatible with the reduced equilibrium.

Examples:

- nutrient demand multipliers should imply the solved uptake coefficients
- crowding and reproduction suppression should imply the solved $q_G$ and $q_B$
- pelagic grazing and detrital feeding should imply the solved $\varepsilon$ and
  $g$ coefficients

### 3. Define regime-specific tests

Tests should stop asking only "did something survive" and instead ask whether a
run remains inside the intended ecological basin.

For the default regime, tests should measure whether multi-day averages remain
near the equilibrium band for:

- $M/A$
- $(G+B)/M$
- $N$
- $P$
- detritus fraction
- persistence of both grazer and detritivore consumer guilds

## Mapping To The Current Simulation

The reduced model parameters should be tied to the current code, not guessed
independently.

### Direct calibration mappings

- `producer_growth_multiplier` -> $\mu_M$, $\mu_A$
- `producer_maintenance_multiplier` -> $m_M$
- `producer_turnover_multiplier` -> $\tau_M$
- `producer_nutrient_demand_multiplier` -> $u_{NM}$, $u_{PM}$, $u_{NA}$, $u_{PA}$
- `phytoplankton_shading_multiplier` -> $k_A$
- `consumer_metabolism_multiplier` -> $m_G$, $m_B$, $m_H$

### Startup and founder mappings

- `producers_per_1000_cells` should be derived from target $M^*$
- founder consumer biomass should be derived from target $G^* + B^* + H^*$
- dissolved and sediment nutrients should initialize near $(N^*, P^*, S_N^*, S_P^*)$
- phytoplankton should initialize near $A^*$, not from an unrelated ad hoc value

### Reproduction and carrying-capacity mappings

The full simulation uses reproductive buffers, cooldowns, mating constraints,
and crowding suppression. In the reduced model these should be represented as
effective coarse parameters:

- $\beta_G$, $\beta_B$, $\beta_H$: effective recruitment efficiency
- $q_G$, $q_B$, $q_H$: crowding suppression / carrying-capacity terms

Those parameters should be fitted from simulation runs, not invented in
isolation.

## How To Use This In Practice

### 1. Build a small equilibrium solver

Implement a separate calibration tool with:

- state vector $x$
- parameter vector $\theta$
- residual function $F(x;\theta)$
- positivity-constrained nonlinear solve
- Jacobian and eigenvalue analysis

Recommended workflow:

1. choose a target regime
2. choose a parameter family
3. solve for equilibrium
4. verify stability
5. export the equilibrium and parameter set

### 2. Use the solver as the realism source of truth

The simulation should then be calibrated against:

- equilibrium biomass ratios
- equilibrium nutrient ranges
- recovery rates after perturbation
- coexistence after small shocks

This is the correct place to define:

- "realistic thriving lake"
- "stable diversity"
- alternative stable equilibria
- resilience versus fragility

### 3. Compare simulation runs against equilibrium targets

Use archived daily history to compare the full simulation to the reduced model.

Required fit diagnostics:

- mean and variance of $M$, $A$, and consumer biomass
- mean $N$ and $P$
- recovery half-life after a perturbation
- extinction frequency by guild
- regime classification over time

The simulation does not need to sit exactly at equilibrium. It should fluctuate
around the correct stable regime.

## Recommended First Regimes

The first three equilibrium families worth solving are:

### 1. Clear-water macrophyte-dominated lake

Use as the default.

Desired properties:

- $M$ dominant
- $A$ present but modest
- $G$ and $B$ persistent
- $H$ optional but bounded
- moderate nutrients
- strong resilience after small shocks

### 2. Mixed regime

Use for more visibly dynamic runs.

Desired properties:

- macrophytes and phytoplankton coexist at similar order of magnitude
- fauna remains persistent
- regime stays stable but more variable

### 3. Turbid phytoplankton-dominated regime

Use as a deliberate failure or alternate-state mode.

Desired properties:

- $A \gg M$
- weaker benthic consumers
- reduced visibility and stronger nutrient trapping

This regime is useful because many shallow lakes are genuinely bistable between
clear-water macrophyte states and turbid phytoplankton states.

## What This Should Become In The Repo

Long term, this document should drive a concrete calibration system:

- `docs/ecology-equilibrium-model.md`
- a small equilibrium solver tool
- a machine-readable parameter file for ecological regimes
- tests that assert the simulation remains near the chosen regime class

That would give the project a real "single source of truth" for realism at the
ecological level.

## Final Recommendation

Yes, this approach makes sense and should help both the simulation and the
research-fun aspect of the project.

The important constraint is this:

- do **not** treat the reduced equations as a replacement for the simulation
- do treat them as the source of truth for macro-ecological realism

That is the right division of responsibility for an evolutionary agent-based
lake model.
