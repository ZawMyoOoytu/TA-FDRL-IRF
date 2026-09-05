# TA-FDRL-IRF

## Trust-Aware Adaptive Federated Deep Reinforcement Learning for Intelligent Radio Fabric in 6G Networks

### Phase 4-A — Polarization-Aware PHY Modeling

[![Phase](https://img.shields.io/badge/Phase-4A%20Polarization--Aware%20PHY-blue)]()
[![Status](https://img.shields.io/badge/Status-Validated-success)]()
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)]()
[![RL](https://img.shields.io/badge/RL-SAC-orange)]()
[![6G](https://img.shields.io/badge/6G-IRF-purple)]()
[![Research](https://img.shields.io/badge/Project-Research-informational)]()

---

## Overview

**TA-FDRL-IRF** is a research framework for investigating **trust-aware adaptive deep reinforcement learning for Intelligent Radio Fabric (IRF) networks in 6G systems**.

The project combines:

* Intelligent Radio Fabric (IRF)
* Trust-aware network intelligence
* Adaptive trust modeling
* Deep Reinforcement Learning (DRL)
* Soft Actor-Critic (SAC)
* Resource allocation
* Polarization-aware wireless channel modeling
* Reproducible multi-seed experimentation
* Statistical validation

The project is developed incrementally through controlled research phases.

**Phase 4-A** introduces a **polarization-aware physical-layer channel model** while deliberately keeping the reinforcement-learning controller unchanged.

This separation allows the experiment to answer a specific research question:

> **What is the measurable effect of introducing a more physically expressive polarization-aware PHY model while retaining the original non-polarization-aware controller and action space?**

The purpose of Phase 4-A is therefore **not to optimize polarization yet**.

Instead, it establishes a controlled PHY baseline for the subsequent **Phase 4-B polarization-aware state representation** and later **joint IRF + polarization control**.

---

# Research Architecture

```text
                         TA-FDRL-IRF
                              │
                              ▼
                  ┌──────────────────────┐
                  │   SAC Controller     │
                  │                      │
                  │ State = 100          │
                  │ Action = 40          │
                  │                      │
                  │ Adaptive Trust       │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   IRF Environment    │
                  │                      │
                  │ Resource Allocation  │
                  │ Power Allocation     │
                  │ Queue Dynamics       │
                  │ Trust Dynamics       │
                  └──────────┬───────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │ Polarization-Aware PHY      │
              │                              │
              │ Direct Channel               │
              │       +                      │
              │ BS → IRF → User              │
              │       +                      │
              │ 2×2 Polarization Channel     │
              └──────────────┬───────────────┘
                             │
                             ▼
                    V / H Channel Response
                             │
                             ▼
                   SINR / SE / EE Metrics
                             │
                             ▼
                     Adaptive Trust
                             │
                             ▼
                         Reward
```

---

# Phase Development

The project follows a controlled progression from a scalar PHY model toward polarization-aware and eventually hardware-oriented intelligent radio control.

```text
Phase 3
Scalar Rayleigh Channel
        │
        ▼
┌─────────────────────────────┐
│ Phase 4-A                   │
│ Polarization-Aware PHY      │
│                             │
│ State = 100                 │
│ Action = 40                 │
│ Adaptive Trust              │
│ RIS Optimization = OFF      │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Phase 4-B                   │
│ Polarization-Aware State    │
│                             │
│ State ≈ 160                 │
│ Action = 40                 │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ Phase 5                     │
│ Joint IRF + Polarization    │
│ Control                     │
└─────────────┬───────────────┘
              │
              ▼
       Hardware Mapping
              │
              ▼
 Digital Controller
              │
              ▼
 Bias / Control Circuit
              │
              ▼
 PIN Diode / Varactor / RF Switch
              │
              ▼
 IRF Unit Cell
              │
              ▼
 EM Reflection / Polarization
              │
              ▼
 RF Front-End
              │
              ▼
 Baseband
```

---

# Phase 4-A

## Objective

Phase 4-A introduces a physically richer wireless channel representation by replacing the scalar channel abstraction with a **2×2 polarization channel model**.

The controller itself is intentionally not redesigned.

This creates a controlled experiment:

```text
Controller
    │
    │ unchanged
    ▼
Original 100-dimensional state
    │
    │
    ▼
Polarization-aware PHY
```

The experiment isolates the effect of **PHY-model augmentation** from the effects of changing the learning architecture or control variables.

---

# Why Polarization?

Conventional wireless simulations often represent a channel using a scalar complex coefficient:

$$
h \in \mathbb{C}
$$

This representation does not explicitly distinguish between orthogonal polarization components.

A polarization-aware model instead represents the channel using a matrix:

$$
\mathbf{H}_p =
\begin{bmatrix}
h_{VV} & h_{VH}\\
h_{HV} & h_{HH}
\end{bmatrix}
$$

where:

* \(h_{VV}\): V → V channel component
* \(h_{VH}\): H → V cross-polarization component
* \(h_{HV}\): V → H cross-polarization component
* \(h_{HH}\): H → H channel component

This provides a more expressive abstraction of the interaction between transmitted and received polarization states.

---

# Phase-4A PHY Model

## Polarization Channel

For each wireless link, the environment maintains a 2×2 complex polarization channel.

The major channel components are:

```text
Direct Channel
    H_direct

BS → IRF
    H_BS-RIS

IRF → User
    H_RIS-user
```

The effective polarization channel is modeled as:

$$
\mathbf{H}_{eff}
=
\mathbf{H}_{direct}
+
\sum_{n=1}^{N}
\phi_n
\mathbf{H}_{RIS-user,n}
\mathbf{H}_{BS-RIS,n}
$$

where:

* \(N\) is the number of IRF elements
* \(\phi_n\) represents the IRF reflection coefficient
* \(\mathbf{H}_{direct}\) is the direct BS-user channel
* \(\mathbf{H}_{BS-RIS,n}\) is the BS-to-IRF channel
* \(\mathbf{H}_{RIS-user,n}\) is the IRF-to-user channel

In Phase 4-A, IRF phase optimization remains disabled:

```text
optimize_ris = False
```

Therefore, the experiment focuses on polarization-aware channel behavior rather than joint IRF phase optimization.

---

# Transmit Polarization Reference

Phase 4-A uses a V-polarized transmit reference:

$$
\mathbf{x}_{tx}
=
\begin{bmatrix}
1\\
0
\end{bmatrix}
$$

The effective received polarization vector is:

$$
\mathbf{y}
=
\mathbf{H}_{eff}
\mathbf{x}_{tx}
$$

which produces V and H receive components:

$$
\mathbf{y}
=
\begin{bmatrix}
y_V\\
y_H
\end{bmatrix}
$$

The total polarization channel gain is computed from both receive components:

$$
G_p =
|y_V|^2 + |y_H|^2
$$

This allows the PHY model to account for both co-polarized and cross-polarized received energy.

---

# Cross-Polarization Modeling

The Phase-4A default configuration uses:

```text
cross_polarization_factor = 0.15
```

The parameter controls the relative strength of cross-polarization coupling.

The model also exposes polarization strength parameters:

```text
polarization_v_strength = 1.0
polarization_h_strength = 1.0
```

These parameters are intentionally kept explicit so that future experiments can investigate polarization imbalance and cross-polarization behavior.

---

# System Configuration

The default Phase-4A environment uses:

| Parameter                 |       Value |
| ------------------------- | ----------: |
| Users                     |          20 |
| IRF elements              |          64 |
| Bandwidth                 |     100 MHz |
| Carrier frequency         |      28 GHz |
| Maximum transmit power    |       1.0 W |
| Circuit power             |       0.1 W |
| Noise figure              |        7 dB |
| Noise density             | −174 dBm/Hz |
| Episode length            |   200 steps |
| RIS optimization          |    Disabled |
| Adaptive trust            |     Enabled |
| Polarization PHY          |     Enabled |
| Cross-polarization factor |        0.15 |

---

# State Space

Phase 4-A deliberately retains the Phase-3 state representation.

Each user contributes five state features:

```text
[SINR,
 Interference,
 Queue,
 Power,
 Trust]
```

With 20 users:

$$
20 \times 5 = 100
$$

Therefore:

```text
State dimension = 100
```

The important experimental constraint is:

> **The polarization-aware PHY does not automatically make the controller polarization-aware.**

The controller continues to observe the original 100-dimensional state.

---

# Action Space

Phase 4-A retains the Phase-3 action representation.

For 20 users:

```text
Bandwidth allocation = 20
Power allocation     = 20
```

Therefore:

$$
20 + 20 = 40
$$

```text
Action dimension = 40
```

The SAC architecture and control variables are unchanged.

---

# Adaptive Trust Model

Trust remains adaptive in Phase 4-A.

Trust dynamics consider:

* service quality
* interference
* queue behavior
* instability

The trust update uses:

```text
trust_memory = 0.90
trust_learning_rate = 0.10
trust_target_rate_bps = 1e7
```

Behavior weights:

```text
service_weight       = 0.35
interference_weight  = 0.20
queue_weight         = 0.25
instability_weight   = 0.20
```

The trust mechanism is therefore coupled to network performance rather than being a fixed external scalar.

---

# Reward Function

The Phase-4A reward combines multiple objectives:

$$
R =
w_{SE}R_{SE}
+
w_{EE}R_{EE}
+
w_T R_T
+
w_{\Delta T}R_{\Delta T}
-
P_I
-
P_P
-
P_Q
$$

where the components represent:

* spectral efficiency
* energy efficiency
* trust
* trust variation
* interference penalty
* power penalty
* queue penalty

Default weights include:

```text
SE weight            = 0.45
EE weight            = 0.20
Trust weight         = 0.15
Trust-delta weight   = 0.10
Interference penalty = 0.05
Power penalty        = 0.03
Queue penalty        = 0.07
```

---

# SAC Controller

The Soft Actor-Critic architecture is inherited from the Phase-3 baseline.

Phase 4-A does **not** introduce a new RL architecture.

Core configuration:

```text
Algorithm: SAC

Hidden dimension: 256

Actor learning rate:
3e-4

Critic learning rate:
3e-4

Temperature learning rate:
3e-4

Discount factor:
0.99

Target smoothing:
0.005

Replay buffer:
100,000

Batch size:
256
```

The purpose of this architectural constraint is experimental isolation.

Any performance difference can therefore be interpreted primarily in relation to the PHY-model change rather than an RL architecture change.

---

# Reproducible Randomness

Phase 4-A uses independent random streams for:

```text
Channel generation
        │
        └── channel_rng

Network dynamics
        │
        └── dynamics_rng
```

The environment uses:

```python
np.random.SeedSequence(seed)
```

and spawns separate random generators for channel and dynamics processes.

This prevents changes in channel-generation logic from unintentionally changing the initial network dynamics.

The original compatibility alias is retained:

```python
self.rng = self.dynamics_rng
```

This design enables controlled paired experiments.

---

# Common-Randomness Experimental Design

The ON/OFF polarization comparison uses the same seed and controlled action sequence.

```text
             Same Seed
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
 Polarization OFF     Polarization ON
        │                 │
        │                 │
        └────────┬────────┘
                 │
                 ▼
          Paired Comparison
```

Initial trust consistency was explicitly verified.

Result:

```text
Maximum initial-trust difference:
0.000000000000e+00

Consistency:
PASS
```

This makes the comparison substantially stronger than comparing unrelated random runs.

---

# Phase-4A Validation

The PHY implementation was validated before statistical benchmarking.

Validation included:

```text
State dimension              = 100
Action dimension             = 40
Users                        = 20
IRF elements                 = 64

Polarization enabled         = True
Cross-polarization factor    = 0.15

Direct channel shape         = (20, 2, 2)
BS-RIS channel shape         = (64, 2, 2)
RIS-user channel shape       = (20, 64, 2, 2)
Effective channel shape      = (20, 2, 2)
```

Mean polarization channel gain observed during validation:

```text
9.447053e+01
```

A 10-step environment execution completed successfully.

```text
Total validation reward:
3.822395

Initial mean trust:
0.857970

Final trust:
0.846262

Trust change:
-0.011708
```

The Phase-4A PHY validation completed successfully before benchmarking.

---

# Controlled Multi-Seed Benchmark

The controlled benchmark used six independent seeds:

```text
42
123
2026
4096
7777
9999
```

Each run used:

```text
200 steps
Adaptive trust = ON
RIS optimization = OFF
State dimension = 100
Action dimension = 40
SAC training = NOT PERFORMED
```

The same controlled action sequence was used for polarization ON/OFF comparison.

This isolates the PHY effect from policy-learning effects.

---

# Benchmark Results

## Aggregate Results

| Metric              | Polarization OFF | Polarization ON |   Change |
| ------------------- | ---------------: | --------------: | -------: |
| Total reward        |        57.359596 |       56.455691 | −1.5759% |
| Mean reward/step    |         0.286798 |        0.282278 | −1.5759% |
| Spectral efficiency |         0.077158 |        0.075900 | −1.6303% |
| Energy efficiency   |        783342.65 |       770387.55 | −1.6538% |
| Mean trust          |         0.597474 |        0.597047 | −0.0713% |
| Trust change        |        −0.539949 |       −0.540577 | −0.1164% |
| Mean queue          |         0.930472 |        0.930882 | +0.0440% |
| Final trust         |         0.314199 |        0.313571 | −0.2001% |
| Final queue         |         0.999864 |        0.999903 | +0.0038% |

---

# Per-Seed Reward Results

| Seed | OFF Reward | ON Reward | Difference |
| ---: | ---------: | --------: | ---------: |
|   42 |  56.952406 | 56.563265 |  −0.389141 |
|  123 |  58.294871 | 57.076171 |  −1.218700 |
| 2026 |  58.069850 | 56.241303 |  −1.828547 |
| 4096 |  56.804922 | 56.526969 |  −0.277953 |
| 7777 |  55.791591 | 55.321823 |  −0.469768 |
| 9999 |  58.243936 | 57.004617 |  −1.239319 |

All six seeds produced a negative ON−OFF reward difference.

---

# Statistical Analysis

Phase 4-A uses paired statistical analysis because each seed produces a matched polarization-OFF and polarization-ON observation.

The analysis includes:

* Paired t-test
* Wilcoxon signed-rank test
* Cohen's \(d_z\)
* 95% confidence interval of the paired difference

---

## Total Reward

```text
OFF mean:
57.359596

ON mean:
56.455691

Mean difference:
-0.903905

95% CI:
[-1.552832, -0.254978]

Paired t-test:
t = -3.580619
p = 0.015864

Wilcoxon:
W = 0
p = 0.031250

Cohen's dz:
-1.461782
```

---

## Mean Reward

```text
OFF mean:
0.286798

ON mean:
0.282278

Mean difference:
-0.004520

95% CI:
[-0.007764, -0.001275]

Paired t-test:
t = -3.580617
p = 0.015865

Wilcoxon:
W = 0
p = 0.031250

Cohen's dz:
-1.461781
```

---

## Spectral Efficiency

```text
OFF mean:
0.077158

ON mean:
0.075900

Mean difference:
-0.001258

95% CI:
[-0.002171, -0.000345]

Paired t-test:
t = -3.540844
p = 0.016548

Wilcoxon:
W = 0
p = 0.031250

Cohen's dz:
-1.445543
```

---

## Energy Efficiency

```text
OFF mean:
783342.65

ON mean:
770387.55

Mean difference:
-12955.10

95% CI:
[-21908.80, -4001.40]

Paired t-test:
t = -3.719373
p = 0.013721

Wilcoxon:
W = 0
p = 0.031250

Cohen's dz:
-1.518428
```

---

# Statistical Interpretation

The Phase-4A experiment shows a consistent performance reduction when the polarization-aware PHY is enabled under the current fixed-control configuration.

Across six seeds:

```text
Reward:
−1.58%

Spectral efficiency:
−1.63%

Energy efficiency:
−1.65%
```

The paired tests indicate statistically detectable differences for the three primary performance metrics.

However, these results **must not be interpreted as evidence that polarization is inherently harmful to 6G network performance**.

The experiment intentionally uses:

```text
Polarization-aware PHY
        +
Polarization-unaware state
        +
Original action space
        +
Original SAC architecture
```

The controller therefore does not directly observe the additional polarization information introduced by the PHY model.

The result is more appropriately interpreted as:

> **Introducing a more physically expressive polarization-aware channel model while retaining a non-polarization-aware controller produces a measurable performance gap.**

This performance gap establishes the motivation for the next experimental stage.

---

# Research Hypothesis

Phase 4-A tests the following hypothesis:

> **H₀:** Introducing polarization-aware PHY modeling does not produce a measurable difference in network performance under the controlled configuration.

> **H₁:** Introducing polarization-aware PHY modeling produces a measurable difference in network performance under the controlled configuration.

The Phase-4A results provide evidence against the null hypothesis for:

* total reward
* mean reward
* spectral efficiency
* energy efficiency

under the tested six-seed configuration.

The next phase investigates whether providing the controller with polarization-aware observations can recover this performance gap.

---

# What Phase 4-A Demonstrates

Phase 4-A successfully establishes:

### 1. Polarization-aware channel representation

The scalar channel model has been extended toward a matrix-based polarization representation.

### 2. Controlled PHY integration

The polarization model is integrated into the existing IRF environment without changing the RL architecture.

### 3. Reproducible experimentation

Separate channel and dynamics RNG streams provide controlled experimental behavior.

### 4. Common-randomness validation

Polarization ON/OFF experiments use matched initial conditions.

### 5. Multi-seed validation

Six independent seeds were evaluated.

### 6. Statistical validation

Paired inferential tests were performed on the primary performance metrics.

### 7. Research baseline

Phase 4-A provides a quantitative baseline for the next state-space augmentation.

---

# What Phase 4-A Does Not Demonstrate

Phase 4-A does **not** demonstrate:

* optimal polarization control
* joint polarization and IRF optimization
* polarization-aware reinforcement learning
* real electromagnetic simulation
* real RF hardware control
* PIN-diode control
* varactor control
* RF-switch control
* measured antenna polarization behavior
* real-world 6G deployment performance

These belong to later research stages.

---

# Current Limitations

## 1. Polarization control is not yet exposed to the agent

The action vector remains:

```text
20 bandwidth variables
+
20 power variables
=
40 actions
```

No explicit polarization-control action is currently available.

---

## 2. Polarization information is not yet fully observable

The state remains:

```text
20 × 5 = 100
```

with:

```text
SINR
Interference
Queue
Power
Trust
```

Polarization-specific observations are not yet explicitly included.

---

## 3. IRF optimization is disabled

```text
optimize_ris = False
```

The Phase-4A experiment therefore does not test joint IRF phase optimization.

---

## 4. PHY abstraction

The polarization model is a research simulation abstraction.

It does not replace a full-wave electromagnetic solver or measured RF hardware.

---

## 5. No SAC training in the controlled PHY benchmark

The multi-seed ON/OFF benchmark uses a controlled action sequence.

This is intentional.

It isolates the effect of the PHY model before allowing an adaptive RL controller to compensate for it.

---

# Reproducibility

## Environment

Recommended environment:

```text
Python 3.11+
NumPy
SciPy
PyTorch
pytest
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate:

```powershell
.venv\Scripts\Activate.ps1
```

Install project dependencies:

```powershell
pip install -r requirements.txt
```

---

# Validation Commands

Compile the environment:

```powershell
python -m py_compile environment\irf_env.py
```

Compile the validation script:

```powershell
python -m py_compile test_environment.py
```

Run the Phase-4A PHY validation:

```powershell
python test_environment.py
```

Expected validation properties:

```text
State dimension = 100
Action dimension = 40
Polarization enabled = True
Adaptive trust = True
RIS optimization = False
```

---

# Benchmark Commands

Run the single-seed controlled benchmark:

```powershell
python experiments\phase4a_polarization_benchmark.py
```

Run the six-seed benchmark:

```powershell
python experiments\phase4a_multiseed_benchmark.py
```

Run paired statistical analysis:

```powershell
python experiments\phase4a_paired_statistics.py
```

---

# Generated Results

Phase-4A produces the following research artifacts:

```text
results/
├── phase4a_polarization_benchmark.txt
├── phase4a_multiseed_benchmark.txt
└── phase4a_paired_statistics.txt
```

The reports contain:

* configuration
* seed information
* controlled benchmark results
* aggregate statistics
* paired differences
* inferential statistics
* confidence intervals
* effect sizes

---

# Project Structure

```text
TA-FDRL-IRF/
│
├── environment/
│   └── irf_env.py
│
├── experiments/
│   ├── phase4a_polarization_benchmark.py
│   ├── phase4a_multiseed_benchmark.py
│   └── phase4a_paired_statistics.py
│
├── results/
│   ├── phase4a_polarization_benchmark.txt
│   ├── phase4a_multiseed_benchmark.txt
│   └── phase4a_paired_statistics.txt
│
├── test_environment.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Experimental Design Summary

```text
                 Phase 4-A
                     │
                     ▼
       ┌─────────────────────────┐
       │ Same Environment        │
       │ Same Seeds              │
       │ Same Actions            │
       │ Same Dynamics RNG       │
       └────────────┬────────────┘
                    │
          ┌─────────┴─────────┐
          │                   │
          ▼                   ▼
   Polarization OFF     Polarization ON
          │                   │
          ▼                   ▼
      Performance          Performance
          │                   │
          └─────────┬─────────┘
                    │
                    ▼
             Paired Analysis
                    │
                    ▼
         Statistical Validation
```

This design provides a clean baseline before introducing polarization-aware observations or control.

---

# Phase 4-A Research Conclusion

Phase 4-A successfully integrates a polarization-aware PHY representation into the TA-FDRL-IRF simulation environment while preserving the original SAC controller, state dimensionality, action dimensionality, adaptive trust mechanism, and disabled IRF optimization.

The six-seed controlled experiment produced:

```text
Total reward:
−1.58%

Spectral efficiency:
−1.63%

Energy efficiency:
−1.65%
```

All six paired seeds produced negative ON−OFF differences for the primary reward metric.

Paired statistical testing found statistically detectable differences in:

```text
Total reward
Mean reward
Spectral efficiency
Energy efficiency
```

with large paired effect sizes.

The result is interpreted as a **controlled model-mismatch baseline**, rather than evidence that polarization itself reduces network performance.

The central research observation is:

```text
More expressive PHY
        +
Original controller
        ↓
Performance gap
        ↓
Need for polarization-aware observability
        ↓
Phase 4-B
```

---

# Phase 4-B Roadmap

The next phase introduces explicit polarization information into the state representation while keeping the action space fixed.

Current:

```text
Phase 4-A

State:
20 × 5 = 100

[SINR,
 Interference,
 Queue,
 Power,
 Trust]

Action:
40
```

Planned:

```text
Phase 4-B

State:
≈ 20 × 8 = 160

[SINR,
 Interference,
 Queue,
 Power,
 Trust,
 V-channel gain,
 H-channel gain,
 Cross-polarization metric]

Action:
40
```

The purpose is to isolate the effect of **polarization-aware observability**.

The controller will be allowed to perceive information that the Phase-4A controller could not explicitly observe.

---

# Phase 5 Roadmap

After establishing the Phase-4B baseline, the project can progress toward joint control.

```text
                 SAC Policy
                     │
                     ▼
            Resource Allocation
                     │
                     ▼
           Polarization Control
                     │
                     ▼
             IRF Configuration
                     │
                     ▼
           Digital Control Plane
                     │
                     ▼
             Bias / RF Control
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
      PIN Diode   Varactor   RF Switch
          │          │          │
          └──────────┼──────────┘
                     ▼
                IRF Unit Cell
                     │
                     ▼
          Electromagnetic Response
                     │
                     ▼
        Polarization / Reflection
                     │
                     ▼
              Wireless Channel
                     │
                     ▼
                RF Front-End
                     │
                     ▼
                 Baseband
```

The long-term objective is to connect the machine-learning control layer to increasingly realistic physical-layer and hardware abstractions.

---

# Research-to-Hardware Direction

The eventual architecture is intended to bridge the following layers:

```text
AI / DRL
   ↓
Network Intelligence
   ↓
Resource Allocation
   ↓
IRF Control
   ↓
Digital Controller
   ↓
Electronic Control Circuit
   ↓
RF Switching / Tuning
   ↓
IRF Metasurface Unit Cell
   ↓
Electromagnetic Field
   ↓
Polarization
   ↓
Wireless Propagation
   ↓
RF Receiver
   ↓
Baseband
```

This layered approach is intended to prevent the research from treating the wireless PHY as an isolated mathematical abstraction.

---

# Research Philosophy

The project follows a controlled progression:

```text
Change one major research variable
        ↓
Validate implementation
        ↓
Benchmark under controlled conditions
        ↓
Repeat across multiple seeds
        ↓
Perform statistical analysis
        ↓
Freeze baseline
        ↓
Introduce next research variable
```

This makes each phase independently interpretable.

---

# Phase Status

| Component                           | Status          |
| ----------------------------------- | --------------- |
| Phase-3 baseline                    | ✅ Frozen        |
| Polarization-aware PHY              | ✅ Implemented   |
| 2×2 polarization channel            | ✅ Implemented   |
| Cross-polarization modeling         | ✅ Implemented   |
| Adaptive trust                      | ✅ Enabled       |
| SAC architecture                    | ✅ Unchanged     |
| State = 100                         | ✅ Preserved     |
| Action = 40                         | ✅ Preserved     |
| IRF optimization                    | ⏸ Disabled      |
| PHY validation                      | ✅ Passed        |
| Single-seed benchmark               | ✅ Completed     |
| Six-seed benchmark                  | ✅ Completed     |
| Common-randomness verification      | ✅ Passed        |
| Paired statistical analysis         | ✅ Completed     |
| SAC training for Phase-4A benchmark | ⏸ Not performed |
| Polarization-aware state            | ⏳ Phase 4-B     |
| Joint polarization control          | ⏳ Phase 5       |
| Hardware mapping                    | ⏳ Future phase  |

---

# Citation

If this repository is used in academic research, please cite the corresponding project/paper when available.

A formal citation will be added as the research publication matures.

---

# License

This project is intended for research and educational purposes.

See the repository license for the applicable terms.

---

# Author

**Zaw Myo Oo**

Research interests include:

* 6G Intelligent Radio Fabric
* Trustworthy AI
* Deep Reinforcement Learning
* Adaptive Network Intelligence
* Quantum Computing
* Quantum Information
* Post-Quantum Cryptography
* AI-driven Cybersecurity
* Polarization-aware wireless communications

---

# Acknowledgment

This project is developed as an experimental research framework for studying the interaction between **trust-aware AI, reinforcement learning, intelligent radio environments, and increasingly realistic wireless physical-layer models**.

---

## Phase-4A Final Statement

> **Phase 4-A establishes the first polarization-aware PHY baseline of TA-FDRL-IRF. By introducing a 2×2 polarization channel while keeping the SAC controller, 100-dimensional state, 40-dimensional action space, adaptive trust mechanism, and IRF optimization setting unchanged, the experiment isolates the performance impact of PHY-level polarization modeling. The resulting performance gap provides the controlled motivation for Phase 4-B, where polarization-aware observations will be introduced into the learning state.**
