# impedance_sweep

Physics-style modeling project. A mechanical system is represented by an RLC network. The script computes frequency response and simulates time-domain dynamics.

## What it does
- Builds an RLC network (mechanical analog).
- Computes an analytical AC response (frequency sweep) for a 1 V ideal input.
- Runs SPICE transient simulations:
  - sinusoidal excitation (example: 10 Hz)
  - step-like excitation
- Plots magnitude/phase (Bode-style) and time traces.

## Model
Mechanical ↔ electrical analogy used here:
- mass m ↔ inductance L
- damping b ↔ resistance R
- stiffness k ↔ 1/C  (so C = 1/k)

Main nodes:
- m1 — first mass node
- m2 — second mass node
- t  — intermediate node (connected to m1 by a near-short wire)

Network (high level):
- L1: m1 -> gnd
- wire: m1 -> t
- C:  t  -> gnd
- R1: t  -> m2
- (R2 || L2): m2 -> gnd

## Files
- sym1.py — main script (AC + transient + step response).

## Requirements
- Python 3
- numpy, matplotlib
- PySpice + a working NGSpice backend

Install Python deps:
```bash
pip install numpy matplotlib PySpice
