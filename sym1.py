#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

logger = Logging.setup_logging()


A_V     = 1@u_V         # амплитуда источника V(t) ~ F(t)
f_src   = 10@u_Hz       # частота синуса для транзиента

L1 = 1@u_H              # ≡ m1
L2 = 0.6@u_H            # ≡ m2
R1 = 2@u_Ohm            # ≡ b1
R2 = 1@u_Ohm            # ≡ b2
C  = (1/200)@u_F        # ≡ 1/k  (k=200 → C=0.005 F)

WIRE_R = 1@u_mOhm       # «провод» (почти 0 Ом)


def add_network(circuit, top_node='m1', t_node='t', m2_node='m2'):
    """
    Пассивная сеть:
      L1: m1→gnd
      wire m1→t
      C:  t→gnd
      R1: t→m2
      (R2 || L2): m2→gnd
    """
    circuit.L('1', top_node, circuit.gnd, L1)
    circuit.R('wire_m1_t', top_node, t_node, WIRE_R)
    circuit.C('1', t_node, circuit.gnd, C)
    circuit.R('1', t_node, m2_node, R1)
    circuit.R('2', m2_node, circuit.gnd, R2)
    circuit.L('2', m2_node, circuit.gnd, L2)

# -------------------------------------------------------
# 1) AC-ОТКЛИК (аналитически, Vin = 1 В)
# -------------------------------------------------------
def ac_response_numpy(f_start=0.5, f_stop=1e3, n_pts=400):
    """
    Возвращает: f, V_m1(f), V_m2(f) при идеальном источнике Vin=1 В.
    Так как m1 и t соединены «коротко», V_m1 ≈ V_t.
    """
    f = np.logspace(np.log10(f_start), np.log10(f_stop), n_pts)
    w = 2*np.pi*f

    # Адмиттансы для справки (влияют на ток, но не на V_m1 при идеальном источнике):
    # Y_L1 = 1/(1j*w*float(L1))
    # Y_C  = 1j*w*float(C)

    # Межузловой элемент R1 между (t=m1) и m2:
    R1_val = float(R1)
    # Шунт на m2: R2 || L2
    Y_R2 = 1/float(R2)
    Y_L2 = 1/(1j*w*float(L2))
    Y_m2 = Y_R2 + Y_L2
    Z_m2 = 1 / Y_m2

    V_m1 = np.ones_like(f, dtype=complex)           # источник держит 1 В
    V_m2 = V_m1 * (Z_m2 / (R1_val + Z_m2))          # делитель через R1 на шунт m2
    return f, V_m1, V_m2

def plot_ac(f, V_m1, V_m2):
    def mag_db(v): return 20*np.log10(np.abs(v))

    plt.figure(figsize=(7.5,4.6))
    plt.semilogx(f, mag_db(V_m1), label='|V(m1)|, dB')
    plt.semilogx(f, mag_db(V_m2), label='|V(m2)|, dB')
    plt.xlabel('Frequency [Hz]'); plt.ylabel('Magnitude [dB]')
    plt.title('AC Response (analytical, Vin = 1 V)')
    plt.grid(True, which='both', ls=':'); plt.legend(); plt.tight_layout()

    plt.figure(figsize=(7.5,4.6))
    plt.semilogx(f, np.angle(V_m1, deg=True), label='∠V(m1) [deg]')
    plt.semilogx(f, np.angle(V_m2, deg=True), label='∠V(m2) [deg]')
    plt.xlabel('Frequency [Hz]'); plt.ylabel('Phase [deg]')
    plt.title('AC Phase (analytical)')
    plt.grid(True, which='both', ls=':'); plt.legend(); plt.tight_layout()

# -------------------------------------------------------
# 2) ТРАНЗИЕНТ: синус на 10 Гц (без phase/delay_time)
# -------------------------------------------------------
def build_circuit_transient():
    c = Circuit('Mechanical Analog - Transient')
    # В старых PySpice безопасно так (без phase/delay_time):
    c.SinusoidalVoltageSource('input', 'in', c.gnd,
                              amplitude=A_V, frequency=f_src)
    c.R('wire_in_m1', 'in', 'm1', WIRE_R)
    add_network(c, top_node='m1', t_node='t', m2_node='m2')
    return c

# -------------------------------------------------------
# 3) СТУПЕНЬ: «очень длинный» PULSE ≈ постоянная 1 В
# -------------------------------------------------------
def build_circuit_step():
    c = Circuit('Mechanical Analog - Step')
    long_pw  = 1e9@u_s   # ширина импульса >> времени моделирования
    long_per = 2e9@u_s
    c.PulseVoltageSource('input', 'in', c.gnd,
                         initial_value=0@u_V, pulsed_value=A_V,
                         rise_time=1e-4@u_s, fall_time=1e-4@u_s,
                         delay_time=0@u_s, pulse_width=long_pw, period=long_per)
    c.R('wire_in_m1', 'in', 'm1', WIRE_R)
    add_network(c, top_node='m1', t_node='t', m2_node='m2')
    return c

# =======================
# ЗАПУСК
# =======================

# AC (аналитически)
f, V_m1_ac, V_m2_ac = ac_response_numpy()
plot_ac(f, V_m1_ac, V_m2_ac)

# Транзиент — синус
c_tr = build_circuit_transient()
sim_tr = c_tr.simulator(temperature=25, nominal_temperature=25)

t_stop = 5.0   # секунд
t_step = 1e-3  # шаг интегратора
analysis_tr = sim_tr.transient(end_time=t_stop@u_s, step_time=t_step@u_s)

time_tr = np.array(analysis_tr.time)
v_in_tr = np.array(analysis_tr['in'])
v_m1_tr = np.array(analysis_tr['m1'])
v_m2_tr = np.array(analysis_tr['m2'])

plt.figure(figsize=(8.5,4.6))
plt.plot(time_tr, v_in_tr, label='V(in)')
plt.plot(time_tr, v_m1_tr, label='V(m1)')
plt.plot(time_tr, v_m2_tr, label='V(m2)')
plt.xlabel('Time [s]'); plt.ylabel('Voltage [V]')
plt.title('Transient: Sinusoidal Excitation (f = 10 Hz)')
plt.grid(True, ls=':'); plt.legend(); plt.tight_layout()

# Ступень
c_step = build_circuit_step()
sim_step = c_step.simulator(temperature=25, nominal_temperature=25)

t_stop_step = 5.0
t_step_step = 1e-3
analysis_step = sim_step.transient(end_time=t_stop_step@u_s, step_time=t_step_step@u_s)

time_st = np.array(analysis_step.time)
v_m1_st = np.array(analysis_step['m1'])
v_m2_st = np.array(analysis_step['m2'])

plt.figure(figsize=(8.5,4.6))
plt.plot(time_st, v_m1_st, label='V(m1)')
plt.plot(time_st, v_m2_st, label='V(m2)')
plt.xlabel('Time [s]'); plt.ylabel('Voltage [V]')
plt.title('Step Response (Vin ≈ step of 1 V)')
plt.grid(True, ls=':'); plt.legend(); plt.tight_layout()

plt.show()
