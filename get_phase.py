import numpy as np
import matplotlib.pyplot as plt
from tools import get_protophase_hilbert, flatten_phase
from numba import njit
from scipy.integrate import solve_ivp


@njit
def fourier_coefficient(protophase: np.ndarray, n: int) -> complex:
    sn = np.exp(-1j * n * protophase)
    Np = sn.size
    return np.sum(sn / Np)


@njit
def get_phase(protophase: np.ndarray, N: int = 0) -> np.ndarray:
    phase_mod_2p = protophase % (2 * np.pi) + np.zeros(protophase.size) * 1j
    phase = phase_mod_2p.copy()
    if N == 0:
        N = protophase.size
    for n in range(1, N + 1):
        Sn = fourier_coefficient(protophase, n)
        phase = phase + 2 * Sn * (np.exp(1j *
                                         (n) * phase_mod_2p) - 1) / (1j * n)
    return phase


@njit
def get_phase_relation(protophase: np.ndarray, N: int = 0) -> np.ndarray:
    phase = np.linspace(0, np.pi * 2, 1000) + np.zeros(1000) * 1j
    new_phase = phase.copy()
    if N == 0:
        N = protophase.size
    for n in range(1, N + 1):
        Sn = fourier_coefficient(protophase, n)
        new_phase = new_phase + 2 * Sn * (np.exp(1j * n * phase) - 1) / (1j *
                                                                         n)
    return new_phase


if __name__ == "__main__":
    solution = np.load("./van_der_pol_solution.npy")
    x = solution[1][:]
    y = solution[2][:]
    t = solution[0][:]
    signal3 = np.exp(x) - 2.2
    signal1 = x
    signal2 = x
    phase1_mod_2pi = get_protophase_hilbert(signal1, 0, 0.)
    phase2_mod_2pi = get_protophase_hilbert(signal2, 0, 1.)
    phase3_mod_2pi = get_protophase_hilbert(signal3, 0, 0)
    phase1_diff = np.diff(phase1_mod_2pi)
    phase2_diff = np.diff(phase2_mod_2pi)
    phase3_diff = np.diff(phase3_mod_2pi)
    jump1_place = np.where(phase1_diff < -5)[0]
    jump2_place = np.where(phase2_diff < -5)[0]
    jump3_place = np.where(phase3_diff < -5)[0]

    T1 = (t[jump1_place[-10]] - t[jump1_place[20]]) / (jump1_place.size - 30)
    T2 = (t[jump2_place[-10]] - t[jump2_place[20]]) / (jump2_place.size - 30)
    T3 = (t[jump3_place[-10]] - t[jump3_place[20]]) / (jump3_place.size - 30)

    omega1_0 = 2 * np.pi / T1
    omega2_0 = 2 * np.pi / T2
    omega3_0 = 2 * np.pi / T3

    protophase1 = flatten_phase(phase1_mod_2pi[jump1_place[50] +
                                               10:jump1_place[-50]])
    protophase2 = flatten_phase(phase2_mod_2pi[jump1_place[50] +
                                               10:jump1_place[-50]])
    protophase3 = flatten_phase(phase3_mod_2pi[jump1_place[50] +
                                               10:jump1_place[-50]])
    t = t[jump1_place[50] + 10:jump1_place[-50]]

    phase1 = get_phase(protophase1, 48).real
    phase1 = flatten_phase(phase1.real)

    phase2 = get_phase(protophase2, 48).real
    phase2 = flatten_phase(phase2.real)

    phase3 = get_phase(protophase3, 48).real
    phase3 = flatten_phase(phase3.real)

    phase_relation = get_phase_relation(protophase3 % (2 * np.pi), 2**8).real
    phase = np.linspace(0, 2 * np.pi, 1000)
    plt.figure()
    plt.hist(phase3_mod_2pi, bins=100, density=True, cumulative=True)
    plt.plot(phase, phase_relation / (np.pi * 2))
    plt.figure()
    plt.plot(phase, phase_relation - phase)

    plt.figure()
    plt.plot(t, protophase3 - omega1_0 * (t - t[0]))
    # plt.ylim([-1, 2])
    plt.plot(t, phase3 - omega1_0 * (t - t[0]))
    # plt.ylim([-1, 2])
    plt.show()