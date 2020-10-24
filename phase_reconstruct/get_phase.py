import numpy as np
import matplotlib.pyplot as plt
from .tools import get_protophase_hilbert, flatten_phase
from scipy.interpolate import interp1d
import time
from numba import jit


@jit
def fourier_coefficient(protophase: np.ndarray, n: int) -> complex:
    """
    Calculate Sn, See Eq.15

    Parameters
    ----------
        protophase : np.ndarray

        n : int

    Returns
    -------
        complex
            Sn
    """
    sn = np.exp(-1j * n * protophase)
    Np = sn.size
    return np.sum(sn / Np)


@jit
def proto_to_phase(protophase: np.ndarray, N: int = 0) -> np.ndarray:
    """
    transfor protophase to phase by using fourier transform

    Parameters
    ----------
        protophase : np.ndarray

        N : int, optional
            number of fourier terms need to be used

    Returns
    -------
        np.ndarray
            phase
    """
    phase_mod_2p = protophase % (2 * np.pi) + np.zeros(protophase.size) * 1j
    phase = phase_mod_2p.copy()
    if N == 0:
        N = protophase.size
    for n in range(1, N + 1):
        Sn = fourier_coefficient(phase_mod_2p, n)
        phase = phase + 2 * Sn * (np.exp(1j *
                                         (n) * phase_mod_2p) - 1) / (1j * n)
    return phase


def proto_to_phase_fast(protophase: np.ndarray,
                        nbins=1000,
                        is_interpolate=False) -> np.ndarray:
    """
    from protophase to phase by using cumulative distribution function directly.

    Parameters
    ----------
        protophase : np.ndarray

        nbins : int, optional
            by default 1000

        is_interpolate : bool, optional
            by default False

    Returns
    -------
        np.ndarray
            phase
    """
    phase_mod_2p = protophase % (2 * np.pi)
    p_proto, bin_edges = np.histogram(phase_mod_2p,
                                      bins=nbins,
                                      range=(0, 2 * np.pi),
                                      density=True)
    bin_center = bin_edges[1:] + bin_edges[:-1]
    bin_center = bin_center / 2.
    proto_to_phase = np.cumsum(p_proto) * 2 * np.pi * (bin_edges[1] -
                                                       bin_edges[0])
    if is_interpolate:
        x = np.zeros(bin_center.size + 2)
        y = np.zeros(bin_center.size + 2)
        x[1:-1] = bin_center
        x[-1] = np.pi * 2
        y[1:-1] = proto_to_phase
        y[-1] = np.pi * 2
        f = interp1d(x, y)
        new_phase = f(phase_mod_2p)
    else:
        idx = np.floor(phase_mod_2p / (2 * np.pi / nbins)).astype(int)
        new_phase = proto_to_phase[idx]
    return new_phase


@jit
def get_phase_relation(protophase: np.ndarray, N: int = 0) -> np.ndarray:
    """
    relation between protophase and phase

    Parameters
    ----------
        protophase : np.ndarray

        N : int, optional
            number of fourier terms need to be used

    Returns
    -------
        np.ndarray
            phase (protophase from 0 to 2pi)
    """
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

    protophase1, _ = flatten_phase(phase1_mod_2pi[jump1_place[50] +
                                                  10:jump1_place[-50]])
    protophase2, _ = flatten_phase(phase2_mod_2pi[jump1_place[50] +
                                                  10:jump1_place[-50]])
    protophase3, _ = flatten_phase(phase3_mod_2pi[jump1_place[50] +
                                                  10:jump1_place[-50]])
    t = t[jump1_place[50] + 10:jump1_place[-50]]

    phase1 = proto_to_phase(protophase1, 48).real
    phase1, _ = flatten_phase(phase1.real)

    phase2 = proto_to_phase(protophase2, 48).real
    phase2, _ = flatten_phase(phase2.real)

    t1 = time.time()
    phase3 = proto_to_phase(protophase3, 48).real
    t2 = time.time()
    print(t2 - t1)
    phase3, _ = flatten_phase(phase3.real)

    t1 = time.time()
    phase32 = proto_to_phase_fast(protophase3, 1000, is_interpolate=True)
    t2 = time.time()
    print(t2 - t1)
    phase32, _ = flatten_phase(phase32)

    phase_relation = get_phase_relation(protophase3 % (2 * np.pi), 2**8).real
    phase = np.linspace(0, 2 * np.pi, 1000)
    plt.figure()
    plt.hist(phase3_mod_2pi, bins=1000, density=True, cumulative=True)
    plt.plot(phase, phase_relation / (np.pi * 2))
    plt.figure()
    plt.plot(phase, phase_relation - phase)

    plt.figure()
    plt.plot(t, protophase3 - omega1_0 * (t - t[0]))
    # plt.ylim([-1, 2])
    plt.plot(t, phase3 - omega1_0 * (t - t[0]))
    plt.plot(t, phase32 - omega1_0 * (t - t[0]))
    # plt.ylim([-1, 2])
    plt.show()