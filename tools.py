from typing import Tuple
import numpy as np
from numba import njit  # jit
import mkl_fft
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


@njit(cache=True)
def rotate(x: np.ndarray, y: np.ndarray,
           theta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    perform a rotation in Euclidean space

    Parameters
    ----------
        x : np.ndarray
            x

        y : np.ndarray
            y

        theta : float
            angle

    Returns
    -------
        Tuple[np.ndarray, np.ndarray]
            new (x, y)
    """
    new_x = np.cos(theta) * x - np.sin(theta) * y
    new_y = np.sin(theta) * x + np.cos(theta) * y
    return new_x, new_y


@njit(cache=True)
def get_length(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    get_length(x, y)

    calculate the length of the trajectory in phase space

    Parameters
    ----------
        x : np.ndarray
            x(t)

        y : np.ndarray
            y(t)

    Returns
    -------
        np.ndarray
            the length L(t)
    """
    diff_x2 = np.zeros_like(x)
    diff_y2 = np.zeros_like(y)
    diff_x2[1:] = np.diff(x)**2
    diff_y2[1:] = np.diff(y)**2
    segment_length = np.sqrt(diff_x2 + diff_y2)
    return np.cumsum(segment_length)


def hilbert(signal: np.ndarray) -> np.ndarray:
    """
    hilbert(signal)

    calculate the analytic signal using the `Hilbert transform` (same with scipy.signal.hilbert)

    .. math::
        S_a = S + iS_H

        where S_H is the hilbert transform of the signal

    Parameters
    ----------
        signal : np.ndarray
            signal

    Returns
    -------
        np.ndarray
           :math:'s + s_{H} i'
    """
    @njit("c16[:](c16[:])")
    def _modifyfft(x: np.ndarray) -> np.ndarray:
        """used by the function hilbert

        Parameters
        ----------
            x : np.ndarray
                [description]

        Returns
        -------
            np.ndarray
                [description]
        """
        N = x.size
        if N % 2 == 0:
            x[1:N // 2] = x[1:N // 2] * 2
            x[N // 2 + 1:] = 0
        else:
            x[1:(N + 1) // 2] = x[1:(N + 1) // 2] * 2
            x[(N + 1) // 2:] = 0
        return x

    Xf = mkl_fft.fft(signal)
    Xf = _modifyfft(Xf)
    x = mkl_fft.ifft(Xf)
    return x


def get_protophase(signal: np.ndarray,
                   event_markers: np.ndarray) -> np.ndarray:
    """
    calculate protophase from signal based on the `general event marker method` (Eq 8)

    Parameters
    ----------
        signal : np.ndarray
            real signal

        event_markers : np.ndarray
            where(index of the signal array) the trajectory go through the Poincare section

    Returns
    -------
        np.ndarray
            protophase (without mod 2\\pi)
    """
    signal = signal - np.mean(signal)
    s_a = hilbert(signal)
    x = s_a.real.copy()
    y = s_a.imag.copy()

    @njit(cache=True)
    def _cal_protophase(x: np.ndarray, y: np.ndarray,
                        event_markers: np.ndarray) -> np.ndarray:
        protophase = np.zeros_like(x)
        for i in range(event_markers.size + 1):
            if i == 0:
                start_i = 0
                end_i = event_markers[i]
            elif i == event_markers.size:
                if event_markers[i - 1] < x.size:
                    start_i = event_markers[i - 1]
                    end_i = x.size
                else:
                    continue
            else:
                start_i = event_markers[i - 1]
                end_i = event_markers[i]
            segment_x, segment_y = x[start_i:end_i], y[start_i:end_i]
            l_t = get_length(segment_x, segment_y)
            l_t = l_t * np.pi * 2 / l_t[-1]
            protophase[start_i:end_i] = l_t + np.pi * 2 * i
        return protophase

    if event_markers is None:
        l_t = get_length(x, y)
        return l_t / l_t[-1]
    else:
        return _cal_protophase(x, y, event_markers)


def get_protophase_hilbert(signal: np.ndarray,
                           y0: float = 0,
                           y0_hat: float = 0) -> np.ndarray:
    """
    calculate protophase from signal based on the hilbert plane phase(Eq 7)

    Parameters
    ----------
        signal : np.ndarray
            real signal

        y0 : float
            y0

        y0_hat : float
            y0_hat
    Returns
    -------
        np.ndarray
    """
    s_a = hilbert(signal)
    # s_a = s_a - np.mean(s_a)
    y = s_a.real - y0
    y_hat = s_a.imag - y0_hat
    protophase = np.arctan2(y_hat, y)
    protophase[protophase < 0] = protophase[protophase < 0] + np.pi * 2
    return protophase


# @njit
def flatten_phase(phase_mod_2pi: np.ndarray,
                  threshold: float = 5) -> np.ndarray:
    """
    flatten a phase whose range is (-pi, pi) 

    Parameters
    ----------
        phase_mod_2pi : np.ndarray
            a 2pi periodic phase

        threshold : float

    Returns
    -------
        np.ndarray
            phase from 0 to inf
    """
    phase_mod_2pi = phase_mod_2pi.copy()
    diff_phase = np.diff(phase_mod_2pi)
    jump_point = np.where(diff_phase < -threshold)[0]
    anti_jump_point = np.where(diff_phase > threshold)[0]
    phase_shift_point = np.zeros(jump_point.size - anti_jump_point.size,
                                 dtype=int)
    ii = 0
    jj = 0
    if anti_jump_point.size != 0:
        for j, idx in enumerate(anti_jump_point):
            for i in range(ii, jump_point.size - 1):
                if jump_point[i] < idx < jump_point[i + 1]:
                    ii = i + 1
                    break
                else:
                    phase_shift_point[jj] = jump_point[i]
                    jj = jj + 1
            start = jump_point[i] + 1
            end = idx + 1
            phase_mod_2pi[start:end] = phase_mod_2pi[start:end] + np.pi * 2
        phase_shift_point[jj:] = jump_point[i + 1:]
    else:
        phase_shift_point = jump_point

    for i in range(phase_shift_point.size + 1):
        if i == 0:
            start = 0
        else:
            start = phase_shift_point[i - 1] + 1
        if i == phase_shift_point.size:
            phase_mod_2pi[start:] = phase_mod_2pi[start:] + np.pi * (2 * i)
        else:
            end = phase_shift_point[i] + 1
        phase_mod_2pi[start:end] = phase_mod_2pi[start:end] + np.pi * (2 * i)
    return phase_mod_2pi


if __name__ == "__main__":

    def van_der_pol(t, x_v, mu, omega):
        x, y = x_v
        dxdt = y
        dydt = 0.05 * np.random.randn() + mu * (1 - x**2) * y - omega**2 * x
        return np.array([dxdt, dydt])

    [mu, omega] = 0.5, 1.11
    sol = solve_ivp(van_der_pol, [0, 500], [2, 0],
                    args=[mu, omega],
                    max_step=0.1)
    x = sol.y[0][250:2000]
    signal = (x**2 - 1.7) * x
    s_a = hilbert(signal)
    y = s_a.real
    y_hat = s_a.imag
    phase_shift = np.pi / 2
    new_y, new_yhat = rotate(y, y_hat, phase_shift)
    phase_before = np.arctan2(y_hat, y)
    phase_after = np.arctan2(new_yhat, new_y)
    flatten_phase_before = flatten_phase(phase_before)
    flatten_phase_after = flatten_phase(phase_after)
    plt.figure()
    plt.subplot(211)
    plt.plot(sol.t[250:2000], phase_before)
    plt.subplot(212)
    plt.plot(sol.t[250:2000], phase_after)

    plt.figure()
    plt.plot(sol.t[250:2000], flatten_phase_before)
    plt.plot(sol.t[250:2000], flatten_phase_after)
    plt.figure()

    plt.plot(np.abs(flatten_phase_after - flatten_phase_before))
    plt.ylim([phase_shift - 0.1, phase_shift + 0.1])
    plt.show()
