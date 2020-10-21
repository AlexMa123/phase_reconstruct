import numpy as np
from numba import njit  # jit
import mkl_fft
import matplotlib.pyplot as plt


@njit(cache=True)
def get_length(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """calculate the length of the trajectory in phase space

    Args:
        x (np.ndarray): x(t)
        y (np.ndarray): y(t)

    Returns:
        np.ndarray: the length L(t)
    """
    diff_x2 = np.zeros_like(x)
    diff_y2 = np.zeros_like(y)
    diff_x2[1:] = np.diff(x)**2
    diff_y2[1:] = np.diff(y)**2
    segment_length = np.sqrt(diff_x2 + diff_y2)
    return np.cumsum(segment_length)


def hilbert(signal: np.ndarray) -> np.ndarray:
    """calculate the analytic signal using the Hilbert transform (same with scipy.signal.hilbert)

    Args:
        signal ([np.ndarray]): signal

    Returns:
        [np.ndarray]: s + s_H * i
    """
    @njit("c16[:](c16[:])")
    def _modifyfft(x: np.ndarray) -> np.ndarray:
        """used by the function hilbert

        Args:
            x (np.ndarray): [description]

        Returns:
            np.ndarray: [description]
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
    """calculate protophase from signal based on the general event marker method (Eq 8)

    Args:
        signal (np.ndarray): real signal
        event_markers (np.ndarray[int]): where(index of the signal array) the trajectory go through the Poincare section

    Returns:
        np.ndarray: protophase (without mod 2pi)
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


if __name__ == "__main__":
    f = 10
    x = np.linspace(0, 8 * np.pi, 1000 * f)
    y = np.sin(x) + np.random.rand(1000 * f) * 1
    w0 = 1
    event_markers = np.array([250 * f, 500 * f, 750 * f])
    plt.figure()
    plt.hist(get_protophase(y, event_markers) - w0 * x, bins=100)
    # plt.ylim([-0.3, 0.3])
    plt.show()