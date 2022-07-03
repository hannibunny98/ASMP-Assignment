import numpy as np

from scipy.signal import stft

from util.file import read_csv, read_wav


def array_transfer_vector(factors: np.ndarray, directions: np.ndarray, sensor_positions: np.ndarray):
    """Computes the array transfer vector.

    Parameters
    ----------
    factors : ndarray (vector of length K)
    directions : ndarray (Qx2 matrix)
    sensor_positions : ndarray (Mx3 matrix)

    Returns
    -------
    A : ndarray (complex)
        The array transfer matrix A[k, m, q] = a_m(u_q; w_k)."""

    directions = np.c_[directions, np.sqrt(1 - np.square(directions).sum(axis=1))]

    return np.exp(np.multiply.outer(factors, (directions @ sensor_positions.T).T))


class ArraySensor:

    def __init__(self, position_file: str, measurments_file: str, velocity_factor: float):
        """

        Parameters
        ----------
        position_file : str
            Path to a .csv file containing the sensor element positions.
            The .csv should have a header row where the x, y and z columns
            are specified.
        measurments_file : str
            Path to a .wav file containing the measurments for all sensors
            as seqerate channels.
        velocity_factor : float
            Wave propagation speed of the measured wave type.
            e.g. 340.29 for sound waves or 299792458 for electromagnetic waves."""

        positions = read_csv(position_file, columns=['x', 'y', 'z'])
        measurments, samplerate = read_wav(measurments_file)

        # confirm number of sensors is matches the number of channels in the measurment
        assert positions.shape[0] == measurments.shape[1]

        self.positions = positions
        self.samplerate = samplerate
        self.measurments = measurments
        # measurments[t, m]

        self.velocity_factor = velocity_factor

    def A(self, directions: np.ndarray, frequencies: np.ndarray):
        """Computes the array transfer vector.

        Parameters
        ----------
        directions : ndarray (Qx2 matrix)
        frequencies : ndarray (vector of length K)

        Returns
        -------
        A : ndarray (complex)
            The array transfer matrix A[k, m, q] = a_m(u_q; w_k)."""

        return array_transfer_vector(1j * frequencies / self.velocity_factor, directions, self.positions)

    def Z(self, window: str = 'hann', nperseg: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the Short Time Fourier Transform (STFT).

        Parameters
        ----------
        window : str or tuple or array_like, optional
            Desired window to use. If `window` is a string or tuple, it is
            passed to `get_window` to generate the window values, which are
            DFT-even by default. See `get_window` for a list of windows and
            required parameters. If `window` is array_like it will be used
            directly as the window and its length must be nperseg. Defaults
            to a Hann window.
        nperseg : int, optional
            Length of each segment. Defaults to 256.

        Returns
        -------
        f : ndarray
            Array of sample frequencies.
        t : ndarray
            Array of segment times.
        Z : ndarray
            STFT of `self.measurments` Z[t, m, k]."""

        f, t, Z = stft(self.measurments, 1 / self.samplerate, axis=0, window=window, nperseg=nperseg)

        return f, t, Z.T
