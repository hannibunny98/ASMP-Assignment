import numpy as np

from tqdm import tqdm
import scipy.optimize as opt

from .sensor import ArraySensor


def projection_matrix(X: np.ndarray, orthogonal: bool = False):
    X = np.atleast_2d(X)
    P = X @ np.linalg.inv(X.conj().T @ X) @ X.conj().T

    if orthogonal:
        return np.eye(P.shape[0]) - P
    else:
        return P


class Beamformer:

    def __init__(self, array_sensor: ArraySensor, frequency_bins: int = 8192):
        self.array_sensor = array_sensor

        self.f, self.t, self.Z = array_sensor.Z(nperseg=frequency_bins)

        ### TMP ###
        T = self.Z.shape[-1] // 2

        self.t = [self.t[T]]
        self.Z = self.Z[..., T][..., None]
        ###########

        self.R = self.Z @ self.Z.swapaxes(-2, -1).conj() / self.Z.shape[-1]
        self.R_inv = np.linalg.inv(self.R)

    def A(self, u: np.ndarray, frequency: float):
        """Returns array transfer vectors.

        Parameters
        ----------
        u : ndarray
            Directions in form of a Qx2 or Qx3 matrix.
        frequency : float
            Center frequency.

        Returns
        -------
        A : ndarray (complex)
            The array transfer matrix A[m, q] = a_m(u_q; frequency)."""

        return self.array_sensor.A(u, frequency)

    def B(self, u: np.ndarray, v: np.ndarray, frequency: float):
        """Returns array transfer vectors with deterministic nulling.

        Parameters
        ----------
        u : ndarray
            Directions in form of a Qx2 or Qx3 matrix.
        v : ndarray
            Directions for deterministic nulling in form of a Qx2 or Qx3 matrix.
        frequency : float
            Center frequency.

        Returns
        -------
        B : ndarray (complex)
            The array transfer matrix B[m, q] = b_m(u_q, v; frequency)."""

        P = projection_matrix(self.A(v, frequency), orthogonal=True)
        A = self.A(u, frequency)

        return P @ A

    def _nearest_k(self, frequency: float):
        gt = np.where(self.f > frequency)[0][0]
        st = np.where(self.f < frequency)[0][-1]

        return (st + gt) // 2

    def beampattern(self, u: np.ndarray, c: np.ndarray, frequency: float):
        """Computes the beampattern function for a given weight vector.

        Parameters
        ----------
        u : ndarray
            Directions in form of a Qx2 or Qx3 matrix.
        c : ndarray
            The weight vector to be applied.
        frequency : float | None
            Frequency for which to calculate c. (default=None)

        Returns
        -------
        beampattern : ndarray"""

        return c.conj().T @ self.A(u, frequency)

    def weighting_vector(self, u: np.ndarray, frequency: float = None) -> np.ndarray:
        """Computes the complex weighting vectors.

        Parameters
        ----------
        u : ndarray
            Directions in form of a Qx2 or Qx3 matrix.
        frequency : float | None
            Frequency for which to calculate c. (default=None)

        Returns
        -------
        c : ndarray (complex)
            The weighting factors c(u) as a MxQ or KxMxQ matrix."""
        if frequency is not None:
            return self._weighting_vector(u, self._nearest_k(frequency))

        result = []

        for idx in tqdm(range(len(self.f))):
            result.append(self._weighting_vector(u, idx))

        return np.array(result)

    def _weighting_vector(self, u: np.ndarray, idx: int) -> np.ndarray:
        raise NotImplementedError()

    def spatial_filter(self, u: np.ndarray, v: np.ndarray, frequency: float = None):
        """Computes the complex weighting vectors for a spatial
        filter with deterministic nulling in directions v.

        Parameters
        ----------
        u : ndarray
            Directions in form of a Qx2 or Qx3 matrix.
        v : ndarray
            Directions for deterministic nulling in form of a Qx2 or Qx3 matrix.
        frequency : float | None
            Frequency for which to calculate c. (default=None)

        Returns
        -------
        c : ndarray (complex)
            The weighting factors c(u) as a MxQ or KxMxQ matrix."""

        if frequency is not None:
            return self._spatial_filter(u, v, self._nearest_k(frequency))

        result = []

        for idx in tqdm(range(len(self.f))):
            result.append(self._spatial_filter(u, v, idx))

        return np.array(result)

    def _spatial_filter(self, u: np.ndarray, v: np.ndarray, idx: int):
        raise NotImplementedError()

    def spatial_power_spectrum(self, u: np.ndarray, frequency: float = None) -> np.ndarray:
        """Computes the spatial power spectrum.

        Parameters
        ----------
        u : ndarray
            Directions in form of a Qx2 or Qx3 matrix.
        frequency : float | None
            Frequency for which to calculate c. (default=None)

        Returns
        -------
        p : ndarray (complex)
            The spatial power spectrum over frequency for each direction in u."""

        if frequency is not None:
            return self._spatial_power_spectrum(u, self._nearest_k(frequency))

        result = 0

        for idx in tqdm(range(len(self.f))):
            result += self._spatial_power_spectrum(u, idx)

        return result

    def _spatial_power_spectrum(self, u: np.ndarray, idx: int) -> np.ndarray:
        raise NotImplementedError()

    def logL(self, log_par, data):
        N = len(data)

        a, b = np.exp(log_par)
        log = N * np.log(a * b) + (a - 1) * np.sum(np.log(data)) + (b - 1) * np.sum(np.log(1 - np.power(data, a)))
        return log

    def maximum_lieklihood(self, u):
        data = self.spatial_power_spectrum(u)
        res = opt.minimize(
        fun=lambda log_params, data: -self.logL(log_params, data),
        x0=np.array([0.5, 0.5]), args=(data,), method='BFGS')

        a, b = np.exp(res.x)
        return a, b




class ConventionalBeamformer(Beamformer):

    def _weighting_vector(self, u: np.ndarray, idx: int) -> np.ndarray:
        a = self.A(u, self.f[idx])

        # should be equivalent to a / sqrt(a.H * a) but way faster
        return a / np.linalg.norm(a, axis=0)

    def _spatial_filter(self, u: np.ndarray, v: np.ndarray, idx: int):
        b = self.B(u, v, self.f[idx])

        # should be equivalent to b / sqrt(b.H * b) but way faster
        return b / np.linalg.norm(b, axis=0)

    def _spatial_power_spectrum(self, u: np.ndarray, idx: int) -> np.ndarray:
        c = self._weighting_vector(u, idx)

        return np.einsum('mq, mn, nq -> q', c.conj(), self.R[idx], c)


class CaponBeamformer(Beamformer):

    def _weighting_vector(self, u: np.ndarray, idx: int) -> np.ndarray:
        a = self.A(u, self.f[idx])

        c = np.einsum('mq, nq, mn -> q', a.conj(), a, self.R_inv[idx])

        return (self.R_inv[idx] @ a) / c

    def _spatial_filter(self, u: np.ndarray, v: np.ndarray, idx: int):
        b = self.B(u, v, self.f[idx])

        c = np.einsum('mq, nq, mn -> q', b.conj(), b, self.R_inv[idx])

        return (self.R_inv[idx] @ b) / c

    def _spatial_power_spectrum(self, u: np.ndarray, idx: int) -> np.ndarray:
        a = self.A(u, self.f[idx])

        c = np.einsum('mq, nq, mn -> q', a.conj(), a, self.R_inv[idx])

        return 1 / c
