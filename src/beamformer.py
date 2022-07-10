import numpy as np

from tqdm import tqdm

from .sensor import ArraySensor


class Beamformer:

    def __init__(self, array_sensor: ArraySensor, frequency_bins: int = 8192):
        f, t, Z = array_sensor.Z(nperseg=frequency_bins)

        ### TMP ###
        T = Z.shape[-1] // 2

        t = [t[T]]
        Z = Z[..., T][..., None]
        ###########

        self.f = f
        self.t = t

        self.Z = Z
        self.R = Z @ Z.swapaxes(-2, -1).conj() / Z.shape[-1]
        self.R_inv = np.linalg.inv(self.R)

        self.A = lambda u, f: array_sensor.A(u, f)

    def beampattern(self, u0: np.ndarray, u_: np.ndarray, frequency: float):
        gt = np.where(self.f > frequency)[0][0]
        st = np.where(self.f < frequency)[0][-1]

        C = self._weighting_vector(u0, (st + gt) // 2)
        A = self.A(u_, frequency)

        return C.conj().T @ A

    def weighting_vector(self, u: np.ndarray) -> np.ndarray:
        """Computes the complex weighting vectors.

        Parameters
        ----------
        u : ndarray
            Directions in form of a Qx2 or Qx3 matrix.

        Returns
        -------
        C : ndarray (complex)
            The weighting factors c(u) as a KxMxQ matrix."""

        result = []

        for idx in tqdm(range(len(self.f))):
            result.append(self._weighting_vector(u, idx))

        return np.array(result)

    def _weighting_vector(self, u: np.ndarray, idx: int) -> np.ndarray:
        raise NotImplementedError()

    def spatial_power_spectrum(self, u: np.ndarray) -> np.ndarray:
        result = 0

        for idx in tqdm(range(len(self.f))):
            result += self._spatial_power_spectrum(u, idx)

        return result

    def _spatial_power_spectrum(self, u: np.ndarray, idx: int) -> np.ndarray:
        raise NotImplementedError()


class ConventionalBeamformer(Beamformer):

    def _weighting_vector(self, u: np.ndarray, idx: int) -> np.ndarray:
        a = self.A(u, self.f[idx])

        # should be equivalent to a / sqrt(a.H * a) or a / np.linalg.norm(a, axis=0) but way faster
        return a / np.sqrt(a.shape[0])

    def _spatial_power_spectrum(self, u: np.ndarray, idx: int) -> np.ndarray:
        a = self.A(u, self.f[idx])

        # c = np.array([a.T[i].conj() @ self.R[idx] @ a[:, i] for i in range(a.shape[-1])])

        c = np.einsum('mq, nq, mn -> q', a.conj(), a, self.R[idx])

        # should be equivalent to c / a.H * a or c / np.linalg.norm(a, axis=0)**2 but way faster
        return c / a.shape[0]


class CaponBeamformer(Beamformer):

    def _weighting_vector(self, u: np.ndarray, idx: int) -> np.ndarray:
        a = self.A(u, self.f[idx])

        c = np.einsum('mq, nq, mn -> q', a.conj(), a, self.R_inv[idx])

        return (self.R_inv[idx] @ a) / c

    def _spatial_power_spectrum(self, u: np.ndarray, idx: int) -> np.ndarray:
        a = self.A(u, self.f[idx])

        c = np.einsum('mq, nq, mn -> q', a.conj(), a, self.R_inv[idx])

        return 1 / c


class WierdBeamformer(Beamformer):

    def _spatial_power_spectrum(self, u: np.ndarray, idx: int) -> np.ndarray:
        a = self.A(u, self.f[idx])

        return np.linalg.norm(a.conj().T @ self.Z[idx], axis=-1)**2
