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

        for idx in range(len(self.f)):
            result.append(self._weighting_vector(u, idx))

        return np.array(result)

    def _weighting_vector(self, u: np.ndarray, idx: int) -> np.ndarray:
        raise NotImplementedError()

    def spatial_power_spectrum(self, u: np.ndarray) -> np.ndarray:
        result = 0

        for idx in tqdm(range(len(self.f))):
            result += np.abs(self._spatial_power_spectrum(u, idx))

        return result

    def _spatial_power_spectrum(self, u: np.ndarray, idx: int) -> np.ndarray:
        raise NotImplementedError()


class ConventionalBeamformer(Beamformer):

    def _weighting_vector(self, u: np.ndarray, idx: int) -> np.ndarray:
        a = self.A(u, self.f[idx])

        # should be equivalent to a / sqrt(a.H * a) but way faster
        return a / 4  # np.linalg.norm(a, axis=0)

    def _spatial_power_spectrum(self, u: np.ndarray, idx: int) -> np.ndarray:
        a = self.A(u, self.f[idx])

        c = np.einsum('mq, nq, mn -> q', a.conj(), a, self.R[idx])

        return c / 4  # np.linalg.norm(a, axis=0)


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

        return (a.conj().T @ self.Z[idx]).sum(axis=1) / self.Z.shape[-1]
