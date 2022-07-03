import numpy as np

from .sensor import ArraySensor


class Beamformer:

    def __init__(self, array_sensor: ArraySensor):
        f, t, Z = array_sensor.Z(nperseg=4096)

        self.f = f
        self.t = t

        self.Z = Z
        self.R = Z @ Z.swapaxes(-2, -1).conj() / Z.shape[-1]
        self.R_inv = np.linalg.inv(self.R)

        self.A = lambda x: array_sensor.A(x, self.f)

    def spatial_receiving_characteristic(self, directions: np.ndarray):
        raise NotImplementedError()

    def spatial_power_spectrum(self, directions: np.ndarray):
        a = self.A(directions)

        c = np.einsum('kmq, tmk -> ktq', a.conj(), self.Z)

        return (np.abs(c)**2).sum(axis=0)


class ConventionalBeamformer(Beamformer):

    def spatial_receiving_characteristic(self, directions: np.ndarray):
        a = self.A(directions)

        c = np.sqrt(np.einsum('kmq, kmq -> kq', a.conj(), a))
        c = a / c[:, None, :]

        return np.einsum('kmq, kmq -> kq', c.conj(), a)

    def spatial_power_spectrum(self, directions: np.ndarray):
        a = self.A(directions)

        return np.einsum('kmq, tnm, kmq -> tknq', a.conj(), self.R, a) \
            / np.einsum('kmq, kmq -> kq', a.conj(), a)


class CaponBeamformer(Beamformer):

    def spatial_receiving_characteristic(self, directions: np.ndarray):
        a = self.A(directions)

        c = np.einsum('tnm, kmq -> tknq', self.R_inv, a) \
            / np.einsum('kmq, tnm, kmq -> tknq', a.conj(), self.R_inv, a)

        return np.einsum('tkmq, kmq -> tkq', c.conj(), a)

    def spatial_power_spectrum(self, directions: np.ndarray):
        a = self.A(directions)

        return 1 / np.einsum(f'knq, tnm, kmq -> tkq', a.conj(), self.R_inv, a)
