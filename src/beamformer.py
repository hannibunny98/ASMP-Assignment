import numpy as np

from .sensor import ArraySensor


class Beamformer:

    def __init__(self, array_sensor: ArraySensor):
        f, t, Z = array_sensor.Z()

        self.f = f
        self.t = t

        self.Z = Z
        self.R = Z @ Z.conj().T / Z.shape[0]
        self.R_inv = np.linag.inv(self.R)

        self.A = lambda x: array_sensor.A(x, self.f)

    def spatial_receiving_characteristic(self, directions: np.ndarray):
        raise NotImplementedError()

    def spatial_power_spectrum(self, directions: np.ndarray):
        raise NotImplementedError()


class Conventional(Beamformer):

    def spatial_receiving_characteristic(self, directions: np.ndarray):
        a = self.A(directions)
        aH = a.swapaxes(-2, -1).conj()

        c = a / np.sqrt(aH @ a)

        return (c).swapaxes(-2, -1).conj() @ a

    def spatial_power_spectrum(self, directions: np.ndarray):
        a = self.A(directions)
        aH = a.swapaxes(-2, -1).conj()

        return (aH @ self.R @ a) / (aH @ a)


class Capon(Beamformer):

    def spatial_receiving_characteristic(self, directions: np.ndarray):
        a = self.A(directions)
        aH = a.swapaxes(-2, -1).conj()

        c = self.R_inv @ a / aH @ self.R_inv @ a

        return (c).swapaxes(-2, -1).conj() @ a

    def spatial_power_spectrum(self, directions: np.ndarray):
        a = self.A(directions)
        aH = a.swapaxes(-2, -1).conj()

        return 1 / (aH @ self.R_inv @ a)
