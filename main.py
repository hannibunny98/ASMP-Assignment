from array import array
from src.sensor import ArraySensor
from src.beamformer import ConventionalBeamformer, CaponBeamformer, WierdBeamformer, Beamformer
import numpy as np
import util.plot as plt2
import matplotlib.pyplot as plt


def get_timesteps(samplerate, channels):
    return np.linspace(0, channels.shape[0] / samplerate, num=channels.shape[0])


if __name__ == '__main__':
    array_sensor = ArraySensor('./Measurements/CrowsNest.csv', './Measurements/measurements1.wav', 343)
    # beamformer = ConventionalBeamformer(array_sensor)

    n = 200
    wh = 2 * n + 1

    d = np.mgrid[:wh, :wh].T.reshape(-1, 2) / n - 1
    i = np.full(d.shape[0], np.nan)

    m = np.where(np.square(d).sum(axis=1) <= 1)

    d_down = np.c_[d, -np.emath.sqrt(1 - np.square(d).sum(axis=1))]

    # b = beamformer.spatial_power_spectrum(d_down[m])
    b = np.angle(array_sensor.A(d[m], 400)[15])
    # b = np.abs(array_sensor.AF(np.array([[0, 0]]), d[m], 500))

    i[m] = b

    fig, ax = plt.subplots()
    im = ax.imshow(i.reshape(wh, wh)[::-1, ::-1])
    plt.show()

    # exit()

    # Time = get_timesteps(array_sensor.samplerate, array_sensor.measurments)
    # plt2.plot_channels(array_sensor.measurments, Time)
    # plt2.plot_channel_spectogram(array_sensor.measurments.T[0], array_sensor.samplerate)
