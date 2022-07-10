from src.sensor import ArraySensor
from src.beamformer import ConventionalBeamformer, CaponBeamformer, WierdBeamformer, Beamformer

import numpy as np

import util.plot as plt


def get_timesteps(samplerate, channels):
    return np.linspace(0, channels.shape[0] / samplerate, num=channels.shape[0])


if __name__ == '__main__':
    array_sensor = ArraySensor('./Measurements/CrowsNest.csv', './Measurements/measurements1.wav', 343)
    beamformer = ConventionalBeamformer(array_sensor)

    # plt.plot_array_factor(array_sensor, 9000, 500, w=1)
    # plt.plot_array_transfer_vector(array_sensor, 9000, 500, w=1)
    plt.plot_spatial_power_spectrum(beamformer, 100, w=-1)
    # plt.plot_beampattern(beamformer, 9000, 100, w=1)

    # Time = get_timesteps(array_sensor.samplerate, array_sensor.measurments)
    # plt2.plot_channels(array_sensor.measurments, Time)
    # plt2.plot_channel_spectogram(array_sensor.measurments.T[0], array_sensor.samplerate)
