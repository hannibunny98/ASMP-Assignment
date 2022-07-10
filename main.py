from src.sensor import ArraySensor
from src.beamformer import ConventionalBeamformer, CaponBeamformer, WierdBeamformer, Beamformer

import numpy as np

import util.plot as plt


def get_timesteps(samplerate, channels):
    return np.linspace(0, channels.shape[0] / samplerate, num=channels.shape[0])


if __name__ == '__main__':
    array_sensor1 = ArraySensor('./Measurements/CrowsNest.csv', './Measurements/measurements1.wav', 343)
    array_sensor2 = ArraySensor('./Measurements/CrowsNest.csv', './Measurements/measurements2.wav', 343)
    array_sensor3 = ArraySensor('./Measurements/CrowsNest.csv', './Measurements/measurements3.wav', 343)
    beamformer1 = CaponBeamformer(array_sensor1)
    beamformer2 = CaponBeamformer(array_sensor2)
    beamformer3 = CaponBeamformer(array_sensor3)

    # plt.plot_array_factor(array_sensor, 9000, 500, w=1)
    # plt.plot_array_transfer_vector(array_sensor, 9000, 500, w=1)
    # plt.plot_spatial_power_spectrum(beamformer1, 100, w=-1)
    print(beamformer1.maximum_lieklihood(array_sensor1))
    # plt.plot_spatial_power_spectrum(beamformer, 50, center=[-0.13833, -0.85917], scale=0.0001, w=-1)
    # plt.plot_beampattern(beamformer, 9000, 100, w=1)

    # Time = get_timesteps(array_sensor.samplerate, array_sensor.measurments)
    # plt2.plot_channels(array_sensor.measurments, Time)
    # plt2.plot_channel_spectogram(array_sensor.measurments.T[0], array_sensor.samplerate)
