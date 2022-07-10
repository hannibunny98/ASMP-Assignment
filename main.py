from src.sensor import ArraySensor
from src.beamformer import ConventionalBeamformer, CaponBeamformer

import util.plot as plt


if __name__ == '__main__':
    array_sensor = ArraySensor('./Measurements/CrowsNest.csv', './Measurements/measurements3.wav', 343)
    beamformer = ConventionalBeamformer(array_sensor)

    ### Task 1 ###
    plt.plot_channels(array_sensor)

    ### Task 2 ###
    plt.plot_channel_spectogram(array_sensor, 0)

    ### Task 3 ###
    plt.plot_array_transfer_vector(array_sensor, 9000, 500, w=1)

    ### Task 4 ###
    plt.plot_spatial_power_spectrum(beamformer, 100, w=-1)

    ### Task 5 ###
    # plt.plot_spatial_power_spectrum(beamformer, 50, center=[-0.13833, -0.85917], scale=0.0001, w=-1)

    ### Task 6 ###
    plt.plot_beampattern(beamformer, [0, 0], 9000, 500, w=-1)  # = array factor
    plt.plot_spatial_filter(beamformer, [0.5, 0.5], [[-0.5, -0.5], [0.3, 0.7]], 9000, 500, w=-1)
