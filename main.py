from src.sensor import ArraySensor
from src.beamformer import ConventionalBeamformer, CaponBeamformer, Beamformer

import util.plot as plt


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

    ### Task 6 ###
    plt.plot_beampattern(beamformer1, [0, 0], 9000, 500, w=-1)  # = array factor
    plt.plot_spatial_filter(beamformer1, [0.5, 0.5], [[-0.5, -0.5], [0.3, 0.7]], 9000, 500, w=-1)
