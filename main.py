from src.sensor import ArraySensor
import numpy as np
import util.plotfunctions as plt


def get_timesteps(samplerate, channels):
    return np.linspace(0, channels.shape[0] / samplerate, num=channels.shape[1])


if __name__ == '__main__':
    array_sensor = ArraySensor('./Measurements/CrowsNest.csv', './Measurements/measurements3.wav')

    print(array_sensor.positions)
    print(array_sensor.measurments)
    print(array_sensor.samplerate)

    Time = get_timesteps(array_sensor.samplerate, array_sensor.measurments)
    plt.plot_channels(array_sensor.measurments, Time)
    plt.plot_channel_spectogram(array_sensor.measurments.T[0], array_sensor.samplerate)
