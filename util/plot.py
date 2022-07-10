import matplotlib.pyplot as plt
import numpy as np

from src.sensor import ArraySensor


def plot_channels(channels, Time):
    plt.figure(1)
    plt.title('Signals')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude (linear scale)')
    for i in range(channels.shape[1]):
        plt.plot(Time, [x / (10 ** 4) / 5 + i + 1 for x in channels.T[i]])
    plt.show()


def plot_channel_spectogram(channel, framerate):

    plt.figure(2)
    plt.specgram(channel, Fs=framerate)
    plt.title('Spectogram of channel 1')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.show()


def plot_array_transfer_vector(array_sensor: ArraySensor, frequency: float, n: int, w: int = 1):
    images = np.full(((n + 1)**2, array_sensor.positions.shape[0]), np.nan)

    d = np.mgrid[:n + 1, :n + 1].T.reshape(-1, 2) * 2 / n - 1
    m = np.where(np.square(d).sum(axis=1) <= 1)

    A = array_sensor.A(np.c_[d[m], w * np.sqrt(1 - np.square(d[m]).sum(axis=1))], frequency)

    images[m] = np.angle(A).T
    images = images.T.reshape((-1, n + 1, n + 1))

    x = np.sqrt(images.shape[0]).astype(int)
    y = images.shape[0] // x + images.shape[0] % x

    fig, ax = plt.subplots(x, y, tight_layout=False, sharey=True, sharex=True)

    for axes, image in zip(ax.reshape(-1), images):
        img = axes.imshow(image[::-1, ::-1], extent=[-1, 1, -1, 1], cmap='hsv')
        axes.set_xticks([-1, 0, 1])
        axes.set_yticks([-1, 0, 1])

    fig.colorbar(img, ax=ax, location='right')

    plt.show()
