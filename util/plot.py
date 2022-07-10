import matplotlib.pyplot as plt
import numpy as np

from src.beamformer import Beamformer
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


def get_u(n: int, w: int = 1, center: list[float, float] = [0, 0], scale: float = 1):
    u = center + scale * (np.mgrid[:n + 1, :n + 1].T.reshape(-1, 2) * 2 / n - 1)
    m = np.where(np.square(u).sum(axis=1) <= 1)

    return np.c_[u[m], w * np.sqrt(1 - np.square(u[m]).sum(axis=1))], m


def get_images(X: np.ndarray, mask: list[int, ], n: int, layer: int = 0):
    if layer != 0:
        shape = (layer, n + 1, n + 1)
    else:
        shape = (n + 1, n + 1)

    images = np.full((np.prod(shape[-2:]), *shape[:-2]), np.nan)
    images[mask] = X.T

    return images.T.reshape(shape)


def show_image(ax, image: np.ndarray, cmap: str, center: list[float, float] = [0, 0], scale: float = 1):
    extend = np.array([np.subtract(center, scale), np.add(center, scale)]).T.reshape(-1).tolist()

    img = ax.imshow(image[::-1], extent=extend, cmap=cmap)
    ax.set_xticks([extend[0], center[0], extend[1]])
    ax.set_yticks([extend[2], center[1], extend[3]])

    return img


def plot_array_transfer_vector(array_sensor: ArraySensor, frequency: float, n: int, w: int = 1):
    u, m = get_u(n, w)

    A = array_sensor.A(u, frequency)

    images = get_images(np.angle(A), m, n, A.shape[0])

    x = np.sqrt(images.shape[0]).astype(int)
    y = images.shape[0] // x + images.shape[0] % x

    fig, ax = plt.subplots(x, y, tight_layout=False, sharey=True, sharex=True)

    img = [show_image(a, i, 'hsv') for a, i in zip(ax.reshape(-1), images)]
    fig.colorbar(img[-1], ax=ax, location='right')

    plt.show()


def plot_array_factor(array_sensor: ArraySensor, frequency: float, n: int, w: int = 1):
    u0 = np.array([[0, 0]])
    u_, m = get_u(n, w)

    AF = array_sensor.AF(u0, u_, frequency)[0]

    image = get_images(np.abs(AF), m, n)

    _, ax = plt.subplots(1)

    show_image(ax, image, 'plasma')

    plt.show()


def plot_beampattern(beamformer: Beamformer, frequency: float, n: int, w: int = 1):
    u0 = np.array([[0, 0]])
    u_, m = get_u(n, w)

    B = beamformer.beampattern(u0, u_, frequency)[0]

    image = get_images(np.abs(B), m, n)

    _, ax = plt.subplots(1)

    show_image(ax, image, 'plasma')

    plt.show()


def plot_spatial_power_spectrum(beamformer: Beamformer, n: int, center: list[float, float] = [0, 0], scale: float = 1, w: int = 1):
    u, m = get_u(n, w, center, scale)

    B = beamformer.spatial_power_spectrum(u)

    image = get_images(np.log10(np.abs(B)), m, n)

    _, ax = plt.subplots(1)

    show_image(ax, image, 'plasma', center, scale)

    plt.show()
