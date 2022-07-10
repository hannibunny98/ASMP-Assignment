import matplotlib.pyplot as plt
import numpy as np

from src.beamformer import Beamformer
from src.sensor import ArraySensor


def plot_channels(array_sensor: ArraySensor):
    channels = array_sensor.measurments.T
    time = np.linspace(0, channels.shape[1] / array_sensor.samplerate, num=channels.shape[1])

    plt.figure(1)
    plt.title('Signals')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude (linear scale)')
    for i, channel in enumerate(channels):
        plt.plot(time, channel / (2.25 * np.abs(channel).max()) + i + 1)
    plt.show()


def plot_channel_spectogram(array_sensor: ArraySensor, channel: int):
    channel = array_sensor.measurments.T[channel]

    plt.figure(2)
    plt.specgram(channel, Fs=array_sensor.samplerate)
    plt.title('Spectogram of channel 1')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.show()


def get_u(n: int, center: list[float, float] = [0, 0], scale: float = 1, w: int = 1):
    """Compute list of direction vectors.

    Parameters
    ----------
    n : int
        Resolution in u and v direction.
    center : list[float, float]
        Center around which (u, v) is sampled. (default=[0,0])
    scale : float
        The size of the sample window (center +- scale). (default=1)
    w : int (-1|+1)
        Either +1 for the upper half sphere or -1 for the lower half sphere.

    Returns
    -------
    u : ndarray
        A ndarray containing (u, v, w) coordinates normed to 1.
    m : list[int, ]
        A list containing the masking index used to create u from a
        uniform sampled window of size (n+1)x(n+1)."""

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
    u, m = get_u(n, w=w)

    A = array_sensor.A(u, frequency)

    images = get_images(np.angle(A), m, n, A.shape[0])

    x = np.sqrt(images.shape[0]).astype(int)
    y = images.shape[0] // x + images.shape[0] % x

    fig, ax = plt.subplots(x, y, tight_layout=False, sharey=True, sharex=True)

    img = [show_image(a, i, 'hsv') for a, i in zip(ax.reshape(-1), images)]
    fig.colorbar(img[-1], ax=ax, location='right')

    plt.show()


def plot_beampattern(beamformer: Beamformer, u0: list[float, float], frequency: float, n: int, w: int = 1):
    u, m = get_u(n, w=w)

    C = beamformer.A(np.array([u0]), frequency)
    B = beamformer.beampattern(u, C, frequency)[0]

    image = get_images(np.abs(B), m, n)

    _, ax = plt.subplots(1)

    show_image(ax, image, 'plasma')

    plt.show()


def plot_spatial_filter(beamformer: Beamformer, u0: list[float, float], v: list[list[float, float], ], frequency: float, n: int, w: int = 1):
    u, m = get_u(n, w=w)

    C = beamformer.spatial_filter(np.array([u0]), np.array(v), frequency)
    B = beamformer.beampattern(u, C, frequency)[0]

    image = get_images(np.abs(B), m, n)

    _, ax = plt.subplots(1)

    show_image(ax, image, 'plasma')

    plt.show()


def plot_spatial_power_spectrum(beamformer: Beamformer, n: int, center: list[float, float] = [0, 0], scale: float = 1, w: int = 1):
    u, m = get_u(n, center, scale, w=w)

    B = beamformer.spatial_power_spectrum(u)

    image = get_images(np.abs(B), m, n)

    _, ax = plt.subplots(1)

    show_image(ax, image, 'plasma', center, scale)

    plt.show()
