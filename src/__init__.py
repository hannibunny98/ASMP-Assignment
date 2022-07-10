import numpy as np


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
