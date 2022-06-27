import numpy as np

from scipy.fft import rfft
from scipy.signal import stft

from util.file import read_csv, read_wav


class ArraySensor:

    def __init__(self, position_file: str, measurments_file: str):
        positions = read_csv(position_file, columns=['x', 'y', 'z'])
        measurments, samplerate = read_wav(measurments_file)

        # confirm number of sensors is consistent
        assert positions.shape[0] == measurments.shape[1]

        self.positions = positions
        self.samplerate = samplerate
        self.measurments = measurments

    def rfft(self, **kwargs) -> np.ndarray:
        """Compute the 1-D discrete Fourier Transform for real input.

        Wraps `scipy.fft.rfft`.

        This function computes the 1-D *n*-point discrete Fourier
        Transform (DFT) of a real-valued array by means of an efficient algorithm
        called the Fast Fourier Transform (FFT).

        Parameters
        ----------
        n : int, optional
            Number of points along transformation axis in the input to use.
            If `n` is smaller than the length of the input, the input is cropped.
            If it is larger, the input is padded with zeros. If `n` is not given,
            the length of the input along the axis specified by `axis` is used.
        norm : {"backward", "ortho", "forward"}, optional
            Normalization mode (see `fft`). Default is "backward".
        workers : int, optional
            Maximum number of workers to use for parallel computation. If negative,
            the value wraps around from ``os.cpu_count()``.
            See :func:`~scipy.fft.fft` for more details.
        plan : object, optional
            This argument is reserved for passing in a precomputed plan provided
            by downstream FFT vendors. It is currently not used in SciPy.

        Returns
        -------
        out : complex ndarray
            The truncated or zero-padded input, transformed along the axis
            indicated by `axis`, or the last one if `axis` is not specified.
            If `n` is even, the length of the transformed axis is ``(n/2)+1``.
            If `n` is odd, the length is ``(n+1)/2``."""

        return rfft(self.measurments, axis=0, **kwargs)

    def stft(self, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the Short Time Fourier Transform (STFT).

        Wraps `scipy.signal.stft`.

        STFTs can be used as a way of quantifying the change of a
        nonstationary signal's frequency and phase content over time.

        Parameters
        ----------
        window : str or tuple or array_like, optional
            Desired window to use. If `window` is a string or tuple, it is
            passed to `get_window` to generate the window values, which are
            DFT-even by default. See `get_window` for a list of windows and
            required parameters. If `window` is array_like it will be used
            directly as the window and its length must be nperseg. Defaults
            to a Hann window.
        nperseg : int, optional
            Length of each segment. Defaults to 256.
        noverlap : int, optional
            Number of points to overlap between segments. If `None`,
            ``noverlap = nperseg // 2``. Defaults to `None`. When
            specified, the COLA constraint must be met (see Notes below).
        nfft : int, optional
            Length of the FFT used, if a zero padded FFT is desired. If
            `None`, the FFT length is `nperseg`. Defaults to `None`.
        detrend : str or function or `False`, optional
            Specifies how to detrend each segment. If `detrend` is a
            string, it is passed as the `type` argument to the `detrend`
            function. If it is a function, it takes a segment and returns a
            detrended segment. If `detrend` is `False`, no detrending is
            done. Defaults to `False`.
        return_onesided : bool, optional
            If `True`, return a one-sided spectrum for real data. If
            `False` return a two-sided spectrum. Defaults to `True`, but for
            complex data, a two-sided spectrum is always returned.
        boundary : str or None, optional
            Specifies whether the input signal is extended at both ends, and
            how to generate the new values, in order to center the first
            windowed segment on the first input point. This has the benefit
            of enabling reconstruction of the first input point when the
            employed window function starts at zero. Valid options are
            ``['even', 'odd', 'constant', 'zeros', None]``. Defaults to
            'zeros', for zero padding extension. I.e. ``[1, 2, 3, 4]`` is
            extended to ``[0, 1, 2, 3, 4, 0]`` for ``nperseg=3``.
        padded : bool, optional
            Specifies whether the input signal is zero-padded at the end to
            make the signal fit exactly into an integer number of window
            segments, so that all of the signal is included in the output.
            Defaults to `True`. Padding occurs after boundary extension, if
            `boundary` is not `None`, and `padded` is `True`, as is the
            default.

        Returns
        -------
        f : ndarray
            Array of sample frequencies.
        t : ndarray
            Array of segment times.
        Zxx : ndarray
            STFT of `x`. By default, the last axis of `Zxx` corresponds
            to the segment times."""

        return stft(self.measurments, 1 / self.samplerate, axis=0, **kwargs)
