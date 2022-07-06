import numpy as np
import wave
import csv


def read_csv(filename: str, header: bool = True, columns: list[str, ] = []) -> np.ndarray:
    """Reads a csv file and returns the content as a numpy array.

    If header is True and columns is []:
        Returns all rows except the first and all columns.
    If header is True and columns is not []:
        Returns all rows except the first and the matching columns.
    If header is False:
        Returns all rows and columns."""

    with open(filename) as file:
        file = csv.reader(file)

        # if file has a header
        if header:
            # get first row of csv
            header = next(file)

            # find index for each column in header, default to -1 if not found
            header = [header.index(c) if c in header else -1 for c in columns]

        # read data from csv
        data = []
        for row in file:
            # if file has a header
            if header:
                # read columns and default to 0 if not specified
                row = np.array(row + [0])[header]

            data.append(row)

        # convert data to np.ndarray of floats and return
        return np.array(data, dtype=float)


def read_wav(filename: str) -> np.ndarray:
    """Reads a wav file and returns the signal and samplerate.

    Takes a filename for a wav file and returns the contained
    signal as a numpy array of shape (nframes, nchannels) as
    well as the samplerate of the signal."""

    wav = wave.open(filename)
    wav.setpos(0)

    type = {1: np.int8, 2: np.int16, 4: np.int32}[wav.getsampwidth()]

    signal = np.frombuffer(wav.readframes(wav.getnframes()), dtype=type)
    signal = signal.reshape(-1, wav.getnchannels())
    # signal = signal / np.iinfo(type).max

    return signal, wav.getframerate()
