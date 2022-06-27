from util import read_csv, read_wav


class ArraySensor:

    def __init__(self, position_file: str, measurments_file: str):
        positions = read_csv(position_file, columns=['x', 'y', 'z'])
        measurments, samplerate = read_wav(measurments_file)

        # confirm number of sensors is consistent
        assert positions.shape[0] == measurments.shape[1]

        self.positions = positions
        self.samplerate = samplerate
        self.measurments = measurments
