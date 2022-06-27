from src.sensor import ArraySensor


if __name__ == '__main__':
    array_sensor = ArraySensor('./Measurements/CrowsNest.csv', './Measurements/measurements3.wav')

    print(array_sensor.positions)
    print(array_sensor.measurments)
    print(array_sensor.samplerate)
