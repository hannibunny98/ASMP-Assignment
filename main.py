import wave
import numpy as np
import plotfuctions as plt


def split_channels(w):
    w.setpos(0)
    signal = w.readframes(w.getnframes())
    type = {1: np.int8, 2: np.int16, 4: np.int32}.get(w.getsampwidth())
    signal = np.frombuffer(signal, dtype=type)

    num_channels = w.getnchannels()

    return [signal[channel::num_channels] for channel in range(num_channels)]

def get_timesteps(w, channels):
    channellength = len(channels[0])
    fs = w.getframerate()
    return np.linspace(0, w.getnframes()/fs, num=channellength)



w = wave.open('Measurements/measurements1.wav', 'r')
channels = split_channels(w)
Time = get_timesteps(w, channels)
plt.plot_channels(channels, Time)
plt.plot_channel_spectogram(channels[0], w.getframerate())