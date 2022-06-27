import matplotlib.pyplot as plt


def plot_channels(channels, Time):
    plt.figure(1)
    plt.title('Signals')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude (linear scale)')
    for i in range(len(channels)):
        plt.plot(Time, [x / (10 ** 4) / 5 + i + 1 for x in channels[i]])
    plt.show()


def plot_channel_spectogram(channel, framerate):

    plt.figure(2)
    plt.specgram(channel, Fs =framerate)
    plt.title('Spectogram of channel 1')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.show()

