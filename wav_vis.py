import wave
import numpy as np
import matplotlib.pyplot as plt

types = {
    1: np.int8,
    2: np.int16,
    4: np.int32
}

namefile = r"meadow lark\birds013.wav"
namefile1 = r"peacock\birds010.wav"

w = wave.open(namefile, 'r')
(nchannels, sampwidth, framerate, nframes, comptype, compname) = w.getparams()
frames = w.readframes(nframes)

w1 = wave.open(namefile1, 'r')
(nchannels1, sampwidth1, framerate1, nframes1, comptype1, compname1) = w1.getparams()
frames1 = w1.readframes(nframes)

samples = np.frombuffer(frames, dtype=types[sampwidth])
samples1 = np.frombuffer(frames1, dtype=types[sampwidth1])


# преобразование фурье
fft = np.absolute(np.fft.fft(samples))
fft = fft / fft.max()

fft1 = np.absolute(np.fft.fft(samples1))
fft1 = fft1 / fft1.max()

# координаты для графика волны
x_fft1 = np.absolute(np.fft.fftfreq(len(samples1), 1 / framerate1))  # вычисляет частоты
y_fft1 = fft1

# координаты для преобразования фурье
x_fft = np.absolute(np.fft.fftfreq(len(samples), 1 / framerate))  # вычисляет частоты
y_fft = fft

# Постройка графика
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(20, 10))
fig.subplots_adjust(wspace=0, hspace=0.5)

axes[0].set_title(namefile1, fontsize=14)
axes[0].set_xlabel('Hz', fontsize=10)
axes[0].set_ylabel('Value fourier', fontsize=10)
axes[0].grid(True, c='lightgray', alpha=0.5)
axes[0].plot(x_fft1, y_fft1, c="orange")

axes[1].set_title(namefile, fontsize=14)
axes[1].set_xlabel('Hz', fontsize=10)
axes[1].set_ylabel('Value fourier', fontsize=10)
axes[1].grid(True, c='lightgray', alpha=0.5)
axes[1].plot(x_fft, y_fft)

axes[2].set_title('Twise', fontsize=14)
axes[2].set_xlabel('Hz', fontsize=10)
axes[2].set_ylabel('Value fourier', fontsize=10)
axes[2].grid(True, c='lightgray', alpha=0.5)
axes[2].plot(x_fft, y_fft)
axes[2].plot(x_fft1, y_fft1, c="orange")


fig.savefig("wave.png")
fig.show()

w.close()

w1.close()
