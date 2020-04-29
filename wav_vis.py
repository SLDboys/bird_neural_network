import wave
import numpy as np
import matplotlib.pyplot as plt

types = {
    1: np.int8,
    2: np.int16,
    4: np.int32
}

namefile = r"sounds\crow\birds017.wav"

w = wave.open(namefile, 'r')
(nchannels, sampwidth, framerate, nframes, comptype, compname) = w.getparams()
frames = w.readframes(nframes)
samples = np.frombuffer(frames, dtype=types[sampwidth])
print(w.getparams())
# преобразование фурье
fft = np.absolute(np.fft.fft(samples))

# координаты для графика волны
x = np.arange(1, len(samples) + 1)
y = samples

# координаты для преобразования фурье
x_fft = np.absolute(np.fft.fftfreq(len(samples), 1 / framerate))  # вычисляет частоты
y_fft = fft

# Постройка графика
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
fig.subplots_adjust(wspace=0, hspace=0.3)

axes[0].set_title('Original audio', fontsize=14)
axes[0].set_xlabel('Number of frames', fontsize=10)
axes[0].set_ylabel('Value frame', fontsize=10)
axes[0].grid(True, c='lightgray', alpha=0.5)


axes[1].set_title('FFT', fontsize=14)
axes[1].set_xlabel('Hz', fontsize=10)
axes[1].set_ylabel('Value fourier', fontsize=10)
axes[1].grid(True, c='lightgray', alpha=0.5)

axes[0].plot(x, y)
axes[1].plot(x_fft, y_fft)

fig.savefig("wave.png")
fig.show()

w.close()
