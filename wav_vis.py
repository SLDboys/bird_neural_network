import wave
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


# прорежает семплы
# thin_factor коэф. обрезания относительно среднего значения
def thinsamples(samples, thin_factor):
    thinned_samples = []  # создаем пустой список
    average = np.sum(np.absolute(samples)) / len(samples)  # абсолютное среднее значение
    for i in samples:
        if np.absolute(i) > average * thin_factor:
            thinned_samples.append(i)
    thinned_samples = np.asarray(thinned_samples) # превращает в массив нампая
    return thinned_samples


types = {
    1: np.int8,
    2: np.int16,
    4: np.int32
}

namefile = r"sounds\meadow lark\birds013.wav"

w = wave.open(namefile, 'r')
(nchannels, sampwidth, framerate, nframes, comptype, compname) = w.getparams()
frames = w.readframes(nframes)
samples = np.frombuffer(frames, dtype=types[sampwidth])
print(w.getparams())

# преобразование фурье
# fft = np.absolute(np.fft.fft(samples))

# ham = np.hamming(len(samples))
# ham = signal.hamming(len(samples))
# fft = np.absolute(np.fft.fft(samples * ham))

# прореживание
thinfactor = 1
thinned_samples = thinsamples(samples, thinfactor) # thin_crop подбирать эмперически
fft = np.zeros(1024) # делаем изначальный нампай массив
win = signal.gaussian(1024, std=15) # создаём гауссово окно размером 1024
np.append(thinned_samples, np.zeros(len(thinned_samples)%1024)) # дополняем прорезанный массив семплов
                                                                # до кратного 1024 нулями
for i in range(len(thinned_samples)//1024):
    temp = thinned_samples[i*1024:(i+1)*1024] # берём часть массива длинной 1024 элемента (семпла)
    fft += np.absolute(np.fft.fft(temp * win)) # прибавляем к итоговому массиву результат
                                               # оконного преобразования фурье части массива
fft = fft / (len(thinned_samples)//1024) # делим массив на количество фрагментов, которые прибавляли
# нормировка
fft = fft / fft.max()

# координаты для графика волны
x = np.arange(1, len(samples) + 1)
y = samples

# координаты для прорежженного графика волны
thin_x = np.arange(1, len(thinned_samples) + 1)
thin_y = thinned_samples

# координаты для преобразования фурье
x_fft = np.absolute(np.fft.fftfreq(1024, 1 / framerate))  # вычисляет частоты
y_fft = fft

# Постройка графика
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10))
fig.subplots_adjust(wspace=0, hspace=0.4)

axes[0].set_title(namefile, fontsize=14)
axes[0].set_xlabel('Number of frames', fontsize=10)
axes[0].set_ylabel('Value frame', fontsize=10)
axes[0].grid(True, c='lightgray', alpha=0.5)

str_thin ="Thin with thin_factor {}".format(thinfactor)

axes[2].set_title(str_thin, fontsize=14)
axes[2].set_xlabel('Number of frames', fontsize=10)
axes[2].set_ylabel('Value frame', fontsize=10)
axes[2].grid(True, c='lightgray', alpha=0.5)

axes[1].set_title('FFT', fontsize=14)
axes[1].set_xlabel('Hz', fontsize=10)
axes[1].set_ylabel('Value fourier', fontsize=10)
axes[1].grid(True, c='lightgray', alpha=0.5)

axes[0].plot(x, y)
axes[2].plot(thin_x, thin_y)
axes[1].plot(x_fft, y_fft)


fig.savefig("wave.png")
fig.show()

w.close()
