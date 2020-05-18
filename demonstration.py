import joblib
import wave
from scipy import signal
import numpy as np

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

# определяет индекс по частоте, кол-ву семплов и фреймрейту
# если индекс получился дробным, то округляет его
def indOfFreq(freq, lenght, framerate):
    return round((freq/framerate) * lenght)

types = {
    1: np.int8,
    2: np.int16,
    4: np.int32
}

namefile = r"Demonstration\(6).wav"
w = wave.open(namefile, 'r')
(nchannels, sampwidth, framerate, nframes, comptype, compname) = w.getparams()
frames = w.readframes(nframes)
samples = np.frombuffer(frames, dtype=types[sampwidth])
fft_len = 1024
thinfactor = 1
thinned_samples = thinsamples(samples, thinfactor)
fft = np.zeros(fft_len)
win = signal.gaussian(fft_len, std=15)
for i in range(len(thinned_samples) // fft_len):
    temp = thinned_samples[i * fft_len:(i + 1) * fft_len]
    fft += np.absolute(np.fft.fft(temp * win))
fft = fft / (len(thinned_samples) // fft_len)
fft = fft / fft.max()
a = indOfFreq(100, 1024, framerate)
b = indOfFreq(6000, 1024, framerate)
fft = fft[a:b]

RFC_model = joblib.load("neuron.pkl")

RFC_predictions = RFC_model.predict(fft.reshape(1, -1))

print(RFC_predictions)


# 1 - Nightingale
# 2 - Tit
# 3 - Duck
# 4 - Duck
# 5 - Nightingale
# 6 - Tit

