import wave
from scipy import signal
import numpy as np


class Date:

    _samples = None
    _thinned_samples = None
    _framerate = 0
    _thin_factor = 1
    _sigma_window = 15
    _fft_array = None
    _date_array = None

    _types = {
        1: np.int8,
        2: np.int16,
        4: np.int32
    }

    # прорежает сепмлы
    # thin_factor - коэф.
    def _thinSamples(self, samples, thin_factor):
        thinned_samples = []  # создаем пустой список
        average = np.sum(np.absolute(samples)) / len(samples)  # абсолютное среднее значение
        for i in samples:
            if np.absolute(i) > average * thin_factor:
                thinned_samples.append(i)
        thinned_samples = np.asarray(thinned_samples)  # превращает в массив нампая
        return thinned_samples

    # Необязательные аргументы thin_factor и sigma нужны для простоты подгона значений
    def __init__(self, namefile, thin_factor=_thin_factor, sigma=_sigma_window):
        w = wave.open(namefile, 'r')

        # получение параметров из wav
        sampwidth = w.getsampwidth()
        self._framerate = w.getframerate()
        nframes = w.getnframes()

        # считываем фреймы в массив и переводим в интовое значение
        frames = w.readframes(nframes)
        self._samples = np.frombuffer(frames, dtype=self._types[sampwidth])

        # прорежает семплы
        self._thinned_samples = self._thinSamples(self._samples, thin_factor)

        # умножение на окно
        window = signal.gaussian(len(self._samples), std=sigma)

        # преобразование фурье (Тут функция Вадима)
        self._fft_array = np.absolute(np.fft.fft(self._samples * window))

        # нормировка
        self._fft_array = self._fft_array / self._fft_array.max()

        # Здесь должна быть функция для уменьшения кол-ва точек вектора или PCA
        # Результат должен записываться в _date_array

        w.close

    def getSamples(self):
        return self._samples

    def getThinnedSamples(self):
        return self._thinned_samples

    def getFramerate(self):
        return self._framerate

    def getFftArray(self):
        return self._fft_array

    def getDateArray(self):
        return self._date_array
