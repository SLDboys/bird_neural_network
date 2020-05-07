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
        np.asarray(thinned_samples)  # превращает в массив нампая
        return thinned_samples

    def __init__(self, namefile):
        w = wave.open(namefile, 'r')

        sampwidth = w.getsampwidth()
        self._framerate = w.getframerate()
        nframes = w.getnframes()

        frames = w.readframes(nframes)
        self._samples = np.frombuffer(frames, dtype=self._types[sampwidth])

        # прорежает семплы
        self._thinned_samples = self._thinSamples(self._samples, self._thin_factor)

        # умножение на окно
        window = signal.gaussian(len(self._samples), std=self._sigma_window)

        # преобразование фурье
        self._fft_array = np.absolute(np.fft.fft(self._samples * window))

        # нормировка
        self._fft_array = self._fft_array / self._fft_array.max()

        w.close

    def getSamples(self):
        return self._samples

    def getThinnedSamples(self):
        return self._thinned_samples

    def getFramerate(self):
        return self._framerate

    def getFftArray(self):
        return self._fft_array
