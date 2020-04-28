import wave
import numpy as np
import matplotlib.pyplot as plt

types = {
    1: np.int8,
    2: np.int16,
    4: np.int32
}

namefile = "crow.wav"

w = wave.open(namefile, 'r')
(nchannels, sampwidth, framerate, nframes, comptype, compname) = w.getparams()

duration = nframes / framerate
frames = w.readframes(nframes)
samples = np.frombuffer(frames, dtype= types[sampwidth])
lens = len(samples)
x = np.arange(1, len(samples)+1)
y = samples

fig, ax = plt.subplots()
ax.plot(x, y)
fig.savefig("test.png")
fig.show()
w.close()