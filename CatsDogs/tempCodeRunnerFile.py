import numpy as np
import librosa

wave, sr = librosa.load("data/cat_1.wav", mono=True)
print(sr)