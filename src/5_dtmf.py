#!/usr/bin/env python3

# %%
import numpy as np
import librosa as lib
from scipy.spatial import distance
# from librosa import stft, amplitude_to_db

import matplotlib.pyplot as plt
import librosa.display

# note: librosa defaults to 22.050 Hz sample rate; adjust if needed!

# %%
dtmf_tones = [
    ('1', 697, 1209), 
    ('2', 697, 1336), 
    ('3', 697, 1477), 
    ('A', 697, 1633),
    ('4', 770, 1209),
    ('5', 770, 1336),
    ('6', 770, 1477),
    ('B', 770, 1633),
    ('7', 852, 1209),
    ('8', 852, 1336),
    ('9', 852, 1477),
    ('C', 852, 1633),
    ('*', 941, 1209),
    ('0', 941, 1336),
    ('#', 941, 1477),
    ('D', 941, 1633),
    ('-', 0, 1130)
    ]

# %%

# TODO
# 1. familiarize with librosa stft to compute powerspectrum
# 2. extract the critical bands from the power spectrum (ie. how much energy in the DTMF-related freq bins?)
# 3. define template vectors representing the state (see dtmf_tones)
# 4. for a new recording, extract critical bands and do DP do get state sequence
# 5. backtrack & collapse

# note: you will need a couple of helper functions...

def getSpectrum(path: str):
    x, sr = lib.load(path)
    stft =  np.abs(lib.stft(x))
    powerSpectrum = lib.amplitude_to_db(stft)
    return powerSpectrum

def decode(y: np.ndarray, sr=22050) -> list:
    """y is input signal, sr is sample rate; returns list of DTMF-signals (no silence)"""
    
    D = np.zeros((len(dtmf_tones),len(y[0])+1), dtype=int)
    # breich 697 bis 941 hat indixes 60 und 90 
    # breich 1209 bis 1633 hat indixes 105 und 160 
    freqList = librosa.fft_frequencies(sr)
    resultArray = []

    for t in range(1,len(y[0])):
        indexLowerBound = np.argmax(y.T[t-1][:91])
        indexUpperBound = 105 + np.argmax(y.T[t-1][105:161])
        maxFreq = (freqList[indexLowerBound], freqList[indexUpperBound])
        minCostBeforeIndex =  np.argmin(D.T[t-1])
        resultArray.append(dtmf_tones[minCostBeforeIndex][0])
        for s in range(len(dtmf_tones)):
            cost = distance.euclidean(dtmf_tones[s][1:3],maxFreq)
            D[s,t] = cost + D[minCostBeforeIndex, t-1]

    collapsedResultArray = []
    failStateCounter = 0
    currentChar = ''
    for result in resultArray: 
        if result != currentChar or failStateCounter == 7:
            if result != '-':
                collapsedResultArray.append(result)
            currentChar = result
            failStateCounter = 0
        elif failStateCounter < 7:
            failStateCounter += 1

    return collapsedResultArray

path = "../../extern/dtmf/dtmf_123_444555_678.wav"
y = getSpectrum(path)

print(decode(y))

# %%
