#!/usr/bin/env python3

# %%
import numpy as np
import librosa as lib
import os
import re
import scipy as sp

from librosa.feature import mfcc

# %%

def getWavPath(names: list, numbers: list) -> list:
    path = "..\..\extern\\recordings\\"
    files = []
    numNames = [num + "_" + name for num in map(str, numbers) for name in names]
    for val in numNames:
        files = files + [f for f in os.listdir(path) if val in f]
    files = [path + s for s in files]
    return files


def loadWavs(wavPath: list) -> list:
    result = []
    print(wavPath)
    for w in wavPath:
        x, sr = lib.load(w)
        result.append((x, sr))
    return result


# %%
def computeMFCC(wavs: list) -> list:
    result = []
    for wav in wavs:
        result.append(mfcc(wav[0], wav[1], \
            hop_length=int(wav[1]*0.01), n_fft=int(wav[1]*0.025)))
    return result

# TODO: read in files, compute MFCC, organize

# %%
def dtw(obs1: list, obs2: list, dist) -> float:
    D = np.full((len(obs1) + 1, len(obs2) + 1), np.inf, dtype=float)
    D[0, 0] = 0

    for i in range(1, len(obs1) + 1):
        for j in range(1, len(obs2) + 1):
            cost = dist[i-1, j-1]
            D[i, j] = cost + min(D[i-1, j],
                                D[i, j-1],
                                D[i-1, j-1])
    return D[len(obs1)][len(obs2)]

def costMatrix(seq1: list, seq2: list) -> list:
    seq1, seq2 = np.atleast_2d(seq1, seq2)
    return sp.spatial.distance.cdist(seq1.T, seq2.T, metric="euclidean")



# %% [markdown]
"""
# Experiment 1

Compute DTW scores between different digits and speakers.
How do scores change across speakers and across digits?
""" 
# 0-49 = 0 George, 50 - 99 = 0 Jackson, 100 - 149 = 5 George, 150 - 199 = 5 Jackson

# names = ["george", "jackson"]
# numbers = [0, 5]
# wp = getWavPath(names, numbers)
# lw = loadWavs(wp)
# mfcc = computeMFCC(lw)

# print(dtw(mfcc[7], mfcc[77], costMatrix(mfcc[7], mfcc[77]))) # g0 j0
# print(dtw(mfcc[7], mfcc[160], costMatrix(mfcc[7], mfcc[160]))) #g0 j5

# print(dtw(mfcc[120], mfcc[5], costMatrix(mfcc[120], mfcc[5]))) # g5 g0
# print(dtw(mfcc[140], mfcc[60], costMatrix(mfcc[140], mfcc[60]))) # g5 j0

# %%
def recognize(obs: list, refs: dict) -> str:
    """
    obs: input observations (mfcc)
    refs: dict of (classname, observations) as references
    returns classname where distance of observations is minumum
    """
    minimum = np.inf
    result = ""
    cost = 0
    for k, v in refs.items():
        cost = dtw(obs, v, costMatrix(obs, v))
        if cost < minimum: 
            minimun = cost
            result = k
    return result


# %% [markdown]
"""
# Experiment 2: speaker-dependent IWR

From the same speaker, pick training and test recordings

# Experiment 3: speaker-independent IWR

Select training/reference set from one speaker, test recordings from the other. 
Can you compute Prec/Recall/F1?
"""
