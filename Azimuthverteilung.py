#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from os import scandir, path
from parameter import AUDIO_TEST, CSV_DIR
from numpy import loadtxt, zeros
import matplotlib.pyplot as plt

# Elevation: Bei Usecols=4
# Azimuth: Bei Usecols=3

csv_frames = []
with scandir(CSV_DIR) as files:
    for csvfile in files:
        csv_frames.extend(loadtxt(open(csvfile.path, "rb"), dtype="int32", delimiter=",", usecols=4)[:])
histvals = plt.hist(csv_frames, bins=67) #36 #10(default)
plt.show()