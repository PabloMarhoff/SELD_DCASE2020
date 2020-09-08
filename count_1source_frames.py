#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 13:43:19 2020

@author: pablo
"""
from csv_reader import rm_2sources_frames, csv_extractor
from os import scandir

csv_files = "TAU-NIGENS/metadata_dev/"

counter_1source_frames = 0
with scandir(csv_files) as files:
    for file in files:
        rawdata = csv_extractor(file.path)
        for i, frame in enumerate(rawdata[:,0]):
            if rawdata[i-1,0] == frame:
                continue
            elif i+1 == len(rawdata):
                counter_1source_frames += 1
            elif rawdata[i+1,0] == frame:
                continue
            else:
                counter_1source_frames += 1
print(counter_1source_frames)