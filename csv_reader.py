#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy import loadtxt, arange, repeat, full


# TODO: 2) Arrayformat ändern? (600,(2,5)) <-- eher nicht...

# Extraktion von Geräuschklasse (Spalte 1), Azi (Spalte 3) und Ele (Spalte 4) aus .csv-Datei
def csv_extractor(filepath):
    csvdata = full((1200,4), 666, dtype="int32")
    csvdata[:,0] = repeat(arange(600),2)
    rawdata = loadtxt(open(filepath, "rb"),dtype="int32",delimiter=",")
    for index, frame in enumerate(rawdata[:,0]):
        if (csvdata[frame*2,1]==666):
            csvdata[frame*2,[1,2,3]] = rawdata[index,[1,3,4]]
        else:
            csvdata[frame*2+1,[1,2,3]] = rawdata[index,[1,3,4]]
    return csvdata
    