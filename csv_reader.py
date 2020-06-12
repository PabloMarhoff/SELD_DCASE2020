#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from numpy import loadtxt, arange, repeat, full, zeros, unique
from numpy import array



# Extraktion von Framenummer (Spalte 0), Geräuschklasse (Spalte 1), Azi (Spalte 3) und Ele (Spalte 4) aus .csv-Datei
def csv_extractor(filepath):
    return loadtxt(open(filepath, "rb"), dtype="int32", delimiter=",", usecols=(0,1,3,4))
    

def rm_2sources_frames(rawdata):
    csvdata = zeros(rawdata.shape, "int32")
    doubles = 0
    for i, frame in enumerate(rawdata[:,0]):
        if rawdata[i-1,0] == frame:
            continue
        elif i+1 == len(rawdata):
            csvdata[i-2*doubles,:] = rawdata[i,:]
        elif rawdata[i+1,0] == frame:
            doubles = doubles + 1
        else:
            csvdata[i-2*doubles,:] = rawdata[i,:]
    return csvdata[:(len(csvdata)-2*doubles),:]

# rawdata = array([[2,324],[3,-12],[4,522],[5,63],[5,124],[6,235],[6,142],[7,74],[7,634],[8,634]], "int32")
# csvdata = zeros(rawdata.shape, "int32")
# doubles = 0
# for i, frame in enumerate(rawdata[:,0]):
#     if rawdata[i-1,0] == frame:
#         x = 23
#     elif i+1 == len(rawdata):
#         csvdata[i-2*doubles,:] = rawdata[i,:]
#     elif rawdata[i+1,0] == frame:
#         doubles = doubles + 1
#     else:
#         csvdata[i-2*doubles,:] = rawdata[i,:]
# csvdata_out = csvdata[:(len(csvdata)-2*doubles),:]





    # csvdata = full((1200,4), 666, dtype="int32")
    # csvdata[:,0] = repeat(arange(600),2)
    # rawdata = loadtxt(open(filepath, "rb"),dtype="int32",delimiter=",")
    # for index, frame in enumerate(rawdata[:,0]):
    #     if (csvdata[frame*2,1]==666):
    #         csvdata[frame*2,[1,2,3]] = rawdata[index,[1,3,4]]
    #     else:
    #         csvdata[frame*2+1,[1,2,3]] = rawdata[index,[1,3,4]]
    # return csvdata
    