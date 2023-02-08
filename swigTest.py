from copy import deepcopy
from math import cos, sin
import serial
import sys
import os
import time
import struct
import math as m
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import random as r
import statistics as st

import swig.FastFireFly as FFF

# mm
RING_RADIUS = 100
RING_HEIGHT = 30

# Load information
readFile = open('data/outData.pkl', 'rb')
# readFile = open('data/data_sideAngle.pkl', 'rb')
dataDict = pkl.load(readFile)
readFile.close()

print('Data Loaded:', end='')
for foo in dataDict: print(foo, end=', ')
print('\n')

# Load image data
imageRange = range(60)
row = np.array( dataDict['row'] )[imageRange]
col = np.array( dataDict['col'] )[imageRange]
xAngle = np.array( dataDict['xAngle'] )[imageRange]
yAngle = np.array( dataDict['yAngle'] )[imageRange]
ptSize = np.array( dataDict['ptSize'] )[imageRange]

confidenceVals = (ptSize + st.median(ptSize)) / max(ptSize + st.median(ptSize))

camVect_Z = np.ones_like(xAngle)
camVect_X = np.tan(xAngle)
camVect_Y = np.tan(yAngle)



def basePoints(row, column):
    i_set = RING_RADIUS * np.cos(col*2*np.pi/37)
    j_set = RING_RADIUS * np.sin(col*2*np.pi/37)
    k_set = (row-1) * RING_HEIGHT
    return([i_set, j_set, k_set])


if True: # Drop low signifigance points
    keepPts = np.where(ptSize > 3)
    keepPts = np.where(ptSize > 10)
    # keepPts = np.where(ptSize > 25)

    row = row[keepPts]
    col = col[keepPts]
    ptSize = ptSize[keepPts]
    confidenceVals = confidenceVals[keepPts]
    camVect_Z = camVect_Z[keepPts]
    camVect_X = camVect_X[keepPts]
    camVect_Y = camVect_Y[keepPts]


basePts = basePoints(row, col)
basePts = np.array(basePts, dtype=float)
inVects = [camVect_X, camVect_Y, camVect_Z]

# plt.scatter(camVect_X, camVect_Y)
# plt.show()

LocalizationSystem = FFF.ledLocalizationFast(basePts[0], basePts[1], basePts[2])

print(LocalizationSystem.fitPositionToVectors(camVect_X, camVect_Y, camVect_Z, range(len(camVect_Z))) )