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

from positionFuncs import *

# mm
RING_RADIUS = 100
RING_HEIGHT = 30

readFile = open('data/outData.pkl', 'rb')
# readFile = open('data/data_sideAngle.pkl', 'rb')
dataDict = pkl.load(readFile)
readFile.close()

row = np.array( dataDict['row'] )
col = np.array( dataDict['col'] )
xAngle = np.array( dataDict['xAngle'] )
yAngle = np.array( dataDict['yAngle'] )
ptSize = np.array( dataDict['ptSize'] )

confidenceVals = (ptSize + st.median(ptSize)) / max(ptSize + st.median(ptSize))

# print(f"row:{row}")
# print(f"col:{col}")

camVect_Z = np.ones_like(xAngle)
# print(f"camVect_Z:{camVect_Z}")

camVect_X = np.tan(xAngle)
# print(f"camVect_X:{camVect_X}")

camVect_Y = np.tan(yAngle)
# print(f"camVect_Y:{camVect_Y}")


print(f"xAngle   {min(xAngle)}   {max(xAngle)}")
print(f"yAngle   {min(yAngle)}   {max(yAngle)}")
print(f"camVect_X   {min(camVect_X)}   {max(camVect_X)}")
print(f"camVect_Y   {min(camVect_Y)}   {max(camVect_Y)}")

def basePoints(row, column):
    row = (row+2)%3
    i_set = RING_RADIUS * np.cos(col*2*np.pi/37)
    j_set = RING_RADIUS * np.sin(col*2*np.pi/37)
    k_set = row * RING_HEIGHT
    return([i_set, j_set, k_set])


keepPts = np.where(ptSize > 10)

row = row[keepPts]
col = col[keepPts]
ptSize = ptSize[keepPts]
confidenceVals = confidenceVals[keepPts]
camVect_Z = camVect_Z[keepPts]
camVect_X = camVect_X[keepPts]
camVect_Y = camVect_Y[keepPts]

basePts = basePoints(row, col)
inVects = [camVect_X, camVect_Y, camVect_Z]


TEST_COUNT = 2000
# TEST_COUNT = 500000

motion_best = [500, 500, 3000, 0, 0, 0]
error_best = testError(inVects, basePts, motion_best, ptSize)

pos_factors = [100, 100, 100, 1, 1, 1]

outLog = open('DataLog.csv', 'w')
outLog.write(f"TestIteration, ModMethod, Error, X, Y, Z, Rx, Ry, Rz, \n")

for testIter in range(TEST_COUNT):
    for jj in range(len(motion_best)):
        motion_test = deepcopy(motion_best)
        motion_test[jj] += pos_factors[jj]*(2*r.random()-1)
        error_test = testError(inVects, basePts, motion_test, ptSize)

        if error_test < error_best:
            print(f"Error reduced by ({jj}) at {testIter}: {round(error_best, 4)}   ->   {round(error_test, 4)}   now   {[round(foo, 2) for foo in motion_best]}")
            outLog.write(f"{testIter}, {jj}, {error_best}, {motion_best[0]}, {motion_best[1]}, {motion_best[2]}, {motion_best[3]}, {motion_best[4]}, {motion_best[5]}, \n")
            motion_best = motion_test
            error_best = error_test

outLog.close()

print(f"Final error: {round(error_best, 2)}, at {[round(foo, 3) for foo in motion_best]}")

bestPts = completeMotion(basePts, motion_best)

def plotPts(inPts):
    ax.scatter(inPts[0], inPts[2], inPts[1], depthshade=False, s = np.round(confidenceVals*16))

def plotDiffs(inVects, basePts):
    closestPts = getClosestPts(inVects, basePts)
    closPts = np.column_stack(closestPts)

    # plotPts(closPts)

    for ii in range(len(closPts[0])):
        ax.plot([closPts[0][ii], basePts[0][ii]], [closPts[2][ii], basePts[2][ii]], [closPts[1][ii], basePts[1][ii]], color='red', alpha=confidenceVals[ii])

        # ax.plot([0, closPts[0][ii]], [0, closPts[1][ii]], [0, closPts[2][ii]], color='yellow')

def plotCamera():
    global camVect_X
    global camVect_Y
    global camVect_Z
    magnitude = 2000
    magnitude = max(bestPts[2])
    plt.plot([0, magnitude*max(camVect_X)], [0, magnitude], [0, magnitude*max(camVect_Y)],color='blue')
    plt.plot([0, magnitude*min(camVect_X)], [0, magnitude], [0, magnitude*min(camVect_Y)],color='blue')
    plt.plot([0, magnitude*max(camVect_X)], [0, magnitude], [0, magnitude*min(camVect_Y)],color='blue')
    plt.plot([0, magnitude*min(camVect_X)], [0, magnitude], [0, magnitude*max(camVect_Y)],color='blue')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

plotPts(bestPts)
plotDiffs(inVects, bestPts)
# plotCamera()

zMax = max(bestPts[2])
zMin = min(bestPts[2])

for ii in range(len(camVect_X)):
    zMin = bestPts[2][ii] -25
    zMax = bestPts[2][ii] +25
    plt.plot([camVect_X[ii]*zMin, camVect_X[ii]*zMax], [zMin, zMax], [camVect_Y[ii]*zMin, camVect_Y[ii]*zMax], color='black', alpha=confidenceVals[ii])

ax.set_xlabel('X')
ax.set_zlabel('Y')
ax.set_ylabel('Z')

set_axes_equal(ax)

plt.show()