from cmath import nan
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

# Load information

readFile = open('data/outData.pkl', 'rb')
# readFile = open('data/data_sideAngle.pkl', 'rb')
dataDict = pkl.load(readFile)
readFile.close()

print('Data Loaded:')
for foo in dataDict: print(foo, end=', ')
print('\n')

row = np.array( dataDict['row'] )
col = np.array( dataDict['col'] )
xAngle = np.array( dataDict['xAngle'] )
yAngle = np.array( dataDict['yAngle'] )
ptSize = np.array( dataDict['ptSize'] )

# plt.scatter(xAngle, yAngle)
# plt.show()

confidenceVals = (ptSize + st.median(ptSize)) / max(ptSize + st.median(ptSize))

camVect_Z = np.ones_like(xAngle)
camVect_X = np.tan(xAngle)
camVect_Y = np.tan(yAngle)


print(f"xAngle   {min(xAngle)}   {max(xAngle)}")
print(f"yAngle   {min(yAngle)}   {max(yAngle)}")
print(f"camVect_X   {min(camVect_X)}   {max(camVect_X)}")
print(f"camVect_Y   {min(camVect_Y)}   {max(camVect_Y)}")

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

# TEST_COUNT = 50000
# TEST_COUNT = 50
TEST_COUNT = 500
RAND_COUNT = 500
POST_RAND_COUNT = 1500



outLog = open('data/PositionLog.csv', 'w')
# outLog.write(f"TestIteration, Error, X, Y, Z, Rx, Ry, Rz, \n")

movePts = []

motion_best = [0, 0, 2000, 0, 0, 0]
pos_factors = [1000, 1000, 1000, 1, 1, 1]

inVects_all = deepcopy(inVects)
basePts_all = deepcopy(basePts)

divCount = 2
divRange = 10
maxError_allowed = 0.012

for fooTest in range((len(inVects[0])-divRange)//divCount):
    if fooTest > 0:   
        TEST_COUNT = 200
        RAND_COUNT = 200
        POST_RAND_COUNT = 300
        
        pos_factors = [100, 100, 100, 1, 1, 1]

    # get subset
    inVects = []
    for foo in inVects_all: inVects.append(foo[fooTest*divCount : fooTest*divCount+divRange])
    
    basePts = []
    for foo in basePts_all: basePts.append(foo[fooTest*divCount : fooTest*divCount+divRange])

    error_best = testError(inVects, basePts, motion_best, ptSize)

    posHist = []


    # Do random guess correction
    for testIter in range(RAND_COUNT):
        for jj in range(6):
            motion_test = deepcopy(motion_best)
            motion_test[jj] += pos_factors[jj]*(2*r.random()-1)
            error_test = testError(inVects, basePts, motion_test, ptSize)

            if error_test < error_best:
                # print(f"Error reduced by ({jj}) at {testIter}: {round(error_best, 4)}   ->   {round(error_test, 4)}   now   {[round(foo, 2) for foo in motion_best]}")
                # outLog.write(f"{testIter}, {jj}, {error_best}, {motion_best[0]}, {motion_best[1]}, {motion_best[2]}, {motion_best[3]}, {motion_best[4]}, {motion_best[5]}, \n")
                motion_best = motion_test
                error_best = error_test    
                
                # print(f"{testIter} {jj}: {round(error_best, 4)}   now   {[round(foo, 2) for foo in motion_best]}")
                # error_best = testError(inVects, basePts, motion_best, ptSize)
                
    # Do more precise adjustments
    for testIter in range(TEST_COUNT):
        # print(f"\n{testIter}   {motion_best}")

        adjustmentFactor = (TEST_COUNT - testIter)/TEST_COUNT

        # Correct position

        bestPts = completeMotion(basePts, motion_best)
        closestPts = getClosestPts(inVects, bestPts)
        closPts = np.column_stack(closestPts)

        for fooAxis in range(3): 
            modifier = sum(closPts[fooAxis] - bestPts[fooAxis]) / len(basePts[fooAxis])
            motion_best[fooAxis] += modifier
            # print(f"{fooAxis}   {modifier}")

        # Correct Rotation
        currPts = completeMotion(basePts, motion_best)
        curr_pairs = np.column_stack(currPts)
        closestPts = getClosestPts(inVects, currPts)


        # Random rotation adjustment
        
        jj = int(r.random()*3+3)
        motion_test = deepcopy(motion_best)
        motion_test[jj] += pos_factors[jj]*(2*r.random()-1)
        error_test = testError(inVects, basePts, motion_test, ptSize)

        if error_test < error_best:
            # print(f"Error reduced by ({jj}) at {testIter}: {round(error_best, 4)}   ->   {round(error_test, 4)}   now   {[round(foo, 2) for foo in motion_best]}")
            motion_best = motion_test
            error_best = error_test


    
            
        # print(f"{testIter} {jj}: {round(error_best, 4)}   now   {[round(foo, 2) for foo in motion_best]}")

        # error_best = testError(inVects, basePts, motion_best, ptSize)
        posHist.append(deepcopy(motion_best))



    # Random changes again

    pos_factors = [100, 100, 100, 0.1, 0.1, 0.1]

    for testIter in range(RAND_COUNT):
        motion_test = deepcopy(motion_best)

        for jj in range(6):
            motion_test[jj] += pos_factors[jj]*(2*r.random()-1)
            error_test = testError(inVects, basePts, motion_test, ptSize)

        if error_test < error_best:
            # print(f"Error reduced by ({jj}) at {testIter}: {round(error_best, 4)}   ->   {round(error_test, 4)}   now   {[round(foo, 2) for foo in motion_best]}")
            # outLog.write(f"{testIter}, {jj}, {error_best}, {motion_best[0]}, {motion_best[1]}, {motion_best[2]}, {motion_best[3]}, {motion_best[4]}, {motion_best[5]}, \n")
            motion_best = motion_test
            error_best = error_test    
            
            # print(f"{testIter} {jj}: {round(error_best, 4)}   now   {[round(foo, 2) for foo in motion_best]}")
            # error_best = testError(inVects, basePts, motion_best, ptSize)

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # currPts = completeMotion(basePts, motion_best)
    # closestPts = getClosestPts(inVects, currPts)
    # closPts = np.column_stack(closestPts)
    # ax.scatter(currPts[0], currPts[2], currPts[1])
    # ax.scatter(closPts[0], closPts[2], closPts[1])
    # ax.set_xlabel('X')
    # ax.set_zlabel('Y')
    # ax.set_ylabel('Z')
    # set_axes_equal(ax)
    # plt.show()

    if error_best > maxError_allowed:
        print(f"Final error: {round(error_best, 3)}, skipping run")
        continue
    
    print(f"Final error: {round(error_best, 3)}")
    for foo in motion_best: print(round(foo,3), end = '   ')
    print('\n')

    
    movePts.append(motion_best)

    outLog.write(f"{fooTest}, {error_best}, {motion_best[0]}, {motion_best[1]}, {motion_best[2]}, {motion_best[3]}, {motion_best[4]}, {motion_best[5]}, \n")


outLog.close()


def plotPts(inPts):
    ax.scatter(inPts[0], inPts[2], inPts[1], depthshade=False, c=range(len(inPts[0])), cmap = 'gist_rainbow_r')#, s = np.round(confidenceVals*16))
    ax.plot(inPts[0], inPts[2], inPts[1])

def plotDiffs(inVects, basePts):
    closestPts = getClosestPts(inVects, basePts)
    closPts = np.column_stack(closestPts)
    # plotPts(closPts)

    for ii in range(len(closPts[0])):
        ax.plot([closPts[0][ii], basePts[0][ii]], [closPts[2][ii], basePts[2][ii]], [closPts[1][ii], basePts[1][ii]], color='red', alpha=confidenceVals[ii])

        # ax.plot([0, closPts[0][ii]], [0, closPts[1][ii]], [0, closPts[2][ii]], color='yellow')

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

plotPts(list(zip(*movePts)))


for foo in movePts: plt.plot([0,foo[0]], [0,foo[2]], [0,foo[1]], c='black', alpha=0.3)


ax.set_xlabel('X')
ax.set_zlabel('Y')
ax.set_ylabel('Z')

# set_axes_equal(ax)

print(f"Total points: {len(inVects_all[0])}")

plt.show()