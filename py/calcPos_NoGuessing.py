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

row = np.array( dataDict['row'] )
col = np.array( dataDict['col'] )
xAngle = np.array( dataDict['xAngle'] )
yAngle = np.array( dataDict['yAngle'] )
ptSize = np.array( dataDict['ptSize'] )

confidenceVals = (ptSize + st.median(ptSize)) / max(ptSize + st.median(ptSize))

camVect_Z = np.ones_like(xAngle)
camVect_X = np.tan(xAngle)
camVect_Y = np.tan(yAngle)


print(f"xAngle   {min(xAngle)}   {max(xAngle)}")
print(f"yAngle   {min(yAngle)}   {max(yAngle)}")
print(f"camVect_X   {min(camVect_X)}   {max(camVect_X)}")
print(f"camVect_Y   {min(camVect_Y)}   {max(camVect_Y)}")

def basePoints(row, column):
    row = (row+2)%3
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

motion_best = [0, 0, 2000, 0, 0, 0]
error_best = testError(inVects, basePts, motion_best, ptSize)


outLog = open('DataLog.csv', 'w')
outLog.write(f"TestIteration, ModMethod, Error, X, Y, Z, Rx, Ry, Rz, \n")


posHist = []

pos_factors = [1000, 1000, 1000, 1, 1, 1]

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

    # # Get vectors from origin of motion to best points and closest points
    # closeVects = np.subtract(closestPts, motion_best[:3])
    # currVects = np.subtract(curr_pairs, motion_best[:3])
    
    # closeVects_stack = np.column_stack(closeVects)
    # currVects_stack = np.column_stack(currVects)

    # closeVects_stack = undoMotion(closeVects_stack, motion_best)
    # currVects_stack = undoMotion(currVects_stack, motion_best)



    # xRotation = np.average( np.arctan(closeVects_stack[2]-closeVects_stack[1]) - np.arctan(currVects_stack[2]-currVects_stack[1]) )
    # yRotation = np.average( np.arctan(closeVects_stack[0]-closeVects_stack[2]) - np.arctan(currVects_stack[0]-currVects_stack[2]) )
    # zRotation = np.average( np.arctan(closeVects_stack[1]-closeVects_stack[0]) - np.arctan(currVects_stack[1]-currVects_stack[0]) )
    
    # motion_best[3] += xRotation
    # motion_best[4] += yRotation
    # motion_best[5] += zRotation 

    ###



    # closestPts_base = undoMotion(closestPts, motion_best)
    # basePts_stack = np.column_stack(basePts)

    def avg_atan_drop(A, B):
        return( np.average( np.arctan(np.divide(A, B, out=np.zeros_like(A), where=B!=0) )))

    # xRotation = avg_atan_drop(closestPts_base[2],closestPts_base[1]) - avg_atan_drop(basePts_stack[2],basePts_stack[1])
    # yRotation = avg_atan_drop(closestPts_base[0],closestPts_base[2]) - avg_atan_drop(basePts_stack[0],basePts_stack[2])
    # zRotation = avg_atan_drop(closestPts_base[1],closestPts_base[0]) - avg_atan_drop(basePts_stack[1],basePts_stack[0])

    # motion_best[3] += xRotation * adjustmentFactor
    # motion_best[4] += yRotation * adjustmentFactor
    # motion_best[5] += zRotation * adjustmentFactor



    # Test rotation adjustment
    if False:
        # Get current and closest points
        currPts = completeMotion(basePts, motion_best) # Get points at current position
        curr_pairs = np.column_stack(currPts) # 
        closestPts = getClosestPts(inVects, currPts)
        closPts = np.column_stack(closestPts)

        # Get closest points to base
        closestPts_base = undoMotion(closPts, motion_best)
        basePts_stack = np.column_stack(basePts)


        xRotation = avg_atan_drop(closestPts_base[2],closestPts_base[1]) - avg_atan_drop(basePts[2],basePts[1])
        motion_best[3] += xRotation * adjustmentFactor
        yRotation = avg_atan_drop(closestPts_base[0],closestPts_base[2]) - avg_atan_drop(basePts[0],basePts[2])
        motion_best[4] += yRotation * adjustmentFactor
        zRotation = avg_atan_drop(closestPts_base[1],closestPts_base[0]) - avg_atan_drop(basePts[1],basePts[0])
        motion_best[5] += zRotation * adjustmentFactor


    # Random rotation adjustment
    # For if translation works but not rotation
    if False:
        jj = int(r.random()*3+3)
        motion_test = deepcopy(motion_best)
        motion_test[jj] += pos_factors[jj]*(2*r.random()-1)
        error_test = testError(inVects, basePts, motion_test, ptSize)

        if error_test < error_best:
            # print(f"Error reduced by ({jj}) at {testIter}: {round(error_best, 4)}   ->   {round(error_test, 4)}   now   {[round(foo, 2) for foo in motion_best]}")
            # outLog.write(f"{testIter}, {jj}, {error_best}, {motion_best[0]}, {motion_best[1]}, {motion_best[2]}, {motion_best[3]}, {motion_best[4]}, {motion_best[5]}, \n")
            motion_best = motion_test
            error_best = error_test


  
        
    ###

    # print(doRotationMatrixes())
    # cross = crossFixed(currVects, np.subtract(closestPts, curr_pairs))

    # cross_adjusted = doRotationMatrixes(np.column_stack(cross), np.array(motion_best[3:])*(-1))

    # xRotation = np.average( np.arctan(cross_adjusted[2]-cross_adjusted[1]))
    # yRotation = np.average( np.arctan(cross_adjusted[0]-cross_adjusted[2]))
    # zRotation = np.average( np.arctan(cross_adjusted[1]-cross_adjusted[0]))
    
    ###

    # print(f"{xRotation}     {yRotation}     {zRotation}    ")
    print(f"{testIter} {jj}: {round(error_best, 4)}   now   {[round(foo, 2) for foo in motion_best]}")

    # error_best = testError(inVects, basePts, motion_best, ptSize)
    outLog.write(f"{testIter}, {jj}, {error_best}, {motion_best[0]}, {motion_best[1]}, {motion_best[2]}, {motion_best[3]}, {motion_best[4]}, {motion_best[5]}, calcTranslateAndRandRotation\n")
    
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
        
        print(f"{testIter} {jj}: {round(error_best, 4)}   now   {[round(foo, 2) for foo in motion_best]}")
        # error_best = testError(inVects, basePts, motion_best, ptSize)
        outLog.write(f"{testIter}, {jj}, {error_best}, {motion_best[0]}, {motion_best[1]}, {motion_best[2]}, {motion_best[3]}, {motion_best[4]}, {motion_best[5]}, randAll\n")


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


# if True: # Display cross product
#     currPts = completeMotion(basePts, motion_best)
#     curr_pairs = np.column_stack(currPts)

#     closestPts = getClosestPts(inVects, currPts)

#     # Get vectors from origin of motion to best points and closest points
#     closeVects = np.subtract(closestPts, motion_best[:3])
#     currVects = np.subtract(curr_pairs, motion_best[:3])

#     cross = crossFixed(currVects, np.subtract(closestPts, curr_pairs))
#     cross = np.divide(cross, np.linalg.norm(cross, axis=1)[:, None])

#     cross *= 10


#     for ii in range(len(cross)):
#         ax.plot(
#             [curr_pairs[ii][0], curr_pairs[ii][0]+cross[ii][0]], 
#             [curr_pairs[ii][2], curr_pairs[ii][2]+cross[ii][2]], 
#             [curr_pairs[ii][1], curr_pairs[ii][1]+cross[ii][1]], 
#             color='blue')




plotPts(bestPts)
plotDiffs(inVects, bestPts)
# plotCamera()

axisMag = 50
axisPts = np.array([
    [axisMag,0,0,0,0,0],
    [0,0,axisMag,0,0,0],
    [0,0,0,0,axisMag,0],
])

axisPts = completeMotion(axisPts, motion_best)
plt.plot(axisPts[0][0:2], axisPts[2][0:2], axisPts[1][0:2], color='red')
plt.plot(axisPts[0][2:4], axisPts[2][2:4], axisPts[1][2:4], color='blue')
plt.plot(axisPts[0][4:6], axisPts[2][4:6], axisPts[1][4:6], color='green')

# posHist = np.column_stack(np.array(posHist))
# plt.plot(posHist[0][50:], posHist[2][50:], posHist[1][50:], color='yellow')

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