import os
import pickle as pkl
import numpy as np

from py.positionFuncs import *
# import py.plot3D as plot3D

import swig.FastFireFly as FFF

# Find data files to load
existingFiles = os.listdir('data/')
print(existingFiles)

# Load data files
dataRuns = []
for fileName in existingFiles:
    if not '.pkl' in fileName: continue

    inFile = open("data/"+fileName, 'rb')
    readDict = pkl.load(inFile)
    inFile.close()
    dataRuns.append(readDict)

# LED Positions by index
LED_X = np.array( [112.5833025, 91.92388155, 65, 0, 0, 0, -112.5833025, -91.92388155, -65, 0, 0, 0] )
LED_Y = np.array( [0, 0, 0, -65, -91.92388155, -112.5833025, 0, 0, 0, 65, 91.92388155, 112.5833025] )
LED_Z = np.array( [68, 94.92388155, 115.5833025, 115.5833025, 94.92388155, 68, 68, 94.92388155, 115.5833025, 115.5833025, 94.92388155, 68] )

# # Set plot color to position range 
# plotColor = []
# for ii in range(len(LED_X)): plotColor.append([1.0-1.0*ii/len(LED_X), 0.0, 0.0])
# plotColor = np.array(plotColor)

# # Plot just LED positions for demo purposes
# plot3D.plotJustLEDPos(LED_X, LED_Y, LED_Z)
# plot3D.showPlot()



# accLocalizationSystem = FFF.ledLocalizationFast(LED_X, LED_Y, LED_Z)

localization = FFF.ledLocalizationFast(LED_X, LED_Y, LED_Z)
outPts = []

def testRun(dataDict):
    # Drop small points
    S = np.array(dataDict['size'])
    ptSet = np.where(S > 20)
    xAngle = np.array(dataDict['xPix'])[ptSet]
    yAngle = np.array(dataDict['yPix'])[ptSet]
    ptSize = np.array(S[ptSet], dtype=np.double)


    # # Override plot color to indicate LED index
    # pltColSubSet = plotColor[ptSet]
    # pltColSubSet = [list(foo) for foo in pltColSubSet]
    # plot3D.setColor(list(pltColSubSet))


    # Setup data arrays from camera data
    camVect_Z = np.ones_like(xAngle)
    camVect_X = np.tan(xAngle)
    camVect_Y = np.tan(yAngle)
    inVects = [camVect_X, camVect_Y, camVect_Z]
    basePts = [np.array(dataDict['LED_X'])[ptSet], np.array(dataDict['LED_Y'])[ptSet], np.array(dataDict['LED_Z'])[ptSet]]
    
    # Actually call localization
    import random
    # print(f"{camVect_X[0]} -> ", end='')
    camVect_X[0] += random.random()/1000
    # print(camVect_X[0])
    motion_best = localization.fitPositionToVectors(camVect_X, camVect_Y, camVect_Z, ptSize, range(len(camVect_Z)))
    bestPts = completeMotion(basePts, motion_best)
    outPts.append(bestPts)
    
    fooError = localization.getError()

    # if fooError < 1000:
    #     plot3D.plotAlpha = 0.5
    #     plot3D.setColor('orange')
    #     plot3D.plotLedPositions(bestPts)
    #     plot3D.setColor('red')
    #     plot3D.plotErrorLines(inVects, bestPts)
    #     plot3D.setColor('black')
    #     plot3D.plotCameraVectorSections(camVect_X, camVect_Y, camVect_Z, bestPts)
    #     plot3D.setColor('blue')
    #     plot3D.plotCameraImageRange(camVect_X, camVect_Y, camVect_Z, bestPts)
    

    if fooError > 1000: print("!!! ", end='')
    else: print("    ", end='')

    print(f"Error:{str(round(fooError, 3)).rjust(15, ' ')}   randFactor:{str(round(localization.getRandFactor(), 3)).rjust(15, ' ')}", end='')
    for foo in motion_best: print(f"{str(round(foo,5)).rjust(15, ' ')}", end='')
    print('')


for fooRun in dataRuns:
    testRun(fooRun)


knownData_X = np.array([-279.4, -101.6, 101.6, 279.4, -152.4, 0, 152.4, -279.4, -101.6, 101.6, 279.4, ])
knownData_Y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ])
knownData_Z = np.array([-228.6, -228.6, -228.6, -228.6, 0, 0, 0, 203.2, 203.2, 203.2, 203.2, ])

cameraOffset_X = 0
cameraOffset_Y = -736.6
cameraOffset_Z = 863.6


def printGrid(inData):
    for ii in range(len(inData[0])):
        print(" | ", end='')
        for jj in range(len(inData)):
            print(str(round(inData[jj][ii], 2)).rjust(8, ' '), end=' ')
    print('')

knownPts = deepcopy([knownData_X, knownData_Y, knownData_Z])
knownPts = completeMotion(knownPts, [0, 0, 0, m.tan(cameraOffset_Z/cameraOffset_Y), 0, 0])
centerDist = magnitude(np.array([cameraOffset_X, cameraOffset_Y, cameraOffset_Z]))
knownPts = completeMotion(knownPts, [0, 0, centerDist, 0, 0, 0])


# plot3D.ax.scatter(knownPts[0], knownPts[2], knownPts[1], color='green')


# plt.cla()
# for fooRun in dataRuns:
#     plt.scatter(fooRun['xPix'], fooRun['yPix'], color='blue')
# plt.show()

for foo in outPts:
    print(foo)


# plot3D.showPlot()