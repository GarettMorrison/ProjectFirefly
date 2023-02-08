import os
import pickle as pkl
import numpy as np
import statistics as st

from py.positionFuncs import *

import swig.FastFireFly as FFF

doPlot3d = True
if doPlot3d: import py.plot3D as plot3D



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


    for foo in readDict:
        print(f"{foo}:{len(readDict[foo])}")

# LED Positions by index
LED_X = np.array( [ 112.5833025, 91.92388155, 65, 32.5, 45.96194078, 56.29165125, -56.29165125, -45.96194078, -32.5, -65, -91.92388155, -112.5833025, -56.29165125, -45.96194078, -32.5, 32.5, 45.96194078, 56.29165125, ] )
LED_Y = np.array( [ 65, 91.92388155, 112.5833025, 112.5833025, 91.92388155, 65, 65, 91.92388155, 112.5833025, 112.5833025, 91.92388155, 65, 65, 91.92388155, 112.5833025, 112.5833025, 91.92388155, 65, ] )
LED_Z = np.array( [ 0, 0, 0, -56.29165125, -79.60841664, -97.5, -97.5, -79.60841664, -56.29165125, 0, 0, 0, 97.5, 79.60841664, 56.29165125, 56.29165125, 79.60841664, 97.5, ] )


# # Set plot color to position range 
# plotColor = []
# for ii in range(len(LED_X)): plotColor.append([1.0-1.0*ii/len(LED_X), 0.0, 0.0])
# plotColor = np.array(plotColor)

# # Plot just LED positions for demo purposes
# plot3D.plotJustLEDPos(LED_X, LED_Z, LED_Y)
# plot3D.showPlot()
# exit()



# accLocalizationSystem = FFF.ledLocalizationFast(LED_X, LED_Y, LED_Z)

localization = FFF.ledLocalizationFast(LED_X, LED_Y, LED_Z)
outPts = []

def testRun(dataDict):
    # Drop small points
    S = np.array(dataDict['size'])
    
    # ptSet = np.where(S >= st.median(S))
    ptSet = np.where(S >= 0)
    xAngle = np.array(dataDict['xPix'])[ptSet]
    yAngle = np.array(dataDict['yPix'])[ptSet]
    ptSize = np.array(S[ptSet], dtype=np.double)/max(S)



    # ptSet = np.where(S >= 0)
    # xAngle = np.array(dataDict['xPix'])
    # yAngle = np.array(dataDict['yPix']) -0.8645972344
    # ptSize = np.power(np.array(S, dtype=np.double)/max(S), np.full(S.shape, 2))



    # # Override plot color to indicate LED index
    # pltColSubSet = plotColor[ptSet]
    # pltColSubSet = [list(foo) for foo in pltColSubSet]
    # plot3D.setColor(list(pltColSubSet))


    # Setup data arrays from camera data
    camVect_X = np.tan(xAngle)
    camVect_Y = np.tan(yAngle)
    camVect_Z = np.ones_like(xAngle)
    inVects = [camVect_X, camVect_Y, camVect_Z]
    basePts = [np.array(dataDict['LED_X'])[ptSet], np.array(dataDict['LED_Y'])[ptSet], np.array(dataDict['LED_Z'])[ptSet]]
    

    
    # print(yAngle)
    # print(yAngle-0.8645972344)
    # print(np.tan(yAngle))
    # print(np.tan(yAngle-0.8645972344))
    # exit()


    # Actually call localization
    motion_best = localization.fitPositionToVectors(camVect_X, camVect_Y, camVect_Z, ptSize, range(len(camVect_Z)))
    bestPts = completeMotion(basePts, motion_best)
    outPts.append(bestPts)

    
    
    
    fooError = localization.getError()

    if True:
    # if fooError < 10000:
    # if doPlot3d: 
        showPts = [np.array(dataDict['LED_X']), np.array(dataDict['LED_Y']), np.array(dataDict['LED_Z'])]
        showPts = completeMotion(showPts, motion_best)
        plot3D.plotAlpha = 0.5
        plot3D.setColor('orange')
        plot3D.plotLedPositions(showPts)
        plot3D.setColor('red')
        plot3D.plotErrorLines(inVects, bestPts)
        plot3D.setColor('black')
        plot3D.plotCameraVectorSections(camVect_X, camVect_Y, camVect_Z, bestPts)
        # plot3D.setColor('blue')
        # plot3D.plotCameraImageRange(camVect_X, camVect_Y, camVect_Z, bestPts)
    

    if fooError > 1000: print("!!! ", end='')
    else: print("    ", end='')

    print(f"Error:{str(round(fooError, 3)).rjust(10, ' ')}   randFactor:{str(round(localization.getRandFactor(), 3)).rjust(10, ' ')}", end='')
    for foo in motion_best: print(f"{str(round(foo,5)).rjust(15, ' ')}", end='')
    print('')


for fooRun in dataRuns:
    testRun(fooRun)

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



# for foo in outPts:
#     print(foo)



if doPlot3d: plot3D.showPlot()