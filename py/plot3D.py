import numpy as np
import matplotlib.pyplot as plt

from py.positionFuncs import *



fig = plt.figure()
ax = fig.add_subplot(projection='3d')

noColorOverride = True
plotColor = 'red'
plotAlpha = 1.0


def setColor(fooColor):
    global noColorOverride, plotColor
    noColorOverride = False
    plotColor = fooColor

def plotLedPositions(inPts):
    global noColorOverride, plotColor

    if noColorOverride:
        ax.scatter(inPts[0], inPts[2], inPts[1], depthshade=False, alpha=plotAlpha)
    else:
        ax.scatter(inPts[0], inPts[2], inPts[1], depthshade=False, alpha=plotAlpha, color=plotColor)

def plotErrorLines(inVects, basePts):
    global noColorOverride, plotColor

    closestPts = getClosestPts(inVects, basePts)
    closPts = np.column_stack(closestPts)

    for ii in range(len(closPts[0])):
        if noColorOverride: plotColor = 'red'
        ax.plot([closPts[0][ii], basePts[0][ii]], [closPts[2][ii], basePts[2][ii]], [closPts[1][ii], basePts[1][ii]], color=plotColor)
        # ax.plot([0, closPts[0][ii]], [0, closPts[1][ii]], [0, closPts[2][ii]], color='yellow')

def plotCameraImageRange(camVect_X, camVect_Y, camVect_Z, bestPts):
    global noColorOverride, plotColor

    magnitude = 2000
    magnitude = max(bestPts[2])

    if noColorOverride: plotColor = 'blue'
    ax.plot([0, magnitude*max(camVect_X)], [0, magnitude], [0, magnitude*max(camVect_Y)],color=plotColor)
    ax.plot([0, magnitude*min(camVect_X)], [0, magnitude], [0, magnitude*min(camVect_Y)],color=plotColor)
    ax.plot([0, magnitude*max(camVect_X)], [0, magnitude], [0, magnitude*min(camVect_Y)],color=plotColor)
    ax.plot([0, magnitude*min(camVect_X)], [0, magnitude], [0, magnitude*max(camVect_Y)],color=plotColor)



def plotCameraVectorSections(camVect_X, camVect_Y, camVect_Z, bestPts):
    global noColorOverride, plotColor

    zMax = max(bestPts[2])
    zMin = min(bestPts[2])
    for ii in range(len(camVect_X)):
        zMin = bestPts[2][ii] -25
        zMax = bestPts[2][ii] +25

        if noColorOverride: plotColor = 'black'
        plt.plot([camVect_X[ii]*zMin, camVect_X[ii]*zMax], [zMin, zMax], [camVect_Y[ii]*zMin, camVect_Y[ii]*zMax], color=plotColor)



def plotJustLEDPos(LED_X, LED_Y, LED_Z):
    global noColorOverride, plotColor

    if noColorOverride: 
        ptCount = len(LED_X)
        plotColor = []
        for ii in range(ptCount):
            colVal = 1.0*ii/ptCount
            plotColor.append([1.0-colVal, 0.0, 0.0])

    ax.scatter(LED_X, LED_Y, LED_Z, color=plotColor, depthshade=False)



def showPlot():
    ax.set_xlabel('X')
    ax.set_zlabel('Y')
    ax.set_ylabel('Z')

    set_axes_equal(ax)
    plt.show()