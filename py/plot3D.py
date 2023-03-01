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

def plotCoords(inPts):
    global noColorOverride, plotColor, ax

    if noColorOverride:
        ax.scatter(inPts[1], inPts[0], inPts[2], depthshade=False)
    else:
        ax.scatter(inPts[1], inPts[0], inPts[2], depthshade=False, color=plotColor)

def plotErrorLines(inVects, basePts):
    global noColorOverride, plotColor, ax

    closestPts = getClosestPts(inVects, basePts)
    closPts = np.column_stack(closestPts)
    
    # for ii in range(len(inVects)):
    #     print(f"vect:({inVects[0][ii]}, {inVects[1][ii]}, {inVects[2][ii]})   basePts:({basePts[0][ii]}, {basePts[1][ii]}, {basePts[2][ii]})   closPts:{closPts[0][ii]}, {closPts[1][ii]}, {closPts[2][ii]}")

    for ii in range(len(closPts[0])):
        if noColorOverride: plotColor = 'red'
        ax.plot([closPts[1][ii], basePts[1][ii]], [closPts[0][ii], basePts[0][ii]], [closPts[2][ii], basePts[2][ii]], color=plotColor)
        # ax.plot([0, closPts[0][ii]], [0, closPts[1][ii]], [0, closPts[2][ii]], color='yellow')

def plotCameraImageRange(camVect_X, camVect_Y, camVect_Z, bestPts):
    global noColorOverride, plotColor, ax

    magnitude = 2000
    magnitude = max(bestPts[2])

    if noColorOverride: plotColor = 'blue'
    ax.plot([0, magnitude*max(camVect_X)], [0, magnitude], [0, magnitude*max(camVect_Y)],color=plotColor)
    ax.plot([0, magnitude*min(camVect_X)], [0, magnitude], [0, magnitude*min(camVect_Y)],color=plotColor)
    ax.plot([0, magnitude*max(camVect_X)], [0, magnitude], [0, magnitude*min(camVect_Y)],color=plotColor)
    ax.plot([0, magnitude*min(camVect_X)], [0, magnitude], [0, magnitude*max(camVect_Y)],color=plotColor)



def plotCamToCenter(position):
    global noColorOverride, plotColor, ax
    ax.plot([0, position[1]], [0, position[0]], [0, position[2]], color=plotColor)



def plotCameraVectorSections(camVects, bestPts):
    global noColorOverride, plotColor, ax

    xMax = max(bestPts[0])
    xMin = min(bestPts[0])
    for ii in range(len(camVects[0])):
        xMin = bestPts[0][ii] -25
        xMax = bestPts[0][ii] +25

        if noColorOverride: plotColor = 'black'
        plt.plot([camVects[1][ii]*xMin, camVects[1][ii]*xMax], [xMin, xMax], [camVects[2][ii]*xMin, camVects[2][ii]*xMax], color=plotColor)



def plotJustLEDPos(LED_X, LED_Y, LED_Z):
    global noColorOverride, plotColor, ax

    if noColorOverride: 
        ptCount = len(LED_X)
        plotColor = []
        for ii in range(ptCount):
            colVal = 1.0*ii/ptCount
            plotColor.append([1.0-colVal, 0.0, 0.0])

    ax.scatter(LED_Y, LED_Z, LED_X, color=plotColor, depthshade=False)



def showPlot():
    global ax
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')

    set_axes_equal(ax)
    plt.show()