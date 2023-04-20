import numpy as np
import matplotlib.pyplot as plt
import math as m

import py.positionFuncs as pf


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
        ax.scatter(inPts[2], inPts[0], inPts[1], depthshade=False)
    else:
        ax.scatter(inPts[2], inPts[0], inPts[1], depthshade=False, color=plotColor)

def plotErrorLines(inVects, basePts):
    global noColorOverride, plotColor, ax

    closestPts = pf.getClosestPts(inVects, basePts)
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



def plotLED_positions(positions):
    plotJustLEDPos(positions[0], positions[1], positions[2])


def plotJustLEDPos(LED_X, LED_Y, LED_Z):
    global noColorOverride, plotColor, ax

    if noColorOverride: 
        ptCount = len(LED_X)
        plotColor = []
        for ii in range(ptCount):
            colVal = 1.0*ii/ptCount
            plotColor.append([1.0-colVal, 0.0, 0.0])

    ax.scatter(LED_Z, LED_X, LED_Y, color=plotColor, depthshade=False)



def plotPosTransform(inMotion, plotRad=10, doSurface=False):
    plotSet = [[0], [0], [0]]
    for ii in range(7):
        plotSet[2].append(plotRad*m.sin(ii*m.pi*2/6))
        plotSet[0].append(plotRad*m.cos(ii*m.pi*2/6))
        plotSet[1].append(0)

    
    if doSurface:
        # X, Z = np.meshgrid(plotSet[1], plotSet[0])
        # Y, F = np.meshgrid(plotSet[2], plotSet[2])

        X = np.array([
            [plotRad/2, plotRad, plotRad/2, ],
            [-plotRad/2, -plotRad, -plotRad/2, ],
        ])
        
        Y = np.array([
            [0, 0, 0],
            [0, 0, 0],
        ])
        
        Z = np.array([
            [-plotRad*m.sqrt(2), 0, plotRad*m.sqrt(2)],
            [-plotRad*m.sqrt(2), 0, plotRad*m.sqrt(2)],
        ])

        arrLen = len(X)

        ptSet = np.array([X.ravel(), Y.ravel(), Z.ravel(), ])
        # ravelData = pf.completeMotion(ptSet, inMotion[:3]+[0, 0, 0])
        ravelData = pf.completeMotion(ptSet, inMotion)
        # ravelData = ptSet
        
        X = ravelData[0].reshape((2, 3))
        Y = ravelData[1].reshape((2, 3))
        Z = ravelData[2].reshape((2, 3))

        print(f"X:{X}")
        print(f"Y:{Y}")
        print(f"Z:{Z}")
        print("\n\n\n\n\n")

        # Plot dims are ZYX
        ax.plot_surface(Z, X, Y, cmap='coolwarm')

    else:
        plotSet = pf.completeMotion(np.array(plotSet), inMotion)
        if noColorOverride:
            ax.plot(plotSet[2], plotSet[0], plotSet[1])
        else:
            ax.plot(plotSet[2], plotSet[0], plotSet[1], color=plotColor)



def showPlot():
    global ax
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')

    pf.set_axes_equal(ax)
    plt.show()
