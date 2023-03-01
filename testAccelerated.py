import os
import pickle as pkl
import numpy as np
import statistics as st
import sys

from py.positionFuncs import *

# sys.path.insert(0, '/swig')
import swig.FastFireFly as FFF


DO_PLOT_3D = True
DO_PLOT_3D = False
if DO_PLOT_3D: import py.plot3D as plot3D

DO_PLOT_2D = True
# DO_PLOT_2D = False
if DO_PLOT_2D: 
    import matplotlib.pyplot as plt
    import random
    
import colorsys
# def hsv2rgb(h,s,v):
#     return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h,s,v))

DO_FILEOUT = True


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
LED_X = np.array( [ -112.5833025,-91.92388155,-65,-32.5,-45.96194078,-56.29165125,56.29165125,45.96194078,32.5,65,91.92388155,112.5833025,56.29165125,45.96194078,32.5,-32.5,-45.96194078,-56.29165125 ] )
LED_Z = np.array( [ 65, 91.92388155, 112.5833025, 112.5833025, 91.92388155, 65, 65, 91.92388155, 112.5833025, 112.5833025, 91.92388155, 65, 65, 91.92388155, 112.5833025, 112.5833025, 91.92388155, 65 ] )
LED_Y = np.array( [ 0, 0, 0, -56.29165125, -79.60841664, -97.5, -97.5, -79.60841664, -56.29165125, 0, 0, 0, 97.5, 79.60841664, 56.29165125, 56.29165125, 79.60841664, 97.5 ] )


ptCount = len(LED_X)
plotColor = []
for ii in range(ptCount):
    colVal = 1.0*ii/ptCount
    plotColor.append([1.0-colVal, 0.0, 0.0])

# # Set plot color to position range 
# plotColor = []
# for ii in range(len(LED_X)): plotColor.append([1.0-1.0*ii/len(LED_X), 0.0, 0.0])
# plotColor = np.array(plotColor)

# # Plot just LED positions for demo purposes
# plot3D.plotJustLEDPos(LED_X, LED_Z, LED_Y)
# plot3D.showPlot()
# exit()


defaultPosition = [1000, 0, 0, 0, 0, 0]
prevMotion = defaultPosition

if DO_FILEOUT: writePoints = open('data/PositionLog.csv', 'w')


plotColSet = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
runIndex = -1

plotCount = len(dataRuns)

def testRun(dataDict):
    global plotColSet, runIndex, prevMotion, defaultPosition, plotCount
    runIndex += 1
    ptAngs = np.array([dataDict['xAng'], dataDict['yAng']], dtype=np.double)
    # ptAngs[1] -= m.radians(30)
    zAngle = ptAngs[0]
    yAngle = ptAngs[1]
    ptCount = len(yAngle)
    ptIndex = np.array(dataDict['index'], dtype=np.uint32)

    global plotColor
    fooPlotColor = np.array(plotColor)[ptIndex]

    # Actually call localization
    localization = FFF.ledLocalizationFast([LED_X, LED_Y, LED_Z], prevMotion)
    motion_best = localization.fitData_imageCentric(ptAngs, ptIndex.tolist(), 1000)
    motion_best = localization.fitData_3D(ptAngs, ptIndex.tolist(), 10000)

    testAttempts = 0
    maxTestAttempts = 10
    
    testParameters = [
        [1000, 40000, 1000],
        [500, 50000, 0],
        [250, 50000, 0],
        [100, 50000, 0],
        [50, 50000, 0],
        [6, 50000, 0],
    ]

    for fooParams in testParameters:
        reqError = fooParams[0]
        fitAttempts = fooParams[1]
        imageCentricAttempts = fooParams[2]
        while localization.getError()/ptCount > reqError and testAttempts < maxTestAttempts:
            print(f"Error > {reqError}: {localization.getError()/ptCount}")
            # localization = FFF.ledLocalizationFast([LED_X, LED_Y, LED_Z], prevMotion)
            if imageCentricAttempts > 0: motion_best = localization.fitData_imageCentric(ptAngs, ptIndex.tolist(), imageCentricAttempts)
            motion_best = localization.fitData_3D(ptAngs, ptIndex.tolist(), fitAttempts)
            testAttempts += 1
        prevMotion = motion_best

        if testAttempts > maxTestAttempts: break


    fooError = localization.getError()/ptCount

    if DO_PLOT_2D:
        # plt.scatter(fooAngles[1], fooAngles[0], color=fooPlotColor)
        plt.scatter(ptAngs[0], ptAngs[1], alpha=0.5, color='red')
        
    if testAttempts >= maxTestAttempts: 
        print("Exceeded max attempts :(")
        return


    if DO_PLOT_2D:
        fooAngles = localization.getTestAngles()
        # fooPlotColor = colorsys.hsv_to_rgb(random.random(), random.random()/2 +0.5, random.random()/2 +0.5)
        fooPlotColor = colorsys.hsv_to_rgb(runIndex/plotCount, 1.0, 0.7)

        plt.scatter(fooAngles[1], fooAngles[0], alpha=0.5, color='blue')
        # print(f"Color: {fooPlotColor}\n")

        # plt.scatter(fooAngles[1], fooAngles[0], color=fooPlotColor)
        # plt.scatter(ptAngs[1], ptAngs[0], alpha=0.5, color=plotColSet[plotColIndex])
        
    elif DO_PLOT_3D:        
        # Print Run info
        fooCoords = localization.getLEDs()
        realPts = np.array(fooCoords)[:, ptIndex]

        vectSet = np.array(localization.get_ang_line_set())
        camVects = np.array([np.ones_like(vectSet[0]), vectSet[0], vectSet[1]] )

        
        plot3D.setColor(colorsys.hsv_to_rgb(runIndex/plotCount, 1.0, 0.7) )
        plot3D.plotCoords(realPts)

        plot3D.noColorOverride = True
        plot3D.plotErrorLines(camVects, realPts)
        # plot3D.plotCamToCenter(motion_best)
        plot3D.plotCameraVectorSections(camVects, realPts)
            
    if DO_FILEOUT:
        # writePoints.write(f"{writeIndex},{motion_best[1]},{motion_best[0]},{motion_best[2]},{motion_best[4]},{motion_best[3]},{motion_best[5]},\n")
        writePoints.write(f"{runIndex},{motion_best[0]},{motion_best[1]},{motion_best[2]},{motion_best[3]},{motion_best[4]},{motion_best[5]},\n")

    # Print Run info
    print(f"Error:{str(round(fooError, 3)).rjust(10, ' ')}   randFactor:{str(round(localization.getRandFactor(), 5)).rjust(10, ' ')}", end='')
    for foo in motion_best: print(f"{str(round(foo,5)).rjust(15, ' ')}", end='')
    print('')

        

for fooRun in dataRuns:
    testRun(fooRun)


if DO_FILEOUT: 
    writePoints.close()


if DO_PLOT_2D:
    plt.title("Position of LEDs in Image")
    plt.legend()
    # plt.xlim((-1, 1))
    # plt.ylim((-9/16, 9/16))
    
    # plt.xlim((-0.5, 0.5))
    # plt.ylim((-0.2, 0.2))

    plt.show()
elif DO_PLOT_3D:
    plt.title("LED Positions in 3D Space")
    plot3D.showPlot()