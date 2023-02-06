import os
import pickle as pkl
import numpy as np

existingFiles = os.listdir('data/')
print(existingFiles)

dataRuns = []
for fileName in existingFiles:
    print(fileName)
    inFile = open("data/"+fileName, 'rb')
    readDict = pkl.load(inFile)
    inFile.close()
    dataRuns.append(readDict)



import matplotlib.pyplot as plt

# for fooRun in dataRuns:
    # print(f"\n\n{foo}")

    # for ii in range(len(fooRun['xPos'])):
    #     plt.plot(fooRun['xPos'][ii])

    # plt.scatter(fooRun['xPos'], fooRun['yPos'], s=np.array(fooRun['size'])/2)#, =np.arange(0.0, 1.0, 1.0/len(fooRun['size'])))

# plt.show()
# exit()






LED_X = np.array( [112.5833025, 91.92388155, 65, 0, 0, 0, -112.5833025, -91.92388155, -65, 0, 0, 0] )
LED_Y = np.array( [0, 0, 0, -65, -91.92388155, -112.5833025, 0, 0, 0, 65, 91.92388155, 112.5833025] )
LED_Z = np.array( [68, 94.92388155, 115.5833025, 115.5833025, 94.92388155, 68, 68, 94.92388155, 115.5833025, 115.5833025, 94.92388155, 68] )



fig = plt.figure()
ax = fig.add_subplot(projection='3d')

        
from py.positionFuncs import *

for foo in dataRuns:
    S = np.array(foo['size'])
    ptSet = np.where(S > 10)

    xAngle = np.array(foo['xPos'])[ptSet]
    yAngle = np.array(foo['yPos'])[ptSet]
    ptSize = S[ptSet]
    
    camVect_Z = np.ones_like(xAngle)
    camVect_X = np.tan(xAngle)
    camVect_Y = np.tan(yAngle)

    inVects = [camVect_X, camVect_Y, camVect_Z]
    basePts = [LED_X[ptSet], LED_Y[ptSet], LED_Z[ptSet]]

    print('\n')
    print(f"ptSize:{len(ptSize)}")
    print(f"camVect_Z:{len(camVect_Z)}")
    print(f"camVect_X:{len(camVect_X)}")
    print(f"camVect_Y:{len(camVect_Y)}")
    print('\n')

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
        ax.scatter(inPts[0], inPts[2], inPts[1], depthshade=False)

    def plotDiffs(inVects, basePts):
        closestPts = getClosestPts(inVects, basePts)
        closPts = np.column_stack(closestPts)

        # plotPts(closPts)

        for ii in range(len(closPts[0])):
            ax.plot([closPts[0][ii], basePts[0][ii]], [closPts[2][ii], basePts[2][ii]], [closPts[1][ii], basePts[1][ii]], color='red')

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


    plotPts(bestPts)
    plotDiffs(inVects, bestPts)
    # plotCamera()

    zMax = max(bestPts[2])
    zMin = min(bestPts[2])

    for ii in range(len(camVect_X)):
        zMin = bestPts[2][ii] -25
        zMax = bestPts[2][ii] +25
        plt.plot([camVect_X[ii]*zMin, camVect_X[ii]*zMax], [zMin, zMax], [camVect_Y[ii]*zMin, camVect_Y[ii]*zMax], color='black' )

    ax.set_xlabel('X')
    ax.set_zlabel('Y')
    ax.set_ylabel('Z')

set_axes_equal(ax)
plt.show()