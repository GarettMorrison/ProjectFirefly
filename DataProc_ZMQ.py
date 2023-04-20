import os
import pickle as pkl
import numpy as np
import statistics as st
import sys

from py.positionFuncs import *

# sys.path.insert(0, '/swig')
import swig.FastFireFly as FFF

DO_FILEOUT = True
CHECK_DUR =0.5


# LED Positions by index
# LED_X = np.array( [ -112.5833025,-91.92388155,-65,-32.5,-45.96194078,-56.29165125,56.29165125,45.96194078,32.5,65,91.92388155,112.5833025,56.29165125,45.96194078,32.5,-32.5,-45.96194078,-56.29165125 ] )
# LED_Z = np.array( [ 65, 91.92388155, 112.5833025, 112.5833025, 91.92388155, 65, 65, 91.92388155, 112.5833025, 112.5833025, 91.92388155, 65, 65, 91.92388155, 112.5833025, 112.5833025, 91.92388155, 65 ] )
# LED_Y = np.array( [ 0, 0, 0, -56.29165125, -79.60841664, -97.5, -97.5, -79.60841664, -56.29165125, 0, 0, 0, 97.5, 79.60841664, 56.29165125, 56.29165125, 79.60841664, 97.5 ] )

LED_X = np.array( [ 0, 0, 0, 28.14582562, 39.80420832, 48.75, 48.75, 39.80420832, 28.14582562, 0, 0, 0, -48.75, -39.80420832, -28.14582562, -28.14582562, -39.80420832, -48.75 ] )
LED_Y = np.array( [ 32.5, 45.96194078, 56.29165125, 56.29165125, 45.96194078, 32.5, 32.5, 45.96194078, 56.29165125, 56.29165125, 45.96194078, 32.5, 32.5, 45.96194078, 56.29165125, 56.29165125, 45.96194078, 32.5 ] )
LED_Z = np.array( [ -56.29165125, -45.96194078, -32.5, -16.25, -22.98097039, -28.14582562, 28.14582562, 22.98097039, 16.25, 32.5, 45.96194078, 56.29165125, 28.14582562, 22.98097039, 16.25, -16.25, -22.98097039, -28.14582562 ] )



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

writePoints = open("data/PositionLog.csv", "w+")
writePoints.close()

def calculatePosition(ptIndex, point_camXAngle, point_camYAngle):
    global plotColSet, runIndex, prevMotion, defaultPosition
    ptAngs = np.array([point_camXAngle, point_camYAngle], dtype=np.double)

    if ptCount == 0:
        print("NO DATA")
        return([0])
    
    # Localization coordinate notes
    # camXAngle & camYAngle refer to angle from center of camera. 
    #   camXAngle is horizontal, and positive to the right
    #   camYAngle is vertical, and positive facing up
    #
    # XYZ has it's origin at the camera
    # X is perpendicular to the image, and increases as we move forward into the image
    # Y is the vertical in the image, and increases towards the top
    # Z is the horizontal in the image, and increases towards the right

    # Actually call localization
    localization = FFF.ledLocalizationFast([LED_X, LED_Y, LED_Z], prevMotion)
    motion_best = localization.fitData_imageCentric(ptAngs, ptIndex.tolist(), 1000)
    motion_best = localization.fitData_3D(ptAngs, ptIndex.tolist(), 10000)

    testAttempts = 0
    maxTestAttempts = 6
    
    testParameters = [
        [1000, 40000, 1000],
        [500, 50000, 0],
        [250, 50000, 0],
        [100, 50000, 0],
        [20, 50000, 0],
        [4, 50000, 0],
        [0.5, 50000, 0],
    ]

    for fooParams in testParameters:
        reqError = fooParams[0]
        fitAttempts = fooParams[1]
        imageCentricAttempts = fooParams[2]
        while localization.getError()/ptCount > reqError and testAttempts < maxTestAttempts:
            print(f"Error > {reqError}: {localization.getError()/ptCount}")
            # localization = FFF.ledLocalizationFast([LED_X, LED_Y, LED_Z], prevMotion)
            # if imageCentricAttempts > 0: motion_best = localization.fitData_imageCentric(ptAngs, ptIndex.tolist(), imageCentricAttempts)
            motion_best = localization.fitData_3D(ptAngs, ptIndex.tolist(), fitAttempts)
            testAttempts += 1
        prevMotion = motion_best

        if testAttempts > maxTestAttempts: break


    fooError = localization.getError()/ptCount

    if testAttempts >= maxTestAttempts and localization.getError()/ptCount > testParameters[-2][0]: 
        print("Exceeded max attempts :(")
        return([1])
    

    if DO_FILEOUT:
        writePoints = open('data/PositionLog.csv', 'a')
        # writePoints.write(f"{writeIndex},{motion_best[1]},{motion_best[0]},{motion_best[2]},{motion_best[4]},{motion_best[3]},{motion_best[5]},\n")
        writePoints.write(f"{motion_best[0]},{motion_best[1]},{motion_best[2]},{motion_best[3]},{motion_best[4]},{motion_best[5]},\n")
        writePoints.close()

    # Print Run info
    print(f"Error:{str(round(fooError, 3)).rjust(10, ' ')}   randFactor:{str(round(localization.getRandFactor(), 5)).rjust(10, ' ')}", end='')
    for foo in motion_best: print(f"{str(round(foo,5)).rjust(15, ' ')}", end='')
    print('')

    return(motion_best)




import zmq

# nameserver for windows, cat /etc/resolv.conf
NameServer = "172.20.80.1"
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.connect(f"tcp://{NameServer}:5555")

np.set_printoptions(suppress=True)

print("Ready for data!")
startPos = []
while True:
    # Read input data
    data = socket.recv()
    # Get the size of the received buffer
    buffer_size = len(data)
    # Determine the number of elements based on the size of the data type
    num_elements = buffer_size // (np.dtype(np.double).itemsize * 2 + np.dtype(np.uint32).itemsize)
    
    # Convert the buffer to a NumPy array
    index_size = np.dtype(np.uint32).itemsize*num_elements
    vect_size = np.dtype(np.double).itemsize*num_elements
    
    point_indices = np.frombuffer(data[:index_size], dtype=np.uint32)
    point_camXAngle = np.frombuffer(data[index_size : index_size+vect_size], dtype=np.double)
    point_camYAngle = np.frombuffer(data[index_size+vect_size : index_size+vect_size*2], dtype=np.double)


    print(f"point_indices:{point_indices}")
    print(f"point_camXAngle:{point_camXAngle}")
    print(f"point_camYAngle:{point_camYAngle}")


    posList = calculatePosition(point_indices, point_camXAngle, point_camYAngle)
    outPosition = np.array(posList, dtype=np.double)
    
    print(f"Sending Position: {outPosition}")

    socket.send(outPosition.tobytes())