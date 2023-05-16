import os
import pickle as pkl
import numpy as np
import statistics as st
import sys

from py.positionFuncs import *
import roverConfig as rc

# sys.path.insert(0, '/swig')
import swig.FastFireFly as FFF

DO_FILEOUT = False
DO_REGRESSION_DATA_OUTPUT = True
CHECK_DUR = 0.5



defaultPosition = [1000, 0, 0, 0, 0, 1.0]
prevMotionSet = {}

writePoints = open("data/PositionLog.csv", "w+")
writePoints.close()

def calculatePosition(roverID, ptIndex, point_camXAngle, point_camYAngle):
    global prevMotionSet, defaultPosition
    ptAngs = np.array([point_camXAngle, point_camYAngle], dtype=np.double)
    
    # Load positional data for rover
    roverIDname = rc.rover_nameByIndex[roverID]
    lanternPointSet = rc.rover_configSet[roverIDname]['LED_positions']
    ptCount = len(lanternPointSet[0])

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

    if roverID in prevMotionSet:
        prevMotion = prevMotionSet[roverID]
    else:
        prevMotion = defaultPosition


    if DO_REGRESSION_DATA_OUTPUT:
        print(f"Starting Localization with Regression Data Output")
        # Actually call localization
        localization = FFF.ledLocalizationFast(lanternPointSet, prevMotion)
        motion_best = localization.fitData_3D(ptAngs, ptIndex.tolist(), 100)

        motionSaveSet = [motion_best]
        improvementSteps = [0]
        errorSteps = [localization.getError()/ptCount]
        
        DataProc_config = rc.rover_configSet[roverIDname]['DataProc_config']
        acceptableError = DataProc_config['acceptableError']
        targetError = DataProc_config['targetError']
        maxAttempts = DataProc_config['maxAttempts']
        
        netTestAttempts = 0
        testAttempts = 0
        while localization.getError()/ptCount > targetError and testAttempts < 10:
            motion_best = localization.fitData_3D(ptAngs, ptIndex.tolist(), 10)
            testAttempts += 1
            netTestAttempts += 1
            
            if motion_best != motionSaveSet[-1]:
                currError = localization.getError()/ptCount
                print(f"Error > {targetError}: {currError}")
                motionSaveSet.append(motion_best)
                improvementSteps.append(netTestAttempts)
                errorSteps.append(currError)
          
        testAttempts = 0      
        while localization.getError()/ptCount > targetError and testAttempts < 100:
            motion_best = localization.fitData_3D(ptAngs, ptIndex.tolist(), 100)
            testAttempts += 1
            netTestAttempts += 1
            
            if motion_best != motionSaveSet[-1]:
                currError = localization.getError()/ptCount
                print(f"Error > {targetError}: {currError}")
                motionSaveSet.append(motion_best)
                improvementSteps.append(netTestAttempts)
                errorSteps.append(currError)

        testAttempts = 0
        while localization.getError()/ptCount > targetError and testAttempts < 5000:
            motion_best = localization.fitData_3D(ptAngs, ptIndex.tolist(), 1000)
            testAttempts += 1
            netTestAttempts += 1
            
            if motion_best != motionSaveSet[-1]:
                currError = localization.getError()/ptCount
                print(f"Error > {targetError}: {currError}")
                motionSaveSet.append(motion_best)
                improvementSteps.append(netTestAttempts)
                errorSteps.append(currError)
        



        pklDict = {
            'motion': motionSaveSet,
            'step': improvementSteps,
            'error':errorSteps,
            'ptIndex':ptIndex,
            'camXAngle':point_camXAngle,
            'camYAngle':point_camYAngle,
        }

        pklOut = open("data/regressionData.pkl", 'wb')
        pkl.dump(pklDict, pklOut)
        pklOut.close()

        prevMotion = motion_best


    else:
        print(f"Starting Localization")
        # Actually call localization
        localization = FFF.ledLocalizationFast(lanternPointSet, prevMotion)
        motion_best = localization.fitData_imageCentric(ptAngs, ptIndex.tolist(), 1000)
        motion_best = localization.fitData_3D(ptAngs, ptIndex.tolist(), 10000)
        
        DataProc_config = rc.rover_configSet[roverIDname]['DataProc_config']
        acceptableError = DataProc_config['acceptableError']
        targetError = DataProc_config['targetError']
        maxAttempts = DataProc_config['maxAttempts']
        
        testAttempts = 0

        while localization.getError()/ptCount > targetError and testAttempts < maxAttempts:
            print(f"Error > {targetError}: {localization.getError()/ptCount}")
            motion_best = localization.fitData_3D(ptAngs, ptIndex.tolist(), 50000)
            testAttempts += 1

        prevMotion = motion_best



    fooError = localization.getError()/ptCount



    # Optional data plot
    if False:
        outPtSet = completeMotion(lanternPointSet, motion_best)
        outXAng = np.arctan2(outPtSet[2], outPtSet[0])
        outYAng = np.arctan2(outPtSet[1], outPtSet[0])
    
        for ii in range(len(ptIndex)):
            plt.plot(
                [point_camXAngle[ii], outXAng[ptIndex[ii]]],
                [point_camYAngle[ii], outYAng[ptIndex[ii]]],
                color='black'
            )
            
        plt.scatter(outXAng, outYAng)
        plt.scatter(point_camXAngle, point_camYAngle)
        plt.show()



    if testAttempts >= maxAttempts and localization.getError()/ptCount > acceptableError: 
        print(f"Exceeded max attempts :(, error > {acceptableError}")
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

    # Save motion to use as base pos for future runs
    prevMotionSet[roverID] = motion_best

    return(motion_best)




import zmq

# nameserver for windows, cat /etc/resolv.conf
NameServer = rc.LOCALHOST_NAME_SERVER
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
    
    roverID = int(data[0])
    point_indices = np.frombuffer(data[1 : 1+index_size], dtype=np.uint32)
    point_camXAngle = np.frombuffer(data[1+index_size : 1+index_size+vect_size], dtype=np.double)
    point_camYAngle = np.frombuffer(data[1+index_size+vect_size : 1+index_size+vect_size*2], dtype=np.double)


    print(f"\nProcessing rover[{roverID}]: {rc.rover_nameByIndex[roverID]}")
    print(f"point_indices:{point_indices}")
    print(f"point_camXAngle:{point_camXAngle}")
    print(f"point_camYAngle:{point_camYAngle}")


    posList = calculatePosition(roverID, point_indices, point_camXAngle, point_camYAngle)
    outPosition = np.array(posList, dtype=np.double)
    
    print(f"Sending Position: {outPosition}")

    socket.send(outPosition.tobytes())