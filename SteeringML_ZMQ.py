import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=2, suppress=True)

import math as m
import pickle as pkl
import zmq

import roverConfig as rc





PLOT_DATA = True
MAX_DATA_HIST = 80
MIN_DATA_PROC = 10

MAX_RUN_DUR = 1000
MIN_RUN_DUR = 5

MIN_TURN_DUR = 15
MAX_TURN_DUR = 250


# MAX_TURNING_ANGLE = 0.75
MAX_TURNING_ANGLE = 0.75



# Function to calculate motion durations for given cal values
# Intended to be called by a function receiving data over tcp://localhost:5559
def calcMotionDurs(currPos, targPos, steeringData):
                
    distOut = m.sqrt( m.pow(targPos[0]-currPos[0], 2) + m.pow(targPos[1]-currPos[1], 2) )
    turnAngle = np.arctan2( targPos[0]-currPos[0], targPos[1]-currPos[1] ) - currPos[2]
    
    if turnAngle > m.pi: turnAngle -= m.pi*2
    if turnAngle < -m.pi: turnAngle += m.pi*2

    driveDur = steeringData[1][0]*np.square(distOut) + steeringData[1][1]*distOut + steeringData[1][2]
    turnDur = steeringData[0][0]*np.square(turnAngle) + steeringData[0][1]*turnAngle + steeringData[0][2]

    print(f"\n   calcMotionDurs:")
    print(f"      turnAngle:{turnAngle}")
    print(f"      turnDur = {round(steeringData[0][0], 2)}*Y^2 + {round(steeringData[0][1], 2)}*Y + {round(steeringData[0][2], 2)}")
    print(f"      distOut:{distOut}")
    print(f"      driveDur = {round(steeringData[1][0], 2)}*Y^2 + {round(steeringData[1][1], 2)}*Y + {round(steeringData[1][2], 2)}")

    if(driveDur > MAX_RUN_DUR): driveDur = MAX_RUN_DUR
    if(driveDur < MIN_RUN_DUR): driveDur = MIN_RUN_DUR
    
    if(abs(turnDur) > MAX_TURN_DUR): turnDur = MAX_TURN_DUR * (turnDur/abs(turnDur))

    turnDur = round(turnDur)
    driveDur = round(driveDur)

    if abs(turnAngle) > MAX_TURNING_ANGLE:
        driveDur = 0
    
    if driveDur == 0 and abs(turnDur) < MIN_TURN_DUR:
        turnDur = MIN_TURN_DUR*turnAngle/abs(turnAngle)

    return(turnDur, driveDur)



if __name__ == "__main__":
    iterationIndex = 0
    dataFileName = "data/roverHistory.pkl"
    dataDict = {}

    calFileName = "data/roverSteeringCal.pkl"

    try:
        file = open(dataFileName, 'rb')
        dataDict = pkl.load(file)
        file.close()
    except:
        print(f"Unable to load {dataFileName}")



    for foo in dataDict:
        print(f"{foo}:")
        for bar in dataDict[foo]:
            print(f"   {bar}:{dataDict[foo][bar]}")


    NameServer = rc.LOCALHOST_NAME_SERVER

    # context = zmq.Context()
    # output_socket = context.socket(zmq.PUB)
    # output_socket.connect(f"tcp://{NameServer}:5558")

    context = zmq.Context()
    dataIntake_socket = context.socket(zmq.SUB)
    dataIntake_socket.connect(f"tcp://{NameServer}:5557")
    dataIntake_socket.setsockopt_string(zmq.SUBSCRIBE, "")


    if PLOT_DATA:
        fig, ax = plt.subplots(1, 2)
        plt.ion()


    while True:
        # Get new data point
        try:
            data = dataIntake_socket.recv(flags=zmq.NOBLOCK)
            
        except:
            # No data loaded, abandoning
            plt.pause(0.1)
            continue

        doubleByteLen = np.dtype(np.double).itemsize*6 # Number of bytes in 3 doubles

        positionData = np.frombuffer(data[:doubleByteLen], dtype=np.double)
        metaData = np.frombuffer(data[doubleByteLen:], dtype=np.int32)

        position_start = positionData[:3]
        position_end = positionData[3:]

        roverID = metaData[2]

        if roverID not in dataDict: # Init arrays for dict
            dataDict[roverID] = {
                "position_start": [],
                "position_end": [],
                "turn_duration": [],
                "move_duration": [],
                "sensorData": [],
            }

        print(f"\nREC: {position_start} -> {position_end}")
        # Save position
        dataDict[roverID]["position_start"].append(position_start)
        dataDict[roverID]["position_end"].append(position_end)
        dataDict[roverID]["turn_duration"].append(metaData[3])
        dataDict[roverID]["move_duration"].append(metaData[4])
        dataDict[roverID]["sensorData"].append(metaData[5:])

        if len(dataDict[roverID]["position_start"]) > MAX_DATA_HIST:
            dataDict[roverID]["position_start"] = dataDict[roverID]["position_start"][1:]
            dataDict[roverID]["position_end"] = dataDict[roverID]["position_end"][1:]
            dataDict[roverID]["turn_duration"] = dataDict[roverID]["turn_duration"][1:]
            dataDict[roverID]["move_duration"] = dataDict[roverID]["move_duration"][1:]
            dataDict[roverID]["sensorData"] = dataDict[roverID]["sensorData"][1:]













        # Calculate angle and distance for each pair
        for foo in dataDict:
            fooDict = dataDict[foo]
            
            distance = []
            angle = []
            relativeX = []
            relativeY = []
            for ii in range(len(fooDict['position_start'])):
                
                distOut = m.sqrt(m.pow(fooDict['position_end'][ii][0]-fooDict['position_start'][ii][0], 2) + m.pow(fooDict['position_end'][ii][1]-fooDict['position_start'][ii][1], 2) )
                distance.append(distOut)

                # angleOut = np.arctan2(fooDict['position_end'][ii][0]-fooDict['position_start'][ii][0], fooDict['position_end'][ii][1]-fooDict['position_start'][ii][1], ) - fooDict['position_start'][ii][2]
                
                angleOut = fooDict['position_end'][ii][2] - fooDict['position_start'][ii][2]

                if angleOut > m.pi and fooDict["turn_duration"][ii] < 0: angleOut -= m.pi*2
                if angleOut < -m.pi and fooDict["turn_duration"][ii] > 0: angleOut += m.pi*2

                angle.append( angleOut)

                if ii == len(fooDict['position_start'])-1: print(f"      angleOut:{angleOut}")
                
                relativeX.append(m.sin(angleOut)*distOut)
                relativeY.append(m.cos(angleOut)*distOut)

            dataDict[foo]['netAngle'] = np.array(angle)
            dataDict[foo]['netDist'] = np.array(distance)
            dataDict[foo]['xDelta'] = np.array(relativeX)
            dataDict[foo]['yDelta'] = np.array(relativeY)


        # # Print dictionary
        # for foo in dataDict:
        #     print(f"{foo}:")
        #     for bar in dataDict[foo]:
        #         dataDict[foo][bar] = np.array(dataDict[foo][bar])
        #         print(f"   {bar}")


        # Setup plot
        if PLOT_DATA:
            ax[0].cla()
            ax[1].cla()

        for fooRoverIndex in dataDict:
            nameLabel = rc.rover_nameByIndex[fooRoverIndex]

            # Get subset of turn data in first and third quandrants, to remove over-rotated points
            durationSet = np.array( dataDict[fooRoverIndex]['turn_duration'] )
            angleSet = np.array( dataDict[fooRoverIndex]['netAngle'] )
            sameSignIndices = np.append(
                np.append(
                    np.intersect1d(np.where(durationSet > 0), np.where(angleSet > 0) ),   
                    np.intersect1d(np.where(durationSet < 0), np.where(angleSet < 0) ),    
                ),
                np.where(np.abs(angleSet) < np.pi)
            )



            durationSet = durationSet[sameSignIndices]
            angleSet = angleSet[sameSignIndices]

            if PLOT_DATA:
                ax[0].scatter(durationSet, angleSet, label=nameLabel)

            if len(durationSet) > MIN_DATA_PROC:
                angleFit = np.polyfit(angleSet, durationSet, 2)

                # if iterationIndex%5 == 0:
                # sendBytes = np.uint16(fooRoverIndex).tobytes() + np.uint16(0).tobytes() + np.array(angleFit, dtype=np.double).tobytes()
                # print(f"Sending angleFit:{angleFit} ({sendBytes})")
                print(f"Sending angleFit:{angleFit}")
                # output_socket.send( sendBytes )

                rc.rover_steeringVals[nameLabel][0] = np.array(angleFit, dtype=np.double)

                if PLOT_DATA:
                    yPts = np.linspace(np.min(angleSet), np.max(angleSet))
                    ax[0].plot(angleFit[0]*np.square(yPts) + angleFit[1]*yPts + angleFit[2], yPts)


            netDistSet = np.array( dataDict[fooRoverIndex]['netDist'] )
            moveDurSet = np.array( dataDict[fooRoverIndex]['move_duration'] )
            dropZerosSubset = np.where(moveDurSet > 0)
            netDistSet = netDistSet[dropZerosSubset]
            moveDurSet = moveDurSet[dropZerosSubset]

            if PLOT_DATA:
                ax[1].scatter(moveDurSet, netDistSet, label=nameLabel)

            if len(moveDurSet) > MIN_DATA_PROC:
                distFit = np.polyfit(netDistSet, moveDurSet, 2)

                # if iterationIndex%5 == 0:
                sendBytes = np.uint16(fooRoverIndex).tobytes() + np.uint16(1).tobytes() + np.array(distFit, dtype=np.double).tobytes()
                # print(f"Sending distFit:{distFit} ({sendBytes})")
                print(f"Sending distFit:{distFit}")

                rc.rover_steeringVals[nameLabel][1] = np.array(distFit, dtype=np.double)

                if PLOT_DATA:
                    yPts = np.linspace(np.min(netDistSet), np.max(netDistSet))
                    ax[1].plot(distFit[0]*np.square(yPts) + distFit[1]*yPts + distFit[2], yPts)
                

        if PLOT_DATA:
            plt.suptitle('Steering Calibration Machine Learning')
            ax[0].set_title('Turn Data')
            ax[0].set_xlabel('Turn Duration (mS)')
            ax[0].set_ylabel('Angle Change (RAD)')
            ax[0].legend()
            
            # ax[0].set_ylim(-np.pi, np.pi)
            
            ax[1].set_title('Drive Data')
            ax[1].set_xlabel('Drive Duration (mS)')
            ax[1].set_ylabel('Distance Traveled (mm)')
            ax[1].legend()
            
            plt.show()



        # Save data 
        try:
            file = open(dataFileName, 'wb')
            pkl.dump(dataDict, file)
            file.close()
        except:
            print(f"Unable to save data")

        try:
            file = open(calFileName, 'wb')
            pkl.dump(rc.rover_steeringVals, file)
            file.close()
        except:
            print(f"Unable to save cal")
            

        
        iterationIndex += 1
