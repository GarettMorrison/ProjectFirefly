import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=2, suppress=True)

import math as m
import zmq

import roverConfig as rc


# Setup ZMQ connection
# nameserver for windows, cat /etc/resolv.conf
NameServer = rc.LOCALHOST_NAME_SERVER
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect(f"tcp://{NameServer}:5556")
socket.setsockopt_string(zmq.SUBSCRIBE, "")


posFileOut = open("data/positionHistory.csv", "+a")

plt.ion()     # turns on interactive mode
plt.draw()    # now this should be non-blocking
plt.pause(0.1)



def drawHex(fooPos, fooCol, radius = 10):
    X = []
    Y = []
    radius = 10

    for ii in range(7):
        X.append(fooPos[0] + radius*m.sin(fooPos[2] + ii*m.pi*2/6))
        Y.append(fooPos[1] + radius*m.cos(fooPos[2] + ii*m.pi*2/6))
    
    plt.plot(X, Y, color=fooCol)


print("Starting Check Loop")
positionHist = {}
waypointList = {}
currTarget = {}

while True:
    # Attempt to load data
    try:
        data = socket.recv(flags=zmq.NOBLOCK)
    except:
        # No data loaded, abandoning
        plt.pause(0.2)
        continue


    # Load data to arrays
    # Double data is XYR position
    # Int data is pointIndex, pointType, roverID, turnduration, runDuration

    doubleByteLen = np.dtype(np.double).itemsize*3 # Number of bytes in 3 doubles
    print(f"")
    print(f"REC len(data): {len(data)}")
    positionData = np.frombuffer(data[:doubleByteLen], dtype=np.double)
    metaData = np.frombuffer(data[doubleByteLen:], dtype=np.int32)

    print(f"REC positionData: {positionData}")
    print(f"REC metaData: {metaData}")


    roverID = metaData[2]

    if roverID not in positionHist: positionHist[roverID] = []
    if roverID not in currTarget: currTarget[roverID] = []
    if roverID not in waypointList: waypointList[roverID] = []


    if metaData[1] == 0: positionHist[roverID].append(positionData) # Save position to history if point is rover position
    # if metaData[1] == 1: currTarget[roverID] = positionData # Save position as currTarget if point is target position
    # if metaData[1] == 2: waypointList[roverID].append(positionData) # Save position to history if point is waypoint position

    # If position is rover data, output to csv
    if metaData[1] == 0:
        for foo in positionData: posFileOut.write(f"{foo},")
        for foo in metaData: posFileOut.write(f"{foo},")
        posFileOut.write(f"\n")
    

    # Data is updated, do plot
    plt.cla()
    plt.axis('equal')


    # # Draw other waypoints
    # for fooPos in waypointList:
    #     drawHex(fooPos, (0.0, 0.0, 1.0))

    plotChannelSels = {
        0: [0, 1],
        1: [2, 1],
        2: [2, 0],
    }

    # Reload plot history
    for fooRoverID in positionHist:
        fooRoverName = rc.rover_nameByIndex[fooRoverID]
        pltChannelA = rc.rover_displayGradients[fooRoverName][0]
        pltChannelB = rc.rover_displayGradients[fooRoverName][1]
        fooPositionHist = positionHist[fooRoverID]

        prevPos = []
        posCount = len(fooPositionHist)
        for posInd in range(len(fooPositionHist)):
            fooPos = fooPositionHist[posInd]
        
            # Calc plotColor
            plotColor = [0.0, 0.0, 0.0]
            if posCount - posInd < 6:
                plotColor[pltChannelA] = 0.9
                plotColor[pltChannelB] = 1.0 - (posCount - posInd)/6
            else:
                plotColor[pltChannelA] = 0.9*(posInd+5)/posCount
            
            posInd += 1


            plotColor = (plotColor[0], plotColor[1], plotColor[2])

            # print(f"\nIndex {posInd} / {posCount}")
            # print(f"Color:{plotColor}")

            # Draw hexagon
            drawHex(fooPos, plotColor)

            # If not first point, draw line
            if len(prevPos) > 0:
                plt.plot([prevPos[0], fooPos[0]], [prevPos[1], fooPos[1]], color=plotColor)
            # Save curr value as previous

            prevPos = fooPos 

        # # Draw current target hex after
        # if len(currTarget) > 0:
        #     drawHex(currTarget, (0.0, 1.0, 0.0), 5)
        #     drawHex(currTarget, (0.0, 1.0, 0.0), 10)
        #     drawHex(currTarget, (0.0, 1.0, 0.0), 15)



