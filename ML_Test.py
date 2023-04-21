import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=2, suppress=True)

import math as m
import pickle as pkl
import zmq
import time

import roverConfig as rc



# ZMQ Communication to receive dead reckoning control values from 
NameServer = rc.LOCALHOST_NAME_SERVER
context = zmq.Context()
steeringML_socket = context.socket(zmq.SUB)
steeringML_socket.connect(f"tcp://{NameServer}:5558")
steeringML_socket.setsockopt_string(zmq.SUBSCRIBE, "")


print(f"Waiting to receive from socket")
while True:
    dataFileName = "data/roverSteeringCal.pkl"
    try:
        file = open(dataFileName, 'rb')
        dataDict = pkl.load(file)
        file.close()
    except:
        print(f"Unable to load {dataFileName}")
    
    print("\n\n")
    print(dataDict)
    time.sleep(1)


exit()
PLOT_DATA = True

dataFileName = "data/roverHistory.pkl"
dataDict = {}

try:
    file = open(dataFileName, 'rb')
    dataDict = pkl.load(file)
    file.close()
except:
    print(f"Unable to load {dataFileName}")



#    position_start
#    position_end
#    turn_duration
#    move_duration
#    sensorData
