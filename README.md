
## ZMQ Messaging notes

### Data Proc
Address: tcp://*:5555
Type: REQ/REP
REQ: navigate.py
REP: DataProc_ZMQ

Data:
Request sends N data points for processing: roverIndex (1), indices (N*uint32), camXAngle (N*double), camYAngle(N*double)
Reply sends position of rover relative to camera (6*double)
A

### Rover Map Position

Address: tcp://*:5556
Type: PUB/SUB
REQ: navigate.py
REP: Plot_ZMQ.py

Data:
Publisher sends XYZ Position (3*double) and metadata (8*int32)
Metadata: [netPointIndex, positionType, roverID, turnDuration, runDuration, sensorData[0], sensorData[1], sensorData[2] ]


### Rover Move Data

Address: tcp://*:5557
Type: PUB/SUB
REQ: navigate.py
REP: SteeringML_ZMQ.py

Data:
Publisher sends XYR start & end position (6*double 3*double),  and metadata (8*int32)
Metadata: [netPointIndex, positionType, roverID, turnDuration, runDuration, sensorData[0], sensorData[1], sensorData[2] ]


### Rover Absolute Position

Address: tcp://*:5558
Type: PUB/SUB
REQ: navigate.py
REP:

Data:
Publisher sends roverID (1*int32), XYZrXrYrZ Position (6*double)


### ML Params

Address: tcp://*:5559
Type: PUB/SUB
REQ: SteeringML_ZMQ.py
REP: navigate.py

Data:
Publisher sends roverID (1*uint8), controlValue (1*uint8) and 3 factors (3*double) for each rover
controlValue: 0->turn, 1->drive