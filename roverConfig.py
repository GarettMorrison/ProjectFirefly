# Define configurations for various rovers
import numpy as np
from copy import deepcopy


# Define XYZ Position sets for various lantern layouts
# lantern_X_3DP =  [ 60.10456705, -80.93500398, 13.90925146, -15.90356982, -54.8012229, 75.85042568, 49.49787875, -67.61465994, 18.11678119, -20.64195979, -44.1945346, 58.99253963 ]
# lantern_Y_3DP =  [ 15, 15, 15, 30, 30, 30, 45, 45, 45, 60, 60, 60 ]
# lantern_Z_3DP =  [ 60.10358575, 25.97162163, -83.85423498, 75.85068534, -54.80032818, -15.90480822, 49.49707061, 18.11788512, -67.61495574, 58.99287665, -44.19381305, -20.64292295 ]


lantern_X_3DP =  [ 60.10456705, -80.93500398, 13.90925146, -15.90356982, -54.8012229, 75.85042568, 49.49787875, -67.61465994, 18.11678119, -20.64195979, -44.1945346, 58.99253963 ]
lantern_Y_3DP =  [ 15, 15, 15, 30, 30, 30, 45, 45, 45, 60, 60, 60 ]
lantern_Z_3DP =  [ 60.10358575, 25.97162163, -83.85423498, 75.85068534, -54.80032818, -15.90480822, 49.49707061, 18.11788512, -67.61495574, 58.99287665, -44.19381305, -20.64292295 ]


lantern_X_WOOD = [ 0, 0, 0, 28.14582562, 39.80420832, 48.75, 48.75, 39.80420832, 28.14582562, 0, 0, 0, -48.75, -39.80420832, -28.14582562, -28.14582562, -39.80420832, -48.75 ]
lantern_Y_WOOD = [ 32.5, 45.96194078, 56.29165125, 56.29165125, 45.96194078, 32.5, 32.5, 45.96194078, 56.29165125, 56.29165125, 45.96194078, 32.5, 32.5, 45.96194078, 56.29165125, 56.29165125, 45.96194078, 32.5 ]
lantern_Z_WOOD = [ -56.29165125, -45.96194078, -32.5, -16.25, -22.98097039, -28.14582562, 28.14582562, 22.98097039, 16.25, 32.5, 45.96194078, 56.29165125, 28.14582562, 22.98097039, 16.25, -16.25, -22.98097039, -28.14582562 ]


# Define exclusion sets for various layouts
# Each list of points specifies a group of LEDs (by index) that cannot be flashed consecutively
# This prevents overlap error
led_exclusion_WOOD = [
    [0, 1, 2], 
    [3, 4, 5], 
    [6, 7, 8],
    [9, 10, 11],
    [12, 13, 14],
    [15, 16, 17],
]

led_exclusion_3DP = [
    [0, 6],
    [1, 7],
    [2, 8],
    [3, 9],
    [4, 10],
    [5, 11],
    
    [0, 1],
    [1, 2],
    [0, 2],
]



test_parameters_WOOD = {
    'acceptableError': 4,
    'targetError': 1,
    'maxAttempts': 6,
}

test_parameters_3DP = {
    'acceptableError': 8,
    'targetError': 2,
    'maxAttempts': 6,
}



# Define configuration template for 3D printed lantern
rover_config_3DP = {
    'LED_positions': np.array([lantern_X_3DP, lantern_Y_3DP, lantern_Z_3DP]),
    'LED_exclusionGroups': np.array(led_exclusion_3DP),
    'LED_hasBrightness': False,
    'LED_brightness': 255,
    'DataProc_config': test_parameters_3DP,
    'Sensor_Angles': [-30, 0, 30],
}



# Define configuration template for Laser Cut Lantern
rover_config_WOOD = {
    'LED_positions': np.array([lantern_X_WOOD, lantern_Y_WOOD, lantern_Z_WOOD]),
    'LED_exclusionGroups': np.array(led_exclusion_WOOD),
    'LED_hasBrightness': True,
    'LED_brightness': 255,
    'DataProc_config': test_parameters_WOOD,
    'Sensor_Angles': [-60, 0, 60],
}


rover_configSet = {
    "BROCK_": deepcopy(rover_config_WOOD),
    "TATE__": deepcopy(rover_config_3DP),
    "LIZA__": deepcopy(rover_config_3DP),
}


rover_nameByIndex = {
    0: "BROCK_",
    1: "TATE__",
    2: "LIZA__",
}



rover_indexByName = {
    "BROCK_": 0,
    "TATE__": 1,
    "LIZA__": 2,
}



rover_displayColor = {
    "BROCK_": (255, 0, 0),
    "TATE__": (0, 255, 255),
    "LIZA__": (255, 0, 255),
}

rover_displayGradients = {
    "BROCK_": [1, 0],
    "TATE__": [2, 1],
    "LIZA__": [0, 2],
}


rover_steeringVals = {
    'BROCK_': [
        [0, 40, 0],
        [0, 0.5, 0]
    ],
    'TATE__': [
        [0, 20, 0],
        [0, 0.5, 0],
    ],
    'LIZA__': [
        [0, 20, 0],
        [0, 0.5, 0],
    ],
}


# nameserver for windows, cat /etc/resolv.conf
LOCALHOST_NAME_SERVER = "172.23.224.1"
