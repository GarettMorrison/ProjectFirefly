from copy import deepcopy
from math import cos, sin
import serial
import sys
import os
import time
import struct
import math as m
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import random as r
import statistics as st

from positionFuncs import *

# mm
RING_RADIUS = 100
RING_HEIGHT = 30

readFile = open('data/outData.pkl', 'rb')
# readFile = open('data/data_sideAngle.pkl', 'rb')
dataDict = pkl.load(readFile)
readFile.close()

for foo in dataDict: print(foo)