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



def doRotationMatrixes(inPts, rotations, transposed = False):
    A = rotations[0]
    B = rotations[1]
    C = rotations[2]

    rotationMags = np.array([
        [cos(C)*cos(B), -sin(C)*cos(A) + cos(C)*sin(A)*sin(B), sin(C)*sin(A)+cos(C)*sin(B)*cos(A)],
        [sin(C)*cos(B), cos(C)*cos(A) + sin(C)*sin(B)*sin(A), -cos(C)*sin(A)+sin(C)*sin(B)*cos(A)],
        [-sin(B), cos(B)*sin(A), cos(B)*cos(A)],
    ])
    
    if transposed: rotationMags = np.matrix.transpose(rotationMags)


    # for foo in inPts[1]:
    #     print(f"{foo} : {foo*rotationMags[0][0]}")

    xPts = sum([inPts[ii]*rotationMags[ii][0] for ii in range(3)])
    yPts = sum([inPts[ii]*rotationMags[ii][1] for ii in range(3)])
    zPts = sum([inPts[ii]*rotationMags[ii][2] for ii in range(3)])

    # for foo in rotationMags: print(foo)

    return([xPts, yPts, zPts])


def completeMotion(inPts, motion):
    offSets = motion[:3]
    rotations = motion[3:6]

    ptSet = doRotationMatrixes(inPts, rotations)

    for ii in range(3): 
        ptSet[ii] += offSets[ii]    
    
    return(ptSet)


def undoMotion(inPts, motion):
    offSets = motion[:3]
    rotations = motion[3:6]
    
    for ii in range(3): 
        inPts[ii] -= offSets[ii]    

    ptSet = doRotationMatrixes(inPts, rotations, transposed=True)
    
    return(ptSet)

def getClosestPts(inVectors, inPts):
    inPts = np.column_stack(inPts)
    inVectors = np.column_stack(inVectors)

    tSet = np.sum(inPts*inVectors, axis=1) / np.sum(inVectors*inVectors, axis=1)
    # print(tSet)
    # print(tSet[:, None])
    outPts = inVectors * tSet[:, None] # Converts NP array to 2d with only one element in row
    # print(outPts)
    return(outPts)

def crossFixed(a:np.ndarray,b:np.ndarray)->np.ndarray: # Fix code unreachable error in some IDES
    return np.cross(a,b)

def ptVectDistSquared(pt, line):
    dVect = line - pt # Get vector from arbitrary point on line to target
    distance = np.sum(np.square(crossFixed(line, dVect)), axis=1)/np.sum(np.square(line), axis=1)
    return(distance)

def getError(inVectors, inPts, ptSize):
    inPts = np.column_stack(inPts)
    inVectors = np.column_stack(inVectors)

    distances = ptVectDistSquared(inVectors, inPts)
    # distances *= ptSize
    return( np.sum(distances) )

def testError(inVectors, inPts, motion, ptSize):
    adjPts = completeMotion(inPts, motion)
    sumError = getError(inVectors, adjPts, ptSize)
    return(sumError)






def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
