import time
import math as m
import cv2
import numpy as np
import pickle as pkl
import imageio
from copy import deepcopy
from py.serialCommunication import listPorts, roverSerial
import random


TRESH_MIN = 120

LED_BRIGHTNESS_MIN = 5
LED_BRIGHTNESS_MAX = 255
LED_BRIGHTNESS_DEFAULT = 150

SPOT_SIZE_MIN = 90
SPOT_SIZE_MAX = 130

MAX_ATTEMPTS = 6

MIN_PHOTO_PERIOD = 0.2



def adjacentImages(imgArr): # Place images into grid for easy comparison
    blankImg = np.zeros_like(imgArr)
    rowMax = max([len(ii) for ii in imgArr])
    
    for fooRow in imgArr:
        while (len(fooRow) < rowMax):
            fooRow.append(blankImg)

    imgRows = []
    for fooRow in imgArr:
        concatRow = np.concatenate(fooRow, axis=1)
        imgRows.append(concatRow)

    return(np.concatenate(imgRows, axis=0))



def edge_filter(img):
    # Filter out noise, get solid particle outline
    kernelDim = 4
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernelDim, kernelDim))
    # kernel = np.ones((kernelDim, kernelDim),np.uint8)
    imgOut = cv2.morphologyEx(img, cv2.MORPH_ELLIPSE, kernel)
    return(imgOut)



class webcam:
    def __init__(self, _LED_positions, _roverComms, _LED_exclusion=[]):
        self.roverComms = _roverComms
        self.LED_positions = _LED_positions # 3xN array of LED positions for processing
        self.LED_count = len(self.LED_positions[0]) # Total number of LEDs

        # Get easier array of mutually exclusive LEDs for each LED
        self.LED_nonConsecutives = np.full((self.LED_count, self.LED_count), 0)
        for fooExclusionSet in _LED_exclusion:
            fooExclusionSetNP = np.array(fooExclusionSet)
            for fooExclude in fooExclusionSetNP:
                self.LED_nonConsecutives[fooExclude][fooExclusionSetNP] = 1
        
        # Setting up cam
        self.cam = cv2.VideoCapture(0)
        result, self.img = self.cam.read()
        self.prevImg = self.img
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.lastImgTime = time.time()
        
        # Set up feedback images
        self.img_posOverLay = np.zeros_like(self.img)
        self.img_subtract = np.zeros_like(self.img)
        self.img_points = np.zeros_like(self.img)
        
        self.img_thresh = np.zeros_like(self.img_gray)
        self.img_subtract = np.zeros_like(self.img_gray)

        self.LED_brightness = np.full(self.LED_positions[0].shape, LED_BRIGHTNESS_DEFAULT)

        # Set up displays to allow rearranging them before recording
        # cv2.imshow('Display', adjacentImages([[self.prevImg, self.prevImg]])) # Show original) # Show original
        # cv2.waitKey(1)
        # plt.draw()
        self.clearData()



    def clearData(self):
        # Output information about LEDs
        self.ledData = { 
            'size': [], # Size of threshold area
            'index': [], # Index of pixel
            'brightness': [], # LED Brightness Setting
            # LED Position in image (in pixels)
            'xPix': [],
            'yPix': [],
            # LED vector angle (rad)
            'xAng': [],
            'yAng': [],
            # LED XYZ Position on Lantern (mm)
            'LED_X': [],
            'LED_Y': [],
            'LED_Z': [],
        }
        
        # Save data from previous attempts
        # Structure is [type of failure (str), size, brightness, number of areas]
        self.failedAttempts = [[] for foo in range(self.LED_count)]
        self.finishedLEDs = []



    # Load input image
    def readImage(self):        
        # Delay if too soon after previous image
        waitTime = m.floor(1000*(MIN_PHOTO_PERIOD - (time.time()-self.lastImgTime) ))
        if waitTime > 0:
            cv2.waitKey(waitTime)
            
        # Save image as previous image
        self.prevImg = self.img
        self.prevImg_gray = self.img_gray
        
        # Get new image
        result, self.img = self.cam.read()
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        self.lastImgTime = time.time()

        return(self.img)

    def adjustBrightness(self, ledIndex, adjustAmount):
        if abs(adjustAmount) < 20: adjustAmount = 20*adjustAmount/abs(adjustAmount)

        newBrightNess = m.floor(self.LED_brightness[ledIndex] + adjustAmount)
        newBrightNess = np.sort([LED_BRIGHTNESS_MIN, newBrightNess, LED_BRIGHTNESS_MAX])[1]
        self.LED_brightness[ledIndex] = newBrightNess

        for fooNeighbor in np.where( self.LED_nonConsecutives[ledIndex] <= 0 )[0]:
            if self.LED_brightness[fooNeighbor] == LED_BRIGHTNESS_DEFAULT: self.LED_brightness[fooNeighbor] = newBrightNess

    # Process most recent image
    def processImg(self, ledIndex):
        # Blur and subtract image
        blurConvolution = (5, 5)        
        self.img_subtract = cv2.subtract(cv2.GaussianBlur(self.img_gray, blurConvolution, 0), cv2.GaussianBlur(self.prevImg_gray, blurConvolution, 0)) 

        # img_subtract = cv2.subtract(cv2.GaussianBlur(self.img, blurConvolution, 0), cv2.GaussianBlur(self.prevImg, blurConvolution, 0)) 
        # imgGrey_sub = cv2.cvtColor(img_subtract, cv2.COLOR_BGR2GRAY)

        # Threshold subtracted image
        ret, self.img_thresh = cv2.threshold(self.img_subtract, TRESH_MIN, 255, cv2.THRESH_BINARY)
        contours, hierarchy= cv2.findContours(self.img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
        
        if len(sorted_contours) == 0:
            self.failedAttempts[ledIndex].append(['NO SPOT', 0, self.LED_brightness[ledIndex], 0])
            self.adjustBrightness(ledIndex, 60)
            return(False)

        elif len(sorted_contours) > 1:
            self.failedAttempts[ledIndex].append(['MULTIPLE SPOTS', 0, self.LED_brightness[ledIndex], len(sorted_contours)])
            self.adjustBrightness(ledIndex, -60)
            return(False)

        # Draw contours and load pixel data
        for bar in sorted_contours[1:]:
            cv2.drawContours(self.img_thresh, bar, -1, 50, 5)
        mass_y, mass_x = np.where(self.img_thresh >= 255)
        imgPos_s = len(mass_x) # Number of pixels in spot


        # Catch if spot size is out of range
        if imgPos_s < SPOT_SIZE_MIN: # Too small
            self.failedAttempts[ledIndex].append(['SPOT UNDER SIZE', imgPos_s, self.LED_brightness[ledIndex], 1])
            if len(self.failedAttempts[ledIndex]) > 1 and self.failedAttempts[ledIndex][-2][0] == 'SPOT UNDER SIZE':
                self.adjustBrightness(ledIndex, 4*(SPOT_SIZE_MIN - imgPos_s))
            else:
                self.adjustBrightness(ledIndex, (SPOT_SIZE_MIN - imgPos_s))
            return(False)

        elif imgPos_s > SPOT_SIZE_MAX and self.LED_brightness[ledIndex] != SPOT_SIZE_MIN: # Too large
            self.failedAttempts[ledIndex].append(['SPOT OVER SIZE', imgPos_s, self.LED_brightness[ledIndex], 1])

            if len(self.failedAttempts[ledIndex]) > 1 and self.failedAttempts[ledIndex][-2][0] == 'SPOT OVER SIZE':
                self.adjustBrightness(ledIndex, 3*(SPOT_SIZE_MAX - imgPos_s))
            else:
                self.adjustBrightness(ledIndex, (SPOT_SIZE_MAX - imgPos_s))
            return(False)


        # Get center of LED position
        imgPos_x = np.average(mass_x)
        imgPos_y = np.average(mass_y)
        imgAng_x = (imgPos_x - self.img_thresh.shape[1]/2) / 668
        imgAng_y = (self.img_thresh.shape[0]/2 - imgPos_y) / 668 
        
        # Save data point
        self.ledData['size'].append(imgPos_s)
        self.ledData['index'].append(ledIndex)
        self.ledData['brightness'].append(self.LED_brightness[ledIndex])
        self.ledData['xPix'].append(imgPos_x)
        self.ledData['yPix'].append(imgPos_y)
        self.ledData['xAng'].append(imgAng_x)
        self.ledData['yAng'].append(imgAng_y)
        self.ledData['LED_X'].append(self.LED_positions[0][ledIndex])
        self.ledData['LED_Y'].append(self.LED_positions[1][ledIndex])
        self.ledData['LED_Z'].append(self.LED_positions[2][ledIndex])
        
        self.finishedLEDs.append(ledIndex)

        print(f"Success {len(self.ledData['LED_X'])} {' '.ljust(80)} img_x:{round(imgPos_x,2)}   img_y:{round(imgPos_y,2)}   img_s:{round(imgPos_s,2)}")
        
        return(True)


        
    def readVectors(self):
        prevLED = -1
        currLED = 0
        # Loop until all LEDs are either measured or abandoned
        while len(self.finishedLEDs) < self.LED_count:
            prevLED = currLED

            # Find new LED to test
            if prevLED > -1: potentials_consec = np.where( self.LED_nonConsecutives[prevLED] <= 0 )[0]
            else: potentials_consec = np.arange(self.LED_count)

             # Get list of LEDs which are able to be shown next
            potentials_notFinished = np.setdiff1d( np.arange(self.LED_count), self.finishedLEDs ) # Drop abandoned LEDs
            potentials = np.intersect1d(potentials_consec, potentials_notFinished)


            # print(f"\n\npotentials_consec:{potentials_consec}")
            # print(f"potentials_notFinished:{potentials_notFinished}")
            # print(f"potentials:{potentials}\n\n")



            if len(potentials) == 0:
                print(f"No Potentials, taking pic")
                self.readImage()
                self.updateDisplay()
                currLED = -1
                continue
            
            currLED = potentials[m.floor(random.random()*len(potentials))]

            print(f"Testing {str(currLED).ljust(4, ' ')}   ", end='')

            self.roverComms.setLED(currLED, [self.LED_brightness[currLED], self.LED_brightness[currLED], self.LED_brightness[currLED]] )
            
            # Read and process image
            self.readImage()
            processSuccess = self.processImg(currLED)

            # If process did not succeed
            if not processSuccess:
                attempt = self.failedAttempts[currLED][-1]

                if len(self.failedAttempts[currLED]) >= MAX_ATTEMPTS:
                    self.finishedLEDs.append(currLED)
                    print("Abandoned ", end='')
                else:
                    print("Failed    ", end='')

                print(f"{attempt[0].ljust(20, ' ')}  {str(attempt[1]).ljust(5, ' ')}  {str(attempt[2]).ljust(5, ' ')}  {str(attempt[3]).ljust(5, ' ')} setting brightness to {self.LED_brightness[currLED]}")


            
            # Turn off LED
            self.roverComms.setLED(currLED, [0, 0, 0] )

            # Display data
            self.updateDisplay()
            



        return(self.ledData)
    
    def getData(self):
        return(self.ledData)
    
    def getFails(self):
        return(self.failedAttempts)


    # def saveGif(self, fileName):
    #     imageio.mimsave(fileName, self.imagelist, fps=0.5)       

    def updateDisplay(self):
        self.img_posOverLay = deepcopy(self.img)
        self.img_points = np.zeros_like(self.img)

        for ii in range(len(self.ledData['xPix'])):
            xPix = round(self.ledData['xPix'][ii])
            yPix = round(self.ledData['yPix'][ii])
            pltRad = round(self.ledData['size'][ii]/30)
            cv2.circle( self.img_points, (xPix, yPix), pltRad, (0,0,255), 1)
            
            crossLen = 4
            cv2.line( self.img_posOverLay, (xPix-crossLen, yPix), (xPix+crossLen, yPix), (0,0,255), 1)
            cv2.line( self.img_posOverLay, (xPix, yPix-crossLen), (xPix, yPix+crossLen), (0,0,255), 1)
        
        # self.img_gray = cv2.cvtColor(self.img_gray, cv2.COLOR_GRAY2BGR)
        display_thresh = cv2.cvtColor(self.img_thresh, cv2.COLOR_GRAY2BGR)
        display_subtract = cv2.cvtColor(self.img_subtract, cv2.COLOR_GRAY2BGR)

        cv2.imshow('Display', adjacentImages([[self.img_posOverLay, self.img_points], [display_subtract, display_thresh]]))


# If this file is run as main, show video with center crosshair for camera calibration
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Display the resulting frame
        
        frame = cv2.circle(frame, (round(frame.shape[1]/2), round(frame.shape[0]/2)), round(frame.shape[0]/4), (0, 0, 255), 1)
        frame = cv2.line(frame, (0, round(frame.shape[0]/2)), (frame.shape[1], round(frame.shape[0]/2)), (0, 0, 255), 1)
        frame = cv2.line(frame, (round(frame.shape[1]/2), 0), (round(frame.shape[1]/2), frame.shape[0]), (0, 0, 255), 1)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()