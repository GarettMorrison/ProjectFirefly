Project Firefly is a novel localization method using a webcam to calculate the position of a LED fixture. It combines a fixed array of LEDs to with closed-loop LED control to reliably calculate position in 3D. Everything is open source. 

An example application is provided, which navigates a small rover through a sequence of waypoints. 

A technical breakdown is available on my [Youtube channel](https://www.youtube.com/watch?v=MSs9HHhLadw).

## Scripts

### navigate.py
Connects to rovers and executes waypoint mission. Communicates with other scripts over ZMQ. Needs DataProc_ZMQ to be running in parallel for processing 
Run $python3 navigate.py cal$ to set up waypoint mission by moving the rover by hand through each desired position.

### DataProc_ZMQ.py
Processes localization data from navigate.py. Needs to be run in a Linux enviroment, I use WSL. To work with WSL update the nameserver at the bottom of roverConfig.py. The localization system is written in C++ and is wrapped using SWIG using the Makefile. It may be able to run on Windows as well with some tweaks. 

### PlotMap_ZMQ.py
Plots waypoints and rover motion over time. 

### SteeringML_ZMQ.py
Records and processes motion history. Finds relationship between rover commands and actual motion and sends improved system back to navigate.py. 

## Camera Calibration
Firefly uses the default OpenCV2 camera calibration system. Start by printing off camera_calibration\calibrationCheckerBoard.pdf and taping it to something flat. Run camera_calibration\calibrateCamera.py with the camera connected. Take several images of the checkerboard at different angles by pressing space. Press escape to process and save the calibration data. This is saved to camera_calibration\cameraCal.yaml for later use. 

## Getting Started
I would recommend building the wooden rover. It is laser cut out of 3mm material, and the files are available in /CAD. The wiring information can be found on page 56 of FireFly_Technical_Report.pdf. We use Platformio to flash the Arduino, although the default IDE should work as well. 
