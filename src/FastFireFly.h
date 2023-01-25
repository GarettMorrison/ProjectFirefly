#ifndef FASTFIREFLY_H
#define FASTFIREFLY_H


#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <iostream>
#include <vector>

using namespace std;

// Initialize localization system with LED array, save to avoid reloading constant information constantly
class ledLocalizationFast{
private:
    // Positions LEDs relative to the center
    vector<double> LED_Set_X;
    vector<double> LED_Set_Y;
    vector<double> LED_Set_Z;

    // Current best set of position (mm) and rotation (rad) values
    vector<double> currentPosition = {0,0,0,0,0,0}; 
    double currentError;

    // Current test set of position (mm) and rotation (rad) values
    vector<double> testPosition = {0,0,0,0,0,0}; 
    double testError;

    // Translated positions of LEDs
    vector<double> LED_TestPos_X;
    vector<double> LED_TestPos_Y;
    vector<double> LED_TestPos_Z;
    // LED position vectors for input
    vector<double> InputVect_X;
    vector<double> InputVect_Y;
    vector<double> InputVect_Z;
    vector<double> InputVect_LED_ID;

    // Factors by which to randomize position for testing
    double default_randomizeFactor = 500;
    double randomizeFactor = 500;

    double acceptableRandomizerVal = 0.01;

    void randomizeTestPosition(); // Randomly modify test position according to randomizeFactor
    void calculateLedTestPositions(); // Calculate LED_Testpos_X arrays from testPosition and LED_Set_XYZ
    double calculateError(); // Calculate error of current test position



public:
    ledLocalizationFast(vector<double> _LED_X, vector<double> _LED_Y, vector<double> _LED_Z, double _pos_x=1000, double _pos_y=0, double _pos_z=0);
    vector<double> getPosition(void){ return(currentPosition); }
    vector<double> fitPositionToVectors(vector<double> vect_X, vector<double> vect_Y, vector<double> vect_Z, vector<double> _LED_Indices);
};

#endif