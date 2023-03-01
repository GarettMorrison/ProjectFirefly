#ifndef FASTFIREFLY_H
#define FASTFIREFLY_H

#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <iostream>
#include <vector>

using namespace std;



class transformationSet{
    private:
        vector<vector<double>> inputs;
        vector<vector<double>> outputs;
        vector<double> transformationFactors;

        
    public:
        transformationSet(); // Init empty set
        transformationSet(vector<vector<double>> _input); // Init with XYZ arrays 
        vector<vector<double>> setTransformation(vector<double> _factors); // Set position being moved to
        
        vector<vector<double>> getInputs(){return(inputs);}; // Get inputs
        vector<vector<double>> getOutputs(){return(outputs);}; // Get outputs
        vector<double> getMotion(){return(transformationFactors);}; // Get motion factors
        const vector<double> operator [] (int ii){return(outputs[ii]); } // Reference output with [], is protected so it doesn't copy the vector
};



// Initialize localization system with LED array, save to avoid reloading constant information constantly
class ledLocalizationFast{
    private:
        // Positions LEDs relative to the Lantern Origin
        vector<vector<double>> LED_Set;

        // Current best transformation data
        transformationSet currPos;
        double currentError;
        
        // Testing data
        vector<double> testPosition = {0,0,0,0,0,0}; // Defined position (for randomization)
        transformationSet testPos;  // Transformation set
        double testError; // Calculated test error

        
        vector<vector<double>> LED_TestAng;// Camera angles to translated LEDs

        // LED Data Angle Vectors
        vector<vector<double>> ang_set; // YZ angles from center of camera
        vector<double> ang_size;    // Size of spots matching angles
        vector<uint32_t> LED_indices;   // Indices of LEDs from angles
        vector<uint32_t> LED_dropValue;   // Indices of LEDs from angles
        bool prevErrorReal = false; // Track if we can trust the error value
        vector<vector<double>> ang_line_set; // Line values from angles { tan(ang_set[0]), tan(ang_set[1]) }

        // Factors by which to randomize position for testing
        double default_randomizeFactor = 1; // Starting randomization factor
        double randomizeFactor = 1.0; // Starting randomization value
        double mininumValueX = 200.0; // How close we can get to camera

        double acceptableRandomizerVal = 0.01; // Randomizer val to stop check at
        double acceptableError = 0.001; // Error val to stop checks at

        void randomizeTestPosition(void); // Randomly modify test position according to randomizeFactor
        void randomizeRotation(void); // Just randomize rotation

        double error_3DpointLine(void); // Calculate error between Lantern points and ang_line_set
        double error_cameraAngles(void); // Calculate error between angles to camera and angles to test points
        double error_cameraPosition(void); // Error between positions on image and positions in real data

    public:
        ledLocalizationFast(vector<vector<double>> _LED_Set, vector<double> startPos = {0, 0, 1000, 0, 0, 0});

        vector<double> fitData_imageCentric(vector<vector<double>> _ang_set, vector<unsigned int> _LED_indices, size_t randomizeCount);

        vector<double> fitData_3D(vector<vector<double>> _ang_set, vector<unsigned int> _LED_indices, size_t randomizeCount); // Fit using 3D line distance error
        // vector<double> regressionFitLessRandom(vector<double> vect_X, vector<double> vect_Y, vector<double> vect_Z, vector<double> _LED_Indices);
        
        vector<double> getPosition(void){ return(currPos.getMotion()); }
        vector<vector<double>> getLEDs(void){ return(currPos.getOutputs()); } // Get outputs from final LED position
        double getError(void){return(currentError);}
        double getRandFactor(void){return(randomizeFactor);}

        vector<vector<double>> getTestAngles(){
            // Calculate and save initial error
            testPosition = currPos.getMotion();
            testPos.setTransformation(testPosition);
            error_cameraPosition();
            return(LED_TestAng);
        }

        vector<vector<double>> get_ang_line_set(){return(ang_line_set);}
};

#endif