/* file : EvoSim.c */
  
#include <stdio.h>
#include <math.h>
#include <iostream>
  
//our header file
#include "FastFireFly.h"

#define PI_VALUE 3.14159265

// Randomly modify test position according to randomizeFactor
// testPosition values are {X position, Y position, Z position, X rotation, Y rotation, Z rotation}
void ledLocalizationFast::randomizeTestPosition(){
    testPosition = currentPosition; // Load current as default

    size_t randCount = rand()%6 +1; // How many times to randomize a value
    for(size_t ii=0; ii<randCount; ii++){
        int randIndex = rand()%6;
        if(randIndex < 3) testPosition[randIndex] += randomizeFactor * (2*(double)rand()/RAND_MAX -1); // Randomize position value
        else testPosition[randIndex] += PI_VALUE*(randomizeFactor/default_randomizeFactor) * (2*(double)rand()/RAND_MAX -1); // Randomize rotation value
    }

    // Make sure position is in front of camera
    if(testPosition[2] < 0.0) testPosition[2] *= -1.0;
    
    // Prevent hitting local optimal by centering on camera
    if(testPosition[2] < mininumValueZ) testPosition[2] += mininumValueZ;
}


// Randomly modify test position according to randomizeFactor
// testPosition values are {X position, Y position, Z position, X rotation, Y rotation, Z rotation}
void ledLocalizationFast::randomizeRotation(){
    testPosition = currentPosition; // Load current as default
    size_t randCount = rand()%4 +1; // How many times to randomize a value
    for(size_t ii=0; ii<randCount; ii++){
        int randIndex = rand()%3 +3;
        testPosition[randIndex] += PI_VALUE*(randomizeFactor/default_randomizeFactor) * (2*(double)rand()/RAND_MAX -1); // Randomize rotation value
    }
}

// Calculate LED_Testpos_XYZ arrays from testPosition and LED_Set_XYZ
void ledLocalizationFast::calculateLedTestPositions(){
    // Load rotations as short names to max rotationMatrix at least a little legible
    double A = testPosition[3];
    double B = testPosition[4];
    double C = testPosition[5];

    // Calculate rotation matrix factors
    vector<double> rotationMatrixA{cos(C)*cos(B), -sin(C)*cos(A) + cos(C)*sin(A)*sin(B), sin(C)*sin(A)+cos(C)*sin(B)*cos(A)};
    vector<double> rotationMatrixB{sin(C)*cos(B), cos(C)*cos(A) + sin(C)*sin(B)*sin(A), -cos(C)*sin(A)+sin(C)*sin(B)*cos(A)};
    vector<double> rotationMatrixC{-sin(B), cos(B)*sin(A), cos(B)*cos(A)};

    // Iterate through LED_TestPos, calculating each rotation and translation
    for(size_t ii=0; ii<LED_TestPos_X.size(); ii++){
        LED_TestPos_X[ii] = testPosition[0] + LED_Set_X[ii]*rotationMatrixA[0] + LED_Set_Y[ii]*rotationMatrixB[0] + LED_Set_Z[ii]*rotationMatrixC[0];
        LED_TestPos_Y[ii] = testPosition[1] + LED_Set_X[ii]*rotationMatrixA[1] + LED_Set_Y[ii]*rotationMatrixB[1] + LED_Set_Z[ii]*rotationMatrixC[1];
        LED_TestPos_Z[ii] = testPosition[2] + LED_Set_X[ii]*rotationMatrixA[2] + LED_Set_Y[ii]*rotationMatrixB[2] + LED_Set_Z[ii]*rotationMatrixC[2];
    }
}



double linePtDistanceSquared(double xVect, double yVect, double zVect, double xPos, double yPos, double zPos){
    return (pow(yVect*zPos - zVect*yPos, 2) + pow(xVect*zPos - zVect*xPos, 2) + pow(xVect*yPos - yVect*xPos, 2)) / (pow(xVect, 2) + pow(yVect, 2) + pow(zVect, 2));
}



// Calculate error of current test position
double ledLocalizationFast::calculateError(){
    testError = 0;

    for(size_t ii=0; ii<InputVect_LED_ID.size(); ii++){
        int LED_Index = InputVect_LED_ID[ii];

        testError += linePtDistanceSquared( InputVect_X[ii], InputVect_Y[ii], InputVect_Z[ii], LED_TestPos_X[LED_Index], LED_TestPos_Y[LED_Index], LED_TestPos_Z[LED_Index] );
    }
    ////////// TODO implement InputVect_LED_ID

    return(testError);
}


// Init class for localization
ledLocalizationFast::ledLocalizationFast(vector<double> _LED_X, vector<double> _LED_Y, vector<double> _LED_Z, double _pos_x, double _pos_y, double _pos_z){
    // Load starting position
    currentPosition[0] = _pos_x;
    currentPosition[1] = _pos_y;
    currentPosition[2] = _pos_z;

    // Save LED position
    LED_Set_X = _LED_X;
    LED_Set_Y = _LED_Y;
    LED_Set_Z = _LED_Z;

    LED_TestPos_X = _LED_X;
    LED_TestPos_Y = _LED_Y;
    LED_TestPos_Z = _LED_Z;

    // Initialize srand
    srand (time(NULL));
}

// Fit to position and run localization
vector<double> ledLocalizationFast::fitPositionToVectors(vector<double> _Vect_X, vector<double> _Vect_Y, vector<double> _Vect_Z, vector<double> _Vect_S, vector<double> _LED_Indices){
    InputVect_X = _Vect_X;
    InputVect_Y = _Vect_Y;
    InputVect_Z = _Vect_Z;
    InputVect_S = _Vect_S;
    InputVect_LED_ID = _LED_Indices;

    randomizeFactor = default_randomizeFactor; // load default randomization factor
    
    // Calculate and save initial error
    testPosition = currentPosition;
    calculateLedTestPositions();
    calculateError();
    currentError = testError;
    bool hasImproved = 0;

    for(size_t ii=0; ii<randomizeCount; ii++){
        randomizeTestPosition();
        calculateLedTestPositions();
        calculateError();
        
        if(testError < currentError){ // New best found
            currentError = testError; // Save best error
            currentPosition = testPosition; // Save best position

            // printf("%10.6f   %6.6f\n", testError, randomizeFactor); // Print error and randomize factor

            // for(double ii : currentPosition) printf("%5.4f   ", ii);
            // printf("\n");

            randomizeFactor = randomizeFactor*2; // Increase margin on improvement
            hasImproved = 1;
        }
        else{
            if(hasImproved) randomizeFactor = randomizeFactor*0.99; // Decrease margin if no improvement
            
            // Exit early if acceptable
            // if(randomizeFactor < acceptableRandomizerVal) return currentPosition;
            if(currentError < acceptableError) return currentPosition;
            
        }
        
    }

    // printf("Error: %5.4f      randomizeFactor: %5.4f      \n\n", currentError, randomizeFactor);

    return(currentPosition);
}




// // Fit to position and run localization
// vector<double> ledLocalizationFast::regressionFitLessRandom(vector<double> _Vect_X, vector<double> _Vect_Y, vector<double> _Vect_Z, vector<double> _LED_Indices){
//     InputVect_X = _Vect_X;
//     InputVect_Y = _Vect_Y;
//     InputVect_Z = _Vect_Z;
//     InputVect_LED_ID = _LED_Indices;

//     randomizeFactor = default_randomizeFactor; // load default randomization factor
    
//     // Calculate and save initial error
//     testPosition = currentPosition;
//     calculateLedTestPositions();
//     calculateError();
//     currentError = testError;

//     for(size_t ii=0; ii<randomizeCount; ii++){
//         bool newBestFound = 0;
//         for(int xTrans=-1; xTrans<2; xTrans += 2){
//             for(int yTrans=-1; yTrans<2; yTrans += 1){
//                 for(int zTrans=-1; zTrans<2; zTrans += 1){
//                     // Modify variables
//                     testPosition = currentPosition;

//                     if(xTrans == 1) testPosition[0] *= randomizeFactor;
//                     else if(xTrans == -1) testPosition[0] *= randomizeFactor;
                    
//                     if(yTrans == 1) testPosition[1] *= randomizeFactor;
//                     else if(yTrans == -1) testPosition[1] *= randomizeFactor;
                    
//                     if(zTrans == 1) testPosition[2] *= randomizeFactor;
//                     else if(zTrans == -1) testPosition[2] *= randomizeFactor;


//                     // Make sure position is in front of camera
//                     if(testPosition[2] < 0.0) testPosition[2] *= -1.0;
//                     // Prevent hitting local optimal by centering on camera
//                     if(testPosition[2] < mininumValueZ) testPosition[2] += mininumValueZ;


//                     calculateLedTestPositions();
//                     calculateError();
//                     if(testError < currentError){ // New best found
//                         newBestFound = 1; // New best found, increase range after this loop
//                         currentError = testError; // Save best error
//                         currentPosition = testPosition; // Save best position
//                     }


//                     // Randomize rotation and check again
//                     for(ii=0; ii<3; ii++){
//                         randomizeRotation();
//                         calculateLedTestPositions();
//                         calculateError();
//                         if(testError < currentError){ // New best found
//                             newBestFound = 1; // New best found, increase range after this loop
//                             currentError = testError; // Save best error
//                             currentPosition = testPosition; // Save best position
//                         }
//                     }
                    
//                 }
//             }
//         }

//         // printf("%5.4f\n", randomizeFactor);

//         if(newBestFound){
//             randomizeFactor = randomizeFactor*2; // Increase margin on improvement
//         }
//         else{
//             randomizeFactor = randomizeFactor*6/7; // Decrease margin if no improvement

//             if(randomizeFactor < acceptableRandomizerVal) return currentPosition; // Exit early if acceptable
//         }


//         for(size_t jj=0; jj<10; jj++){
//             randomizeRotation();
//             calculateLedTestPositions();
//             calculateError();
            
//             if(testError < currentError){ // New best found
//                 currentError = testError; // Save best error
//                 currentPosition = testPosition; // Save best position
//             }
//         }
//     }

//     return(currentPosition);
// }