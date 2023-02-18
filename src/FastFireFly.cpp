/* file : EvoSim.c */
  
#include <stdio.h>
#include <math.h>
#include <iostream>
  
//our header file
#include "FastFireFly.h"

#define PI_VALUE 3.14159265


transformationSet::transformationSet(vector<vector<double>> _input){
    inputs = _input;
    outputs = _input;
    transformationFactors = {0, 0, 0, 0, 0, 0};
}

transformationSet::transformationSet(){
    inputs = {{0}, {0}, {0}};
    outputs = {{0}, {0}, {0}};
    transformationFactors = {0, 0, 0, 0, 0, 0};
}

vector<vector<double>> transformationSet::setTransformation(vector<double> _factors){
    transformationFactors = _factors;

    // Load rotations as short names to make rotationMatrix at least a little legible
    double A = transformationFactors[3];
    double B = transformationFactors[4];
    double C = transformationFactors[5];

    // Calculate rotation matrix factors
    vector<double> rotationMatrixA{ cos(B)*cos(C),  sin(A)*sin(B)*cos(C) -cos(A)*sin(C),  cos(A)*sin(B)*cos(C) +sin(A)*sin(C) };
    vector<double> rotationMatrixB{ cos(B)*sin(C),  sin(A)*sin(B)*sin(C) +cos(A)*cos(C),  cos(A)*sin(B)*sin(C) -sin(A)*cos(C) };
    vector<double> rotationMatrixC{ -sin(B),  sin(A)*cos(B),  cos(A)*cos(B) };


    // Iterate through input, calculating each rotation and translation
    for(size_t ii=0; ii<inputs[0].size(); ii++){
        outputs[0][ii] = inputs[0][ii]*rotationMatrixA[0] + inputs[1][ii]*rotationMatrixA[1] + inputs[2][ii]*rotationMatrixA[2] + transformationFactors[0];
        outputs[1][ii] = inputs[0][ii]*rotationMatrixB[0] + inputs[1][ii]*rotationMatrixB[1] + inputs[2][ii]*rotationMatrixB[2] + transformationFactors[1];
        outputs[2][ii] = inputs[0][ii]*rotationMatrixC[0] + inputs[1][ii]*rotationMatrixC[1] + inputs[2][ii]*rotationMatrixC[2] + transformationFactors[2];
    }

    return(outputs);
}









// Randomly modify test position according to randomizeFactor
// testPosition values are {X position, Y position, Z position, X rotation, Y rotation, Z rotation}
void ledLocalizationFast::randomizeTestPosition(){
    testPosition = currPos.getMotion(); // Load current as default

    size_t randCount = rand()%6 +1; // How many times to randomize a value
    for(size_t ii=0; ii<randCount; ii++){
        int randIndex = rand()%6;
        if(randIndex < 3) testPosition[randIndex] += randomizeFactor * (2*(double)rand()/RAND_MAX -1); // Randomize position value
        else{   // Randomize rotation value
            testPosition[randIndex] += PI_VALUE*(randomizeFactor/default_randomizeFactor) * (2*(double)rand()/RAND_MAX -1);
            // limit rotations to +- PI
            while(testPosition[randIndex] > PI_VALUE) testPosition[randIndex] -= 2*PI_VALUE;
            while(testPosition[randIndex] < -PI_VALUE) testPosition[randIndex] += 2*PI_VALUE;
        }
    }

    // Make sure position is in front of camera
    if(testPosition[2] < 0.0) testPosition[2] *= -1.0;
    
    // Prevent hitting local optimal by centering on camera
    if(testPosition[2] < mininumValueZ) testPosition[2] += mininumValueZ;
}


// Randomly modify test position according to randomizeFactor
// testPosition values are {X position, Y position, Z position, X rotation, Y rotation, Z rotation}
void ledLocalizationFast::randomizeRotation(){
    testPosition = currPos.getMotion(); // Load current as default
    size_t randCount = rand()%4 +1; // How many times to randomize a value
    for(size_t ii=0; ii<randCount; ii++){
        int randIndex = rand()%3 +3;
        testPosition[randIndex] += PI_VALUE*(randomizeFactor/default_randomizeFactor) * (2*(double)rand()/RAND_MAX -1); // Randomize rotation value
    }
}


double linePtDistanceSquared(double xVect, double yVect, double zVect, double xPos, double yPos, double zPos){
    return(
        ( pow(yVect*zPos - zVect*yPos, 2)
        + pow(zVect*xPos - xVect*zPos, 2) 
        + pow(xVect*yPos - yVect*xPos, 2) )
        / (pow(xVect, 2) + pow(yVect, 2) + pow(zVect, 2))
    );

    // return(
    //     ( pow(yVect*zPos - zVect*yPos, 2) 
    //     +pow(xVect*zPos - zVect*xPos, 2) 
    //     +pow(xVect*yPos - yVect*xPos, 2) ) 
    //     / (pow(xVect, 2) + pow(yVect, 2) + pow(zVect, 2))
    // );
}



// Calculate error of current test position
double ledLocalizationFast::calculateError(){
    testError = 0;

    for(size_t ii=0; ii<InputVect_LED_ID.size(); ii++){
        int LED_Index = InputVect_LED_ID[ii];
        testError += linePtDistanceSquared( InputVect_X[ii], InputVect_Y[ii], InputVect_Z[ii], testPos[0][LED_Index], testPos[1][LED_Index], testPos[2][LED_Index] ) * InputVect_S[ii];
    }
    
    return(testError);
}


// Init class for localization
ledLocalizationFast::ledLocalizationFast(vector<double> _LED_X, vector<double> _LED_Y, vector<double> _LED_Z, double _pos_x, double _pos_y, double _pos_z){
    // Load starting position
    LED_Set = {_LED_X, _LED_Y, _LED_Z};
    vector<double> defaultMotion = {_pos_x, _pos_y, _pos_z, 0, 0, 0};

    currPos = transformationSet(LED_Set);
    currPos.setTransformation(defaultMotion);

    testPos = transformationSet(LED_Set);
    testPos.setTransformation(defaultMotion);

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
    testPosition = currPos.getMotion();
    testPos.setTransformation(testPosition);
    calculateError();
    currentError = testError;
    bool hasImproved = 0;

    for(size_t ii=0; ii<randomizeCount; ii++){
        randomizeTestPosition();
        testPos.setTransformation(testPosition);
        calculateError();
        
        if(testError < currentError){ // New best found
            currentError = testError; // Save best error
            currPos.setTransformation(testPos.getMotion()); // Save best position

            // printf("%10.6f   %6.6f   ", testError, randomizeFactor); // Print error and randomize factor

            // for(double ii : currentPosition) printf("%5.4f   ", ii);
            // printf("\n");

            randomizeFactor = randomizeFactor*2; // Increase margin on improvement
            hasImproved = 1;
        }
        else{
            if(hasImproved) randomizeFactor = randomizeFactor*0.99; // Decrease margin if no improvement
            
            // Exit early if acceptable
            // if(randomizeFactor < acceptableRandomizerVal) return currentPosition;
            if(currentError < acceptableError) return currPos.getMotion();
            
        }
        
    }

    // printf("Error: %5.4f      randomizeFactor: %5.4f      \n\n", currentError, randomizeFactor);

    return(currPos.getMotion());
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