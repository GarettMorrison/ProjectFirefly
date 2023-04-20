/* file : EvoSim.c */
  
#include <stdio.h>
#include <math.h>
#include <iostream>
  
//our header file
#include "FastFireFly.hh"

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

    size_t randCount = rand()%5 +2; // How many times to randomize a value
    for(size_t ii=0; ii<randCount; ii++){
        int randIndex = rand()%6;
        if(randIndex < 3){
            testPosition[randIndex] += randomizeFactor * (2*(double)rand()/RAND_MAX -1); // Randomize position value
        }
        else{   // Randomize rotation value
            testPosition[randIndex] += PI_VALUE*(2*randomizeFactor/default_randomizeFactor) * (2*(double)rand()/RAND_MAX -1);
            // limit rotations to +- PI
            while(testPosition[randIndex] > PI_VALUE) testPosition[randIndex] -= 2*PI_VALUE;
            while(testPosition[randIndex] < -PI_VALUE) testPosition[randIndex] += 2*PI_VALUE;
        }
    }

    // Make sure position is in front of camera
    if(testPosition[0] < 0.0) testPosition[0] *= -1.0;
    
    // Prevent hitting local optimal by centering on camera
    if(testPosition[0] < mininumValueX) testPosition[0] += mininumValueX;
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






// Init class for localization
ledLocalizationFast::ledLocalizationFast(vector<vector<double>> _LED_Set, vector<double> startPos){
    // Load starting position
    LED_Set = _LED_Set;
    vector<double> defaultMotion = startPos;

    currPos = transformationSet(LED_Set);
    currPos.setTransformation(defaultMotion);

    testPos = transformationSet(LED_Set);
    testPos.setTransformation(defaultMotion);

    // Initialize srand
    srand (time(NULL));
}


// Calculate error of current test position
double ledLocalizationFast::error_cameraAngles(){
    // Calculate center of lantern in image
    double lanternOrigin_Z = atan( testPosition[2] / testPosition[0] );
    double lanternOrigin_Y = atan( testPosition[1] / testPosition[0] );

    // Calculate image angles from test position
    testError = 0;
    for(size_t ii=0; ii<LED_indices.size(); ii++){
        unsigned int fooIndex = LED_indices[ii]; // Get corresponding LED data array

        // Calculate test angle for each set of position points
        LED_TestAng[0][ii] = atan( testPos[2][fooIndex] / testPos[0][fooIndex] );
        LED_TestAng[1][ii] = atan( testPos[1][fooIndex] / testPos[0][fooIndex] );

        // printf("fooIndex:%ld     testPos:(%4.4f, %4.4f, %4.4f)     LED_TestAng:(%4.4f, %4.4f) \n", fooIndex, testPos[0][fooIndex], testPos[1][fooIndex], testPos[2][fooIndex], LED_TestAng[0][ii], LED_TestAng[1][ii]);

        double A = -(LED_TestAng[1][ii] - lanternOrigin_Z);
        double B = (LED_TestAng[0][ii] - lanternOrigin_Y);
        double C = -lanternOrigin_Z*B -lanternOrigin_Y*A;
        
        testError += abs(A*LED_TestAng[0][ii] + B*LED_TestAng[1][ii] + C) / (pow(A, 2) + pow(B, 2));

        // testError += pow(LED_TestAng[0][ii]-ang_set[0][ii], 2) + pow(LED_TestAng[1][ii]-ang_set[1][ii], 2); // Point distance sum

        // printf("%6.6f, %6.6f     %6.6f, %6.6f     \n", LED_TestAng[0][ii], LED_TestAng[1][ii], ang_set[0][ii], ang_set[1][ii]);
    }
    prevErrorReal = false;
    return(testError);
}

// Calculate error of current test position
double ledLocalizationFast::error_cameraPosition(){
    // // Calculate center of lantern in image
    // double lanternOrigin_Z = atan( testPosition[2] / testPosition[0] );
    // double lanternOrigin_Y = atan( testPosition[1] / testPosition[0] );

    // Calculate image angles from test position
    testError = 0;
    for(size_t ii=0; ii<LED_indices.size(); ii++){
        unsigned int fooIndex = LED_indices[ii]; // Get corresponding LED data array

        // Calculate test angle for each set of position points
        if(testPos[0][fooIndex] > 0.0001){
            LED_TestAng[0][ii] = atan( testPos[2][fooIndex] / testPos[0][fooIndex] );
            LED_TestAng[1][ii] = atan( testPos[1][fooIndex] / testPos[0][fooIndex] );
        }
        else{
            LED_TestAng[0][ii] = 3.14159/2;
            LED_TestAng[1][ii] = 3.14159/2;
        }
        
        testError += pow(ang_set[0][ii] - LED_TestAng[1][ii], 2) + pow(ang_set[1][ii] - LED_TestAng[0][ii], 2);

        // printf("%6.6f, %6.6f     %6.6f, %6.6f     \n", LED_TestAng[0][ii], LED_TestAng[1][ii], ang_set[0][ii], ang_set[1][ii]);
    }
    return(testError);
}

// Fit to position and run localization
vector<double> ledLocalizationFast::fitData_imageCentric(vector<vector<double>> _ang_set, vector<unsigned int> _LED_indices, size_t randomizeCount){
    ang_set = _ang_set;
    LED_indices = _LED_indices;
    
    LED_TestAng = ang_set;

    ang_line_set = _ang_set;
    for(size_t ii=0; ii<LED_indices.size(); ii++){
        ang_line_set[0][ii] = tan(ang_set[0][ii]);
        ang_line_set[1][ii] = tan(ang_set[1][ii]);
    }

    randomizeFactor = default_randomizeFactor; // load default randomization factor
    
    // Calculate and save initial error
    testPosition = currPos.getMotion();
    testPos.setTransformation(testPosition);
    error_cameraPosition();
    currentError = testError;
    bool hasImproved = 0;

    for(size_t ii=0; ii<randomizeCount; ii++){
        randomizeTestPosition();
        testPos.setTransformation(testPosition);
        error_cameraPosition();
        
        if(testError < currentError){ // New best found
            currentError = testError; // Save best error
            currPos.setTransformation(testPos.getMotion()); // Save best position

            // printf("%10.6f   %6.6f   ", testError, randomizeFactor); // Print error and randomize factor

            randomizeFactor = randomizeFactor*2; // Increase margin on improvement
            hasImproved = 1;
            
            // printf("%4ld   %4.6f      %6.3f     !!!!!!!!!!!!!!!!!!    %4.6f\n", ii, testError, randomizeFactor, currentError);
        }
        else{
            if(hasImproved) randomizeFactor = randomizeFactor*0.99; // Decrease margin if no improvement
            
            // Exit early if acceptable
            // if(randomizeFactor < acceptableRandomizerVal) return currentPosition;
            if(currentError < acceptableError) return currPos.getMotion();
            // printf("%4ld   %4.6f      %6.3f\n", ii, testError, randomizeFactor);
            
        }

        
        
    }

    // printf("Error: %5.4f      randomizeFactor: %5.4f      \n\n", currentError, randomizeFactor);

    return(currPos.getMotion());
}



double linePtDistanceSquared(double xVect, double yVect, double zVect, double xPos, double yPos, double zPos){
    return(
        ( pow(yVect*zPos - zVect*yPos, 2)
        + pow(zVect*xPos - xVect*zPos, 2) 
        + pow(xVect*yPos - yVect*xPos, 2) )
        / (pow(xVect, 2) + pow(yVect, 2) + pow(zVect, 2))
    );
}

// Calculate error of current test position
double ledLocalizationFast::error_3DpointLine(){
    double fooError = 0;
    testError = 0;

    for(size_t ii=0; ii<LED_indices.size(); ii++){
        int LED_Index = LED_indices[ii];
        fooError = linePtDistanceSquared( 1.0, ang_line_set[1][ii], ang_line_set[0][ii], testPos[0][LED_Index], testPos[1][LED_Index], testPos[2][LED_Index] );
        testError += fooError;
    }
    
    return(testError);
}

// Fit to position and run localization
vector<double> ledLocalizationFast::fitData_3D(vector<vector<double>> _ang_set, vector<unsigned int> _LED_indices, size_t randomizeCount){
    ang_set = _ang_set;
    LED_indices = _LED_indices;
    LED_dropValue = _LED_indices;
    
    LED_TestAng = ang_set;

    ang_line_set = _ang_set;
    for(size_t ii=0; ii<LED_indices.size(); ii++){
        LED_dropValue[ii] = 0;
        ang_line_set[0][ii] = tan(ang_set[0][ii]);
        ang_line_set[1][ii] = tan(ang_set[1][ii]);
    }

    randomizeFactor = default_randomizeFactor; // load default randomization factor
    
    // Calculate and save initial error
    testPosition = currPos.getMotion();
    testPos.setTransformation(testPosition);
    error_3DpointLine();
    currentError = testError;

    for(size_t ii=0; ii<randomizeCount; ii++){
        randomizeTestPosition();
        testPos.setTransformation(testPosition);
        error_3DpointLine();
        
        if(testError < currentError){ // New best found
            currentError = testError; // Save best error
            currPos.setTransformation(testPosition); // Save best position

            randomizeFactor = randomizeFactor*2; // Increase margin on improvement
            
            // printf("%4.6f     %4.6f, %4.6f, %4.6f\n", currentError, testPosition[0], testPosition[1], testPosition[2]);

            // printf("%4ld   %4.6f      %6.3f     !!!!!!!!!!!!!!!!!!    %4.6f\n", ii, testError, randomizeFactor, currentError);
            if(currentError < acceptableError) return currPos.getMotion();
        }
        else{
            randomizeFactor = randomizeFactor*0.98; // Decrease margin if no improvement
            // printf("%4ld   %4.6f      %6.3f\n", ii, testError, randomizeFactor);
        }
    }

    // printf("Error: %5.4f      randomizeFactor: %5.4f      \n\n", currentError, randomizeFactor);

    return(currPos.getMotion());
}
