#include <Arduino.h>

// // Uncomment one of these to specify which rover is being flashed
// const char ROVER_ID[] =  "TATE__";
// #include "StandardLEDS.h"

// const char ROVER_ID[] =  "LIZA__";
// #include "StandardLEDS.h"

static char ROVER_ID[] =  "BROCK_";
#include "LED_Strips.h"

#define COLLISION_DISTANCE 100 // Distance to stop to prevent collisions

// Constants for most recently referenced message
uint8_t messageBytes[5];
uint8_t checkSum;
uint8_t checkSumRead;


// VL53L0X
#include "VL53L0X.h"
#include <Wire.h>
const size_t VL53L0X_count = 3; // How many sensors
const uint8_t VL53L0X_xShut[] = {11, 12, 13}; // VL53L0X shutdown pins (for setting addresses individually)
const int VL53L0X_address[] = {0x30, 0x31, 0x32}; // New addresses to set sensors to
VL53L0X  VL53L0X_lox[3]; // Library sensor object
uint16_t VL53L0X_data[3]; // Library sensor object

// VL53L0X Setup
void VL53L0X_config(){
  // Iterate through sensors and set low
  for(size_t ii=0; ii<VL53L0X_count; ii++){
    // VL53L0X_lox[ii] = Adafruit_VL53L0X();
    pinMode(VL53L0X_xShut[ii], OUTPUT);
    digitalWrite(VL53L0X_xShut[ii], LOW); // Set shutdown pin LOW to disable
  }
  delay(10); // Pause to let VL53L0X start up

  // // Reset all sensors by setting to low for 10 mS and then turning back on
  // for(size_t ii=0; ii<VL53L0X_count; ii++) digitalWrite(VL53L0X_xShut[ii], LOW);
  // delay(10);

  // Activate sensors individually and turn on
  for(size_t ii=0; ii<VL53L0X_count; ii++){
    digitalWrite(VL53L0X_xShut[ii], HIGH); // Pull shutdown pin HIGH to enable sensor
    delay(10); // Pause to let VL53L0X wake up

    // Set address of VL53L0X sensor
    VL53L0X_lox[ii].setAddress(VL53L0X_address[ii]);
  }

  
  delay(10);
  // Initialize LV53L0X
  for(size_t ii=0; ii<VL53L0X_count; ii++){
    VL53L0X_lox[ii].init();
    VL53L0X_lox[ii].setTimeout(500);
  }

}

void readSensors(){
  // Iterate through sensors
  for(size_t ii=0; ii<VL53L0X_count; ii++){
    VL53L0X_data[ii] = VL53L0X_lox[ii].readRangeSingleMillimeters();
  }
}

void readSingleSensor(size_t ii){
    VL53L0X_data[ii] = VL53L0X_lox[ii].readRangeSingleMillimeters();
}


// Motors

// Motor Pins
#define M1A A0
#define M1B A1
#define M2A A2
#define M2B A3

// Setup motor control pins
void motorConfig(){
  // Init motor control pins as outputs
  pinMode(M1A, OUTPUT);
  pinMode(M1B, OUTPUT);
  pinMode(M2A, OUTPUT);
  pinMode(M2B, OUTPUT);

  // Init as low
  digitalWrite(M1A, LOW);
  digitalWrite(M1B, LOW);
  digitalWrite(M2A, LOW);
  digitalWrite(M2B, LOW);
}

void roverDrive(uint8_t driveSel, uint16_t runDur){
  uint32_t startTime = millis();

  uint8_t driveSel_1 = 0;
  uint8_t driveSel_2 = 0;

  switch (driveSel)
  {
  // Go Forward
  case 1:
    driveSel_1 = M1B;
    driveSel_2 = M2B;
    break;
    
  // Go Backward
  case 2:
    driveSel_1 = M1A;
    driveSel_2 = M2A;
    break;
    
  // Turn Left
  case 4:
    driveSel_1 = M1A;
    driveSel_2 = M2B;
    break;

  // Turn Left
  case 8:
    driveSel_1 = M1B;
    driveSel_2 = M2A;
    break;
  
  default: // Default to forward
    driveSel_1 = M1A;
    driveSel_2 = M2A;
    break;
  }

  uint8_t currIter = 0;

  digitalWrite(driveSel_1, HIGH);
  digitalWrite(driveSel_2, HIGH);

  while(true){
    // Check time 
    uint32_t currTime = millis();
    if(currTime - startTime >= runDur) break;

    // Detect collision
    if(currIter < 3){
      readSingleSensor(currIter);
      if(VL53L0X_data[currIter] < COLLISION_DISTANCE) break;
    }

    // readSensors();
    // for(size_t ii=0; ii<VL53L0X_count; ii++){
    //   if(VL53L0X_data[ii] < COLLISION_DISTANCE) break;
    // }

    // Do PWM
    if(currIter == 1){
      digitalWrite(driveSel_1, LOW);
      digitalWrite(driveSel_2, LOW);
    }
    else{
      digitalWrite(driveSel_1, HIGH);
      digitalWrite(driveSel_2, HIGH);
    }

    currIter += 1;
    if(currIter > 2) currIter = 0;
  }

  digitalWrite(driveSel_1, LOW);
  digitalWrite(driveSel_2, LOW);
}



//HC06

// #define HC06_AT 11 // Pin to switch to AT mode

// // Configure Bluetooth serial setup
// void bluetoothSerialConfig(){
//   pinMode(HC06_AT, OUTPUT);
//   digitalWrite(HC06_AT, LOW);

//   // TODO: use AT command mode to set name and password on HC05 module to rover ID
//   // https://www.keuwl.com/electronics/rduino/bluet/09-baud-rate/
// }




void setup() {
  delay( 1000 ); // Power-up safety delay

  motorConfig(); // Setup motor output pins

  Serial.begin(9600); // Default communication rate of the Bluetooth module
  Wire.begin(); // Init wire (I2C) communication

  LED_setup(); // Init LEDs from config function

  VL53L0X_config(); // Sensor init

  // Wait until serial port opens for native USB devices
  while (! Serial) {
    delay(1);
  }
  
  
  // celebration();
}



uint8_t led_position = 0;
uint8_t led_brightness = 0;

void loop() {
  // Read all Serial commands
  while(Serial.available() >= 5){
    checkSum = 0;
    for(size_t ii=0; ii<4; ii++){
      messageBytes[ii] = Serial.read();
      checkSum += messageBytes[ii];
    }
    checkSumRead = Serial.read();



    if(checkSum == checkSumRead){
        // Turn on LED
      if(messageBytes[0] == 1){
          led_position = messageBytes[1];
          led_brightness = messageBytes[2];
          setLEDval(led_position, led_brightness);
      }
        // Drive
      else if(messageBytes[0] == 2){
        // Serial.write(0xFFFFFF);
        roverDrive(messageBytes[1], messageBytes[2]*256+messageBytes[3]);
      }
      // Do light show
      else if(messageBytes[0] == 4){
        celebration();
      }

      // The remainder are serial callbacks, send checkSum before doing action

      // Lidar Data Request
      else if(messageBytes[0] == 8){
        readSensors(); // Load data into VL53L0X_data
        
        Serial.write(checkSum);

        uint8_t checkSum_2 = 0;
        for(size_t ii=0; ii<3; ii++){
          Serial.write(VL53L0X_data[ii]%256); // Send LSB
          checkSum_2 += VL53L0X_data[ii]%256;

          Serial.write(VL53L0X_data[ii]/256); // Send MSB
          checkSum_2 += VL53L0X_data[ii]/256;
        }
        Serial.flush();
        continue; // Do not finish loop (which would resend checksum), bail from here
      }
    
      // Chip ID Request
      else if(messageBytes[0] == 16){
        // uint8_t checkSum_2 = 0;
        Serial.write(checkSum);
        
        for(size_t ii=0; ii<6; ii++){
          Serial.write(ROVER_ID[ii]); // Send character ID
          // checkSum_2 += ROVER_ID[ii];
        }
        Serial.flush();
        continue; // Do not finish loop (which would resend checksum), bail from here
      }
    }
    else{ // Checksum error, clear buffer
      delay(1);
      while (Serial.available() > 0) Serial.read();
    }
    
    Serial.write(checkSum);
    Serial.flush();
  }
  
  // delay(10);
}