#include <Arduino.h>

#include "config.h"

// #include <FastLED.h>
// #define LED_TYPE    WS2811
// #define LED_PIN    6
// #define NUM_LEDS    18
// #define COLOR_ORDER GRB
// CRGB leds[NUM_LEDS];


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


// VL53L0X

#include "Adafruit_VL53L0X.h"
#include <Wire.h>
const size_t VL53L0x_count = 2; // How many sensors
const uint8_t VL53L0x_xShut[] = {11, 12, 13}; // VL53L0X shutdown pins (for setting addresses individually)
const int VL53L0x_address[] = {0x30, 0x31, 0x32}; // New addresses to set sensors to
Adafruit_VL53L0X VL53L0x_lox[2]; // Library sensor object
// Adafruit_VL53L0X lox_1;
// Adafruit_VL53L0X lox_2;
// Adafruit_VL53L0X lox_3;

// VL53L0X Setup
void VL53L0X_config(){
  Serial.println("\nSetting up VL53L0X");

  // VL53L0x_lox[1] = &lox_1;
  // VL53L0x_lox[2] = &lox_2;
  // VL53L0x_lox[3] = &lox_3;

  // Iterate through sensors and set low
  for(size_t ii=0; ii<VL53L0x_count; ii++){
    // VL53L0x_lox[ii] = Adafruit_VL53L0X();
    pinMode(VL53L0x_xShut[ii], OUTPUT);
    digitalWrite(VL53L0x_xShut[ii], LOW); // Set shutdown pin LOW to disable
  }
  delay(10); // Pause to let VL53L0X start up

  // // Reset all sensors by setting to low for 10 mS and then turning back on
  // for(size_t ii=0; ii<VL53L0x_count; ii++) digitalWrite(VL53L0x_xShut[ii], LOW);
  // delay(10);

  Serial.println("AAAAAAA");
  // Activate sensors individually and turn on
  for(size_t ii=0; ii<VL53L0x_count; ii++){
    digitalWrite(VL53L0x_xShut[ii], HIGH); // Pull shutdown pin HIGH to enable sensor
    delay(10); // Pause to let VL53L0X wake up

    // Begin VL53L0X sensor and check if online
    if(VL53L0x_lox[ii].begin(VL53L0x_address[ii], false, &Wire, Adafruit_VL53L0X::VL53L0X_SENSE_DEFAULT)){
    // if( VL53L0x_lox[ii].begin(VL53L0x_address[ii], false, &Wire, Adafruit_VL53L0X::VL53L0X_SENSE_DEFAULT) ){
      Serial.print("VL53L0x_lox[");
      Serial.print(ii);
      Serial.print("] Success at ");
      Serial.println(VL53L0x_address[ii]);
    }
    else{
      Serial.print("VL53L0x_lox[");
      Serial.print(ii);
      Serial.print("] Failure at ");
      Serial.println(VL53L0x_address[ii]);
    }
  }

}

void readSensors(){
  // Iterate through sensors
  for(size_t ii=0; ii<VL53L0x_count; ii++){
    int measuredRange = VL53L0x_lox[ii].readRange();
    Serial.print(ii);
    Serial.print(": ");
    Serial.println(measuredRange);
  }

}




//HC06

#define HC06_AT 11 // Pin to switch to AT mode

// Configure Bluetooth serial setup
void bluetoothSerialConfig(){
  pinMode(HC06_AT, OUTPUT);
  digitalWrite(HC06_AT, LOW);

  // TODO: use AT command mode to set name and password on HC05 module to rover ID
  // https://www.keuwl.com/electronics/rduino/bluet/09-baud-rate/
}




void setup() {
  delay( 1000 ); // Power-up safety delay

  motorConfig(); // Setup motor output pins

  Serial.begin(9600); // Default communication rate of the Bluetooth module
  Wire.begin(); // Init wire (I2C) communication

  if(USE_BLUETOOTH_SERIAL) bluetoothSerialConfig(); // if using bluetooth serial, config HC06 Module

  // Init LED pins as outputs (Pins 2->9)
  for(size_t ii=2; ii<10; ii++){
    pinMode(ii, OUTPUT);
    digitalWrite(ii, LOW);
  }

  VL53L0X_config(); // Sensor init

  // Wait until serial port opens for native USB devices
  while (! Serial) {
    delay(1);
  }
  
  // FastLED.addLeds<LED_TYPE, LED_PIN, COLOR_ORDER>(leds, NUM_LEDS).setCorrection( TypicalLEDStrip ); // Set up LED strip
}

uint8_t led_position = 0;


// uint8_t readByte(){
//   uint8_t outVal = Serial.read();
//   // Serial.write(outVal);
//   return(outVal);
// }

uint8_t messageBytes[4];
uint8_t checkSum;
uint8_t checkSumRead;

void loop() {
  // VL53L0X_RangingMeasurementData_t measure;
    
  // Serial.print("Reading a measurement... ");
  // lox.rangingTest(&measure, false); // pass in 'true' to get debug data printout!

  // if (measure.RangeStatus != 4) {  // phase failures have incorrect data
  //   Serial.print("Distance (mm): "); Serial.println(measure.RangeMilliMeter);
  // } else {
  //   Serial.println(" out of range ");
  // }
    
  // delay(100);






  readSensors();
  Serial.println("");
  while(Serial.available() > 0){ Serial.write(Serial.read()); }
  Serial.println("");
  delay(500);



  // while(Serial.available() >= 5){
  //   checkSum = 0;
  //   for(size_t ii=0; ii<4; ii++){
  //     messageBytes[ii] = Serial.read();
  //     checkSum += messageBytes[ii];
  //   }
  //   checkSumRead = Serial.read();
    

  //   if(checkSum == checkSumRead){
  //     led_position = messageBytes[0];
  //     leds[led_position].r = messageBytes[1];
  //     leds[led_position].g = messageBytes[2];
  //     leds[led_position].b = messageBytes[3];

  //     FastLED.show(); // apply the function on led strip
  //   }
  //   else{
  //     delay(1);
  //     while (Serial.available()) Serial.read();
  //   }
    
  //   Serial.write(checkSum);    
  // }
  




  // delay(10);
}