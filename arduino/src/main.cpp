#include <Arduino.h>
#include <FastLED.h>

#define LED_TYPE    WS2811
#define LED_PIN    6
#define NUM_LEDS    18
#define COLOR_ORDER GRB

CRGB leds[NUM_LEDS];





void setup() {
  delay( 1000 ); // power-up safety delay
  Serial.begin(4800); // Default communication rate of the Bluetooth module
  FastLED.addLeds<LED_TYPE, LED_PIN, COLOR_ORDER>(leds, NUM_LEDS).setCorrection( TypicalLEDStrip ); // Set up LED strip
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
  // if(Serial.available() > 0) Serial.write(Serial.read());

  while(Serial.available() >= 5){
    checkSum = 0;
    for(size_t ii=0; ii<4; ii++){
      messageBytes[ii] = Serial.read();
      checkSum += messageBytes[ii];
    }
    checkSumRead = Serial.read();
    

    if(checkSum == checkSumRead){
      led_position = messageBytes[0];
      leds[led_position].r = messageBytes[1];
      leds[led_position].g = messageBytes[2];
      leds[led_position].b = messageBytes[3];

      FastLED.show(); // apply the function on led strip
    }
    else{
      delay(1);
      while (Serial.available()) Serial.read();
    }
    
    Serial.write(checkSum);    
  }
  

  // delay(10);
}