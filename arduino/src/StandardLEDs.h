#ifndef CONFIG_STANDARD_H
#define CONFIG_STANDARD_H

#include <Arduino.h>

// How long to wait before changing LED states during celebration (mS)
#define FLASH_DELAY 25

// Init LED data
void LED_setup(){
    for(uint8_t ii=2; ii<6; ii++){
        pinMode(ii, OUTPUT);
        digitalWrite(ii, HIGH);
    }
}

// Set specific LED Value
void setLEDval(uint8_t led_position, uint8_t led_brightness){
    if(led_brightness > 0){ // Bit select each output pin to the multiplexer
        digitalWrite(5, (0b00000001 & led_position) > 0);
        digitalWrite(4, (0b00000010 & led_position) > 0);
        digitalWrite(3, (0b00000100 & led_position) > 0);
        digitalWrite(2, (0b00001000 & led_position) > 0);
    }
    else{   // Set all pins high so only max pin is on
        digitalWrite(5, HIGH);
        digitalWrite(4, HIGH);
        digitalWrite(3, HIGH);
        digitalWrite(2, HIGH);
    }
}


// Blink lights in fun pattern
void celebration(){
    for(uint8_t ii=0; ii<12; ii++){
        setLEDval(ii, 255);
        delay(FLASH_DELAY);
    }
    for(uint8_t ii=0; ii<12; ii++){
        setLEDval(ii, 255);
        delay(FLASH_DELAY);
    }
}



#endif