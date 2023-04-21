#ifndef CONFIG_STRIPS_H
#define CONFIG_STRIPS_H

#include <Arduino.h>



// LED Strip definitions
#include <FastLED.h>
#define LED_TYPE    WS2811
#define LED_PIN    6
#define NUM_LEDS    18
#define COLOR_ORDER GRB
CRGB leds[NUM_LEDS];



// How long to wait before changing LED states during celebration (mS)
#define FLASH_DELAY 100

// Init LED data
void LED_setup(){
  FastLED.addLeds<LED_TYPE, LED_PIN, COLOR_ORDER>(leds, NUM_LEDS).setCorrection( TypicalLEDStrip ); // Set up LED strip
}

// Set specific LED Value
void setLEDval(uint8_t led_position, uint8_t led_brightness){
    leds[led_position].r = led_brightness;
    leds[led_position].g = led_brightness;
    leds[led_position].b = led_brightness;
    FastLED.show(); // apply the function on led strip
}


// Blink lights in fun pattern
void celebration(){
    for(int pattern=0; pattern<3; pattern++){  
        for(int ii=0; ii<NUM_LEDS; ii++){
        if(ii%6 == pattern-0 || ii%6  == 5-pattern) leds[ii].g = 255;
        else leds[ii].g = 0;
        }

        FastLED.show(); // apply the function on led strip
        delay(FLASH_DELAY);
    }
    
    for(int ii=0; ii<NUM_LEDS; ii++) leds[ii].g = 0;

    for(int pattern=0; pattern<3; pattern++){  
        for(int ii=0; ii<NUM_LEDS; ii++){
        if(ii%6 == pattern-0 || ii%6  == 5-pattern) leds[ii].b = 255;
        else leds[ii].b = 0;
        }

        FastLED.show(); // apply the function on led strip
        delay(FLASH_DELAY);
    }
    
    for(int ii=0; ii<NUM_LEDS; ii++) leds[ii].b = 0;
  
    FastLED.show(); // apply the function on led strip
}



#endif