#include "Arduino.h"
#include "motor.h"

void setup() {
  Serial.begin(9600);
}

void loop() {
  Serial.println("Hello World!");
  delay(1000);
}