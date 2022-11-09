#include "Arduino.h"
#include <Adafruit_MotorShield.h>
#include "motor.h"

MotorController left(1, 150, 0);
MotorController right(2, 150, 1);

void setup() {
  Serial.begin(9600);

  Serial.println("Motor PID test");

  while (left.actualDistance < 100 && right.actualDistance < 100) { // move 100mm
    left.PID_iteration(100, FORWARD);
    right.PID_iteration(100, FORWARD);
  }
  left.stop();
  right.stop();
}

void loop() {
  Serial.println("Hello World!");
  delay(1000);
}