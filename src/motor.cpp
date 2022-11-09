#include "Arduino.h"
#include "motor.h"
#include <Adafruit_MotorShield.h>

MotorController::MotorController(int motorNumber, int speed, int _sensorPin) {
    Adafruit_MotorShield AFMS = Adafruit_MotorShield();
    motor = AFMS.getMotor(motorNumber);
    motor->setSpeed(speed);
    pinMode(sensorPin, INPUT);
    sensorPin = _sensorPin;
}

void MotorController::setSpeed(int speed) {
    motor->setSpeed(speed);
}

void MotorController::run(int direction) {
    motor->run(direction);
}

// position based PID control
void MotorController::PID_iteration(long targetDistance, int direction) {
    // PID loop iteration

    sensorState = digitalRead(sensorPin) > 512; // true for black, false for white
    if (sensorState != previousSensorState) {
      actualDistance += R * PI / n; // two state changes per tooth rotation
    }
    previousSensorState = sensorState;

    error = targetDistance - actualDistance;
    integral = integral + error;
    derivative = error - previous_error;
    previous_error = error;
    double speed = Kp * error + Ki * integral + Kd * derivative;
    motor->setSpeed(speed);
    motor->run(direction);
}

void MotorController::move(double distance, int direction) {
    // Move the motor a certain distance
    while (actualDistance < distance) {
        PID_iteration(distance, direction);
    }
    stop();
}

void MotorController::stop() {
    motor->run(RELEASE);
    // reset PID variables
    integral = 0;
    derivative = 0;
    previous_error = 0;
    actualDistance = 0;
    sensorState = false;
    previousSensorState = false;
}


