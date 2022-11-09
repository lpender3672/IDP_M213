#include "Arduino.h"
#include <Adafruit_MotorShield.h>


class MotorController {
  public:
    MotorController(int motorNumber, int speed, int _sensorPin);
    void setSpeed(int speed);
    void run(int direction);
    void stop();
    void PID_iteration(long targetDistance, int direction);
    void move(double distance, int direction);

    double actualDistance = 0;
    bool sensorState = false;
    bool previousSensorState = false;

  private:
    Adafruit_DCMotor *motor;
    int sensorPin;

    // Drive constants
    const double R = 50.0;
    const int n = 30; // teeth

    // PID constants
    const double Kp = 0.1;
    const double Ki = 0.01;
    const double Kd = 0.01;

    // PID variables
    double error = 0;
    double integral = 0;
    double derivative = 0;
    double previous_error = 0;

};