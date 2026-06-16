#include <Servo.h>

Servo myServo;

void setup() {
  Serial.begin(9600);      // Match the baud rate used in Python
  myServo.attach(9);       // Servo signal wire connected to Digital Pin 9
  myServo.write(90);        // Start at 0 degrees
}

void loop() {
  if (Serial.available() > 0) {
    char receivedChar = Serial.read();
    
    if (receivedChar == 'B') {
      myServo.write(45);   // Move to 90 degrees if a battery is detected
      delay(100);
    } 
    else if (receivedChar == 'C') {
      myServo.write(135);    // Return to 0 degrees if a cap is detected
      delay(100);
    }
  }
}
