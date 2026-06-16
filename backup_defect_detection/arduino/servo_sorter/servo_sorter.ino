/*
  Servo Sorter for Raspberry Pi TensorFlow Lite Defect Detection

  Raspberry Pi sends serial commands over USB.

  Commands:
    BAD or PUSH  -> push bad object
    RESET        -> return to rest position
    PING         -> reply READY

  Wiring:
    Servo signal wire  -> Arduino pin 9
    Servo red wire     -> external 5V power
    Servo brown/black  -> external GND
    Arduino GND        -> external power GND

  Important:
    Do NOT power a medium/large servo directly from Raspberry Pi.
    Use external 5V servo power and connect all grounds together.
*/

#include <Servo.h>

const int SERVO_PIN = 9;

int REST_ANGLE = 20;
int PUSH_ANGLE = 95;

int PUSH_HOLD_MS = 450;
int RETURN_DELAY_MS = 250;
int COOLDOWN_MS = 1000;

Servo sorterServo;
unsigned long lastPushTime = 0;

String readLine() {
  String input = "";

  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '\n') {
      break;
    }

    if (c != '\r') {
      input += c;
    }

    delay(2);
  }

  input.trim();
  input.toUpperCase();
  return input;
}

void pushBadObject() {
  unsigned long now = millis();

  if (now - lastPushTime < (unsigned long)COOLDOWN_MS) {
    Serial.println("IGNORED_COOLDOWN");
    return;
  }

  lastPushTime = now;

  Serial.println("PUSH_START");

  sorterServo.write(PUSH_ANGLE);
  delay(PUSH_HOLD_MS);

  sorterServo.write(REST_ANGLE);
  delay(RETURN_DELAY_MS);

  Serial.println("PUSH_DONE");
}

void setup() {
  Serial.begin(9600);

  sorterServo.attach(SERVO_PIN);
  sorterServo.write(REST_ANGLE);

  delay(500);
  Serial.println("READY");
}

void loop() {
  if (Serial.available() > 0) {
    String command = readLine();

    if (command == "BAD" || command == "PUSH") {
      pushBadObject();
    }
    else if (command == "RESET") {
      sorterServo.write(REST_ANGLE);
      Serial.println("RESET_DONE");
    }
    else if (command == "PING") {
      Serial.println("READY");
    }
    else if (command.length() > 0) {
      Serial.print("UNKNOWN_COMMAND:");
      Serial.println(command);
    }
  }
}
