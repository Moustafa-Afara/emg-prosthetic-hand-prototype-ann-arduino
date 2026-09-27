/*
  Five-servo prosthetic hand — serial grip receiver.

  Receives one line per grip change from the PC controller (controller/hand_controller.py):
      G <thumb> <index> <middle> <ring> <little>\n      angles 0..180 (0 = open, 180 = closed)
  and moves the five finger servos. Replies "OK a1 a2 a3 a4 a5" or "ERR <reason>".

  Wiring (Arduino Uno/Nano): servo signal pins 3, 5, 6, 9, 10. Pins 0 and 1 are left free for
  the USB serial link. Power the servos from a separate 5-6 V supply with a common ground —
  five servos can draw more current than the board's 5 V pin can give.

  Status: compiles for the Arduino Uno (avr-gcc, Arduino core 1.8.19, Servo 1.2.2);
  NOT tested on hardware.
*/
#include <Servo.h>

const uint8_t N_FINGERS = 5;
const uint8_t SERVO_PINS[N_FINGERS] = {3, 5, 6, 9, 10};   // thumb, index, middle, ring, little
const uint8_t OPEN_ANGLE[N_FINGERS] = {0, 0, 0, 0, 0};     // start with the hand open

Servo fingers[N_FINGERS];
char line[40];
uint8_t len = 0;

void apply(const char *msg) {
  int a[N_FINGERS];
  // expected: "G a b c d e"
  if (msg[0] != 'G' || sscanf(msg + 1, "%d %d %d %d %d", &a[0], &a[1], &a[2], &a[3], &a[4]) != N_FINGERS) {
    Serial.println(F("ERR format: G a1 a2 a3 a4 a5"));
    return;
  }
  Serial.print(F("OK"));
  for (uint8_t i = 0; i < N_FINGERS; i++) {
    a[i] = constrain(a[i], 0, 180);
    fingers[i].write(a[i]);
    Serial.print(' ');
    Serial.print(a[i]);
  }
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  for (uint8_t i = 0; i < N_FINGERS; i++) {
    fingers[i].attach(SERVO_PINS[i]);
    fingers[i].write(OPEN_ANGLE[i]);
  }
  Serial.println(F("READY"));
}

void loop() {
  // Non-blocking line reader: no Serial.readString() timeout, so a command acts immediately.
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      if (len > 0) {
        line[len] = '\0';
        apply(line);
        len = 0;
      }
    } else if (len < sizeof(line) - 1) {
      line[len++] = c;
    } else {
      len = 0;  // overlong line: drop it
      Serial.println(F("ERR line too long"));
    }
  }
}
