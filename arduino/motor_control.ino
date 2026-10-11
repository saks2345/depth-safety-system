/*
 * Motor cutoff controller for the depth-estimation safety system.
 *
 * Receives single characters from zeo.py over USB serial (9600 baud):
 *   'G' = GO   -> motor ON,  buzzer OFF
 *   'S' = STOP -> motor OFF, buzzer ON, LED ON
 *
 * Failsafe: zeo.py sends a command every frame. If nothing arrives for
 * FAILSAFE_TIMEOUT_MS (script crashed, cable unplugged, laptop froze),
 * the motor is cut. The motor is also OFF at power-up until the first 'G'.
 *
 * EDIT THE PINS BELOW TO MATCH YOUR WIRING.
 */

// ---- Configuration ----
const uint8_t  MOTOR_PIN   = 7;      // Relay IN pin or motor-driver enable pin
const uint8_t  BUZZER_PIN  = 8;      // Active buzzer (optional)
const uint8_t  LED_PIN     = LED_BUILTIN;
const bool     MOTOR_ACTIVE_LOW = true;   // true: relay modules that switch ON with LOW
                                          // false: transistor/driver that switches ON with HIGH
const unsigned long FAILSAFE_TIMEOUT_MS = 1500;
const long     BAUD_RATE   = 9600;   // Must match zeo.py

unsigned long lastCommandTime = 0;
bool motorOn = false;

void setMotor(bool on) {
  motorOn = on;
  digitalWrite(MOTOR_PIN, (on != MOTOR_ACTIVE_LOW) ? HIGH : LOW);
  digitalWrite(BUZZER_PIN, on ? LOW : HIGH);
  digitalWrite(LED_PIN, on ? LOW : HIGH);
}

void setup() {
  pinMode(MOTOR_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  setMotor(false);                   // safe state at boot
  Serial.begin(BAUD_RATE);
  lastCommandTime = millis();
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == 'G') {
      setMotor(true);
      lastCommandTime = millis();
    } else if (c == 'S') {
      setMotor(false);
      lastCommandTime = millis();
    }
    // any other byte is ignored
  }

  // Failsafe: no command from the PC -> stop
  if (motorOn && (millis() - lastCommandTime > FAILSAFE_TIMEOUT_MS)) {
    setMotor(false);
  }
}
