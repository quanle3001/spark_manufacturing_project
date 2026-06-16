"""
Arduino servo sorter helper.

Safe behavior:
- Arduino is disabled by default.
- If pyserial is missing or Arduino is not connected, detection can still run.
- Servo command is only sent when push_bad() is called.
"""

import time


class ArduinoSorter:
    def __init__(self, port="/dev/ttyACM0", baudrate=9600, enabled=False, cooldown_seconds=1.0):
        self.port = port
        self.baudrate = baudrate
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        self.serial = None
        self.last_push_time = 0.0

    def connect(self):
        if not self.enabled:
            print("[ArduinoSorter] Disabled. No serial connection opened.")
            return False

        try:
            import serial
        except ImportError:
            print("[ArduinoSorter] pyserial is not installed. Run: pip3 install pyserial")
            self.enabled = False
            return False

        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2.0)
            print(f"[ArduinoSorter] Connected on {self.port}")
            self.ping()
            return True
        except Exception as exc:
            print(f"[ArduinoSorter] Could not connect on {self.port}: {exc}")
            self.enabled = False
            self.serial = None
            return False

    def send(self, command):
        if not self.enabled or self.serial is None:
            print(f"[ArduinoSorter] Disabled. Would send: {command}")
            return False

        try:
            message = (command.strip().upper() + "\n").encode("utf-8")
            self.serial.write(message)
            self.serial.flush()
            return True
        except Exception as exc:
            print(f"[ArduinoSorter] Failed to send '{command}': {exc}")
            return False

    def read_response(self):
        if not self.enabled or self.serial is None:
            return None

        try:
            if self.serial.in_waiting > 0:
                return self.serial.readline().decode("utf-8", errors="ignore").strip()
        except Exception:
            return None

        return None

    def ping(self):
        self.send("PING")
        time.sleep(0.2)
        response = self.read_response()
        if response:
            print(f"[ArduinoSorter] Arduino response: {response}")
        return response

    def push_bad(self):
        now = time.time()

        if now - self.last_push_time < self.cooldown_seconds:
            print("[ArduinoSorter] Push ignored because cooldown is active.")
            return False

        self.last_push_time = now
        print("[ArduinoSorter] Sending BAD command.")
        return self.send("BAD")

    def reset(self):
        print("[ArduinoSorter] Sending RESET command.")
        return self.send("RESET")

    def close(self):
        if self.serial is not None:
            try:
                self.serial.close()
                print("[ArduinoSorter] Serial connection closed.")
            except Exception:
                pass

        self.serial = None
