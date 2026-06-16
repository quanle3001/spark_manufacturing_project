import argparse
import time

from arduino_sorter import ArduinoSorter


def main():
    parser = argparse.ArgumentParser(description="Manual Arduino servo sorter test.")
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--baudrate", type=int, default=9600)
    parser.add_argument("--enabled", action="store_true", help="Actually connect and send commands.")
    args = parser.parse_args()

    sorter = ArduinoSorter(
        port=args.port,
        baudrate=args.baudrate,
        enabled=args.enabled,
        cooldown_seconds=1.0,
    )

    sorter.connect()

    print("Testing servo push in 2 seconds...")
    time.sleep(2)

    sorter.push_bad()

    time.sleep(2)
    sorter.reset()
    sorter.close()


if __name__ == "__main__":
    main()
