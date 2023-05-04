import RPi.GPIO as GPIO
import time
import numpy as np

__author__ = 'Gus (Adapted from Adafruit)'
__license__ = "GPL"
__maintainer__ = "pimylifeup.com"

GPIO.setmode(GPIO.BOARD)

# define the pin that goes to the circuit
pin_to_circuit = 7

# datbase to store the date in
res = []


def rc_time(pin_to_circuit):
    count = 0

    # Output on the pin for
    GPIO.setup(pin_to_circuit, GPIO.OUT)
    GPIO.output(pin_to_circuit, GPIO.LOW)
    precision = 100
    time.sleep(precision)

    # Change the pin back to input
    GPIO.setup(pin_to_circuit, GPIO.IN)

    # Count until the pin goes high
    while (GPIO.input(pin_to_circuit) == GPIO.LOW):
        count += 1

    return count


# Catch when script is interupted, cleanup correctly
try:
    # Main loop
    while True:
        ans = rc_time(pin_to_circuit)
        res.append(str(ans))
except KeyboardInterrupt:
    f = open("test_run.csv", "w")
    f.write(",\n".join(res))
    f.close()
finally:
    GPIO.cleanup()