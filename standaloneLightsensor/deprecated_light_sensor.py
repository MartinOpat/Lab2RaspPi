import RPi.GPIO as GPIO
import time
import pandas as pd
import numpy as np

__author__ = 'Gus (Adapted from Adafruit)'
__license__ = "GPL"
__maintainer__ = "pimylifeup.com"

GPIO.setmode(GPIO.BOARD)

# define the pin that goes to the circuit
pin_to_circuit = 16

# datbase to store the date in
res = []


def rc_time(pin_to_circuit):
    count = 0

    # Output on the pin for
    GPIO.setup(pin_to_circuit, GPIO.OUT)
    GPIO.output(pin_to_circuit, GPIO.LOW)
    precision = 0.01
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
        res.append(ans)
except KeyboardInterrupt:
    pd.DataFrame(np.array(res)).to_csv("test_run.csv")
finally:
    GPIO.cleanup()