import config
import util
import light_sensor
import random

from logzero import logger, logfile
from sense_hat import SenseHat
from gpiozero import LightSensor
from datetime import datetime, timedelta
from time import sleep

sh = SenseHat()
ldr = LightSensor(config.PIN)

# Configure logging
logfile(util.path_for(config.LOGFILE))

# Configure runtime
start_time = datetime.now()
end_time = start_time + timedelta(minutes=config.RUNTIME)

logger.info(f'Starting logging at {start_time}')
logger.info(f'Will log for {config.RUNTIME} minutes ({end_time})')

# Configure CSV
csvfile = util.path_for_data(1)
logger.info(f'Logging to {csvfile}')

while True:

    pixel1 = random.choice(range(0, 7 + 1))
    pixel2 = random.choice(range(0, 7 + 1))
    r = random.choice(range(0, 255 + 1))
    g = random.choice(range(0, 255 + 1))
    b = random.choice(range(0, 255 + 1))
    sh.set_pixel(pixel1, pixel2, (r, g, b))

    # Check for end time
    now = datetime.now()
    if now >= end_time:
        logger.info(f'Finished run at {now}')
        break

    # Main loop
    try:
        sh.set_pixel()  # to show that working

        orientation = sh.get_orientation_degrees()
        compass = sh.get_compass()
        compass_raw = sh.get_compass_raw()
        gyro = sh.get_gyroscope()
        gyro_raw = sh.get_gyroscope_raw()
        accelerometer_raw = sh.get_accelerometer_raw()
        intensity = light_sensor.capture_intensity(ldr)

        util.add_csv_data(csvfile, (
            now,
            intensity,
            sh.get_humidity(),
            sh.get_temperature(),
            sh.get_temperature_from_humidity(),
            sh.get_temperature_from_pressure(),
            sh.get_pressure(),
            orientation['roll'],
            orientation['pitch'],
            orientation['yaw'],
            compass,
            compass_raw['x'],
            compass_raw['y'],
            compass_raw['z'],
            gyro['roll'],
            gyro['pitch'],
            gyro['yaw'],
            gyro_raw['x'],
            gyro_raw['y'],
            gyro_raw['z'],
            accelerometer_raw['x'],
            accelerometer_raw['y'],
            accelerometer_raw['z'],
        ))
    except Exception as e:
        logger.error('{}: {})'.format(e.__class__.__name__, e))

    sleep(config.SLEEPTIME)