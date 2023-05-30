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
header_string = "now,intensity,humidity,temperature,temperature_from_humidity,"+ \
                              "temperature_from_pressure,pressure,orientation_roll,orientation_pitch,orientation_yaw,"+ \
                              "compass,compass_raw_x,compass_raw_y,compass_raw_z,gyro_roll,gyro_pitch,gyro_yaw,"+ \
                              "gyro_raw_x,gyro_raw_y,gyro_raw_z,accelerometer_raw_x,accelerometer_raw_y,"+ \
                              "accelerometer_raw_z,accelerometer_roll,accelerometer_pitch,accelerometer_yaw"
util.create_csv_file(csvfile, header_string.split(","))
logger.info(f'Logging to {csvfile}')

while True:
    # Check for end time
    now = datetime.now()
    if now >= end_time:
        logger.info(f'Finished run at {now}')
        break

    # Main loop
    try:
        # to show that working
        pixel1 = random.randint(0, 7)
        pixel2 = random.randint(0, 7)
        r = random.randint(0, 60)
        g = random.randint(0, 60)
        b = random.randint(0, 60)
        sh.clear()
        sh.set_pixel(pixel1, pixel2, (r, g, b))

        orientation = sh.get_orientation_degrees()
        compass = sh.get_compass()
        compass_raw = sh.get_compass_raw()
        gyro = sh.get_gyroscope()
        gyro_raw = sh.get_gyroscope_raw()
        accelerometer = sh.get_accelerometer()
        accelerometer_raw = sh.get_accelerometer_raw()
        intensity = light_sensor.capture_intensity(ldr)

        print(f'time={now}, intensity={intensity}, '
              f'acc_x={accelerometer_raw["x"]}, acc_y={accelerometer_raw["y"]}, acc_z={accelerometer_raw["z"]}')

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
            accelerometer['roll'],
            accelerometer['pitch'],
            accelerometer['yaw']
        ))
        sleep(config.SLEEPTIME)
    except KeyboardInterrupt:
        print("Exited successfully!")
        sh.clear()
        break
    except Exception as e:
        logger.error('{}: {})'.format(e.__class__.__name__, e))
        sh.clear()
        break

sh.clear()
