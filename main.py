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
sh.set_imu_config(True, False, False)  # Set to only use gyroscope
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
header_string = "now,intensity,humidity,temperature,temperature_from_humidity," + \
                              "temperature_from_pressure,pressure,gyro_roll,gyro_pitch,gyro_yaw," + \
                              "gyro_raw_x,gyro_raw_y,gyro_raw_z"
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
        r = random.randint(50, 60)
        g = random.randint(50, 60)
        b = random.randint(50, 60)
        sh.clear()
        sh.set_pixel(pixel1, pixel2, (r, g, b))

        gyro = sh.get_gyroscope()
        gyro_raw = sh.get_gyroscope_raw()
        intensity = light_sensor.capture_intensity(ldr)

        print(f'time={now}, intensity={intensity}')

        util.add_csv_data(csvfile, (
            now,
            intensity,
            sh.get_humidity(),
            sh.get_temperature(),
            sh.get_temperature_from_humidity(),
            sh.get_temperature_from_pressure(),
            sh.get_pressure(),
            gyro['roll'],
            gyro['pitch'],
            gyro['yaw'],
            gyro_raw['x'],
            gyro_raw['y'],
            gyro_raw['z']
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
