from gpiozero import LightSensor


def capture_intensity(ldr: LightSensor):
    return ldr.value
