from gpiozero import LightSensor, Buzzer

ldr = LightSensor(4)  # alter if using a different pin
res = []

try:
    while True:
        val = ldr.value
        print(val)
        res.append(str(val))

except KeyboardInterrupt:
    f = open("test_run2.csv", "w")
    f.write(",\n".join(res))
    f.close()
