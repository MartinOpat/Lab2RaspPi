import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

db = pd.read_csv("test_run.csv")
yVals = np.array(db["reading"])[100:]

dt = 0.01
xVals = np.linspace(0, dt*len(yVals), len(yVals))

plt.plot(xVals, yVals)
plt.show()

##############################
yVals = np.array(pd.read_csv("test_run2.csv"))
yVals = yVals[len(yVals)//2+435:]
xVals = np.linspace(0, dt*len(yVals), len(yVals))

plt.plot(xVals, yVals)
plt.show()

