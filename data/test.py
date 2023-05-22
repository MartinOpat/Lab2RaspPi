import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from datetime import datetime

# Step 1: Load csv file
df = pd.read_csv('22May23/mirrors/data02.csv')

# Step 2: Convert "now" column to "time_ms"
df['now'] = pd.to_datetime(df['now'])
df['time_ms'] = (df['now'] - df['now'][0]).dt.total_seconds() * 1000

# Step 3: Plot "intensity" vs "time_ms"
plt.figure()
plt.plot(df['time_ms'], df['intensity'])
plt.xlabel('Time (ms)')
plt.ylabel('Intensity')
plt.title('Intensity vs Time')
plt.show()

# Step 4: Create new column "omega"
# Please note that, how you derive omega from accelerometer_yaw depends on your specific needs.
# Here, we are assuming it as a derivative which can be obtained by the 'diff' method in pandas.
df['accelerometer_yaw'] = df['accelerometer_yaw'].astype(float)  # Ensure the column data type is float
df['omega'] = df['accelerometer_yaw'].diff() / df['time_ms'].diff()  # Angular speed = rate of change of angle with respect to time

# Step 5: Plot "omega" vs "time_ms"
plt.figure()
plt.plot(df['time_ms'], df['omega'])
plt.xlabel('Time (ms)')
plt.ylabel('Omega')
plt.title('Omega vs Time')
plt.show()

# Step 6: Plot "intensity" vs "omega" and fit a line
def linear_func(x, a, b):
    return a * x + b

popt, pcov = curve_fit(linear_func, df['omega'][1:], df['intensity'][1:])  # Ignore the first row because 'omega' has NaN value there

plt.figure()
plt.scatter(df['omega'][1:], df['intensity'][1:], label='data')  # Plot the data
plt.plot(df['omega'][1:], linear_func(df['omega'][1:], *popt), 'r-', label='fit: a=%5.3f, b=%5.3f' % tuple(popt))  # Plot the fit
plt.xlabel('Omega')
plt.ylabel('Intensity')
plt.title('Intensity vs Omega')
plt.legend()
plt.show()
