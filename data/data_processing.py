import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.size': 18})

# consts
pi = np.pi
R = 10 / 100  # m
c = 299792458  # m / s
lambda_c = 620 * 10**(-9)  # m
A = pi*R**2

header = "now,intensity,humidity,temperature,temperature_from_humidity,temperature_from_pressure,pressure," \
         "orientation_roll,orientation_pitch,orientation_yaw,compass,compass_raw_x,compass_raw_y,compass_raw_z," \
         "gyro_roll,gyro_pitch,gyro_yaw,gyro_raw_x,gyro_raw_y,gyro_raw_z,accelerometer_raw_x,accelerometer_raw_y," \
         "accelerometer_raw_z,accelerometer_roll,accelerometer_pitch,accelerometer_yaw\n"


def append_headers(i_start, i_end):
    for i in range(i_start, i_end):
        if i < 10:
            f = open(f"data0{i}.csv", "r+")
        else:
            f = open(f"data{i}.csv", "r+")
        content = f.read()
        f.seek(0, 0)
        f.write(header + content)
        f.close()


def load_data(path):
    data = []
    for i in range(1, num_runs+1):
        if i < 10:
            filename = f"data0{i}.csv"
        else:
            filename = f"data{i}.csv"

        data.append(pd.read_csv(path+filename).dropna())
    return data


def calculate_time(data, i):
    data[i]["now"] = pd.to_datetime(data[i]["now"])
    data[i]["time"] = (data[i]['now'] - data[i]['now'][0]).dt.total_seconds()


def calculate_omega(data, i):
    if "time" not in data[i]:
        calculate_time(data, i)

    data[i]['accelerometer_yaw'] = data[i]['accelerometer_yaw'].apply(np.deg2rad)
    data[i]["accelerometer_yaw"] = np.unwrap(data[i]["accelerometer_yaw"])
    data[i]["omega"] = data[i]["accelerometer_yaw"].diff() / data[i]["time"].diff()


def calculate_omega2(data, i):
    if "time" not in data[i]:
        calculate_time(data, i)

    data[i]["yaw_rad"] = data[i]["gyro_yaw"].apply(np.deg2rad)
    data[i]["yaw_rad"] = np.unwrap(data[i]["yaw_rad"])
    data[i]["omega"] = data[i]["yaw_rad"].diff() / data[i]["time"].diff()


def calculate_omega3(data, i):
    if "time" not in data[i]:
        calculate_time(data, i)

    data[i]["omega"] = data[i]["gyro_raw_z"]


def calculate_rel_diff_intensity(data, i, I0=0.915):
    data[i]["intensity_rel_diff"] = (data[i]["intensity"] - I0) / I0


def plot_omega(data, i, ax, iax):
    ax[iax].scatter(data[i]["time"], data[i]["omega"])
    ax[iax].title.set_text("Graphs of rotational speed as a function of time")
    ax[iax].set_xlabel("time [s]")
    ax[iax].set_ylabel("Rotational speed [rad / s]")


def plot_intensity(data, i, ax, iax):
    ax[iax].scatter(data[i]["time"], data[i]["intensity"])
    ax[iax].title.set_text("Graphs of intensity as a function of time")
    ax[iax].set_xlabel("time [s]")
    ax[iax].set_ylabel("Intensity [%]")


def plot_intensity_vs_omega(data, i, ax, iax):
    ax[iax].scatter(abs(data[i]["omega"]), data[i]["intensity"])
    ax[iax].title.set_text("Graphs of intensity as a function of omega")
    ax[iax].set_xlabel("|omega| [rad / s]")
    ax[iax].set_ylabel("Intensity [%]")


def plot_rel_diff_intensity(data, i, ax, iax):
    ax[iax].scatter(data[i]["time"], data[i]["intensity_rel_diff"])
    ax[iax].title.set_text("Graphs of rel. diff. intensity as a function of time")
    ax[iax].set_xlabel("time [s]")
    ax[iax].set_ylabel(f"$\Delta ~ Intensity_{{rel}}$ [%]")


def plot_run(data, i):
    fig, ax = plt.subplots(1, 4, figsize=(48, 12))
    fig.suptitle(f"All relevel graphs for {i}th run")

    plot_omega(data, i, ax, 0)
    plot_intensity(data, i, ax, 1)
    plot_intensity_vs_omega(data, i, ax, 2)
    plot_rel_diff_intensity(data, i, ax, 3)

    plt.show()

if __name__ == "__main__":
    num_runs = 24
    runs = load_data("2June23/")

    for i in range(num_runs):
        calculate_time(runs, i)
        calculate_omega3(runs, i)
        calculate_rel_diff_intensity(runs, i)

    i_run = 23
    plot_run(runs, i_run)



