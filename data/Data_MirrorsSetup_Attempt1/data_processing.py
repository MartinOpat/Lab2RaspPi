import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import scipy
from scipy.optimize import curve_fit
matplotlib.rcParams.update({'font.size': 18})


header = "now,intensity,humidity,temperature,temperature_from_humidity," + \
                              "temperature_from_pressure,pressure,gyro_roll,gyro_pitch,gyro_yaw," + \
                              "gyro_raw_x,gyro_raw_y,gyro_raw_z\n"


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


def interpolate_curve(x, a, b, c):
    # return a*x + b
    return a * np.log(b * x) + c


def interpolate_omega(data_int):
    data = data_int[(data_int["time"] > 6) & (data_int["time"] < 15)]
    time = np.array(data["time"])
    omega = np.array(data["omega"])
    popt, pcov = curve_fit(
        interpolate_curve,
        time, omega
    )
    return lambda x: interpolate_curve(x, *popt), data_int


def plot_omega(data, i, ax, iax):
    start = 6
    end = 12
    x_int = np.linspace(start, end, int((end - start) / 0.2))
    data_int = pd.DataFrame.copy(data[i][abs(data[i]["omega"]) < 9.9])
    f_int, data_int = interpolate_omega(data_int)

    ax[iax].scatter(data_int["time"], data_int["omega"], c="C0")
    ax[iax].scatter(x_int, f_int(x_int), c="C0")
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
    fig.suptitle(f"All relevant graphs for {i}th run")

    plot_omega(data, i, ax, 0)
    plot_intensity(data, i, ax, 1)
    plot_intensity_vs_omega(data, i, ax, 2)
    plot_rel_diff_intensity(data, i, ax, 3)

    plt.show()


def calculate_phi(data, i, I0):
    data[i]["phi"] = np.arccos(np.sqrt(data[i]["intensity"] / I0))
    # data[i]["phi"] = abs(np.arccos(np.sqrt(data[i]["intensity"] / I0)) - 0.74)


def calculate_omega_res(data, i):
    # data[i]["omega_res"] = lambda_c * c * data[i]["phi"] / (8 * pi * A)
    data[i]["omega_res"] = lambda_c * c * data[i]["phi"] / (2 * pi * L * D * n)


def plot_omega_res(data, i, ax, iax):
    ax[iax].scatter(data[i]["time"], data[i]["omega_res"])
    ax[iax].title.set_text("Graphs of calculated rotational speed as a function of time")
    ax[iax].set_xlabel("time [s]")
    ax[iax].set_ylabel("Rotational speed [rad / s]")


def plot_phi(data, i, ax, iax):
    ax[iax].scatter(data[i]["time"], data[i]["phi"])
    ax[iax].title.set_text("Graphs of phi as a function of time")
    ax[iax].set_xlabel("time [s]")
    ax[iax].set_ylabel("Phi [rad]")


def plot_run_res(data, i):
    fig, ax = plt.subplots(1, 4, figsize=(48, 12))
    fig.suptitle(f"All relevant graphs for {i}th run")

    ax[0].set_ylim([-16, 1])
    ax[2].set_ylim([-1, 16])

    plot_omega(data, i, ax, 0)
    plot_intensity(data, i, ax, 1)
    plot_omega_res(data, i, ax, 2)
    plot_phi(data, i, ax, 3)

    plt.savefig("res.png")
    plt.show()


# consts
pi = np.pi
c = 299792458  # m / s

# consts laser
# R = 15 / 100  # m
lambda_c = 620 * 10**(-9)  # m
# A = 1.5*pi*R**2
L = 2  # m
D = 23.5 / 100  # m
n = 1.25  # number of turns

# consts glass tubes
# A = (9.3*12.25) / 10000  # m
# lambda_c = 532 * 10**(-9)  # m
# L = 4*pi*5.85/100 + 2*(9.3+12.25)/100  # m
# D = 15.38/100  # m


if __name__ == "__main__":
    num_runs = 24
    # num_runs = 17
    # num_runs = 3
    runs = load_data("Data_FibreCablesSetup_Attempt/")
    # runs = load_data("Data_GlassTubesSetup_Attempt1/")
    # runs = load_data("Data_FibreCablesSetup_Attempt2/")

    for i in range(num_runs):
        calculate_time(runs, i)
        calculate_omega3(runs, i)
        calculate_rel_diff_intensity(runs, i)
        calculate_phi(runs, i, 0.915)
        # calculate_phi(runs, i, 1.19)
        calculate_omega_res(runs, i)

    i_run = 23
    # plot_run(runs, i_run)
    plot_run_res(runs, i_run)



