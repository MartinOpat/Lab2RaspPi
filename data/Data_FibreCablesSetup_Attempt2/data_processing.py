import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import scipy
from scipy.optimize import curve_fit
matplotlib.rcParams.update({'font.size': 20})


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

    data[i]["omega"] = abs(data[i]["gyro_raw_z"])


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


def plot_omega(data, i, ax, iax, iax2):
    data_int = pd.DataFrame.copy(data[i])
    if any(data[i]["omega"] >= 9.9):
        start = 6
        end = 12
        x_int = np.linspace(start, end, int((end - start) / 0.2))
        data_int = data_int[abs(data_int["omega"]) < 9.9]
        f_int, data_int = interpolate_omega(data_int)
        ax[iax][iax2].scatter(x_int, f_int(x_int), c="C0")

    ax[iax][iax2].scatter(data_int["time"], data_int["omega"], c="C0")
    ax[iax][iax2].title.set_text(f"Graph of rotational speed $\omega$ as a function of time")
    ax[iax][iax2].set_xlabel("time [s]")
    ax[iax][iax2].set_ylabel(f"$\omega$ [rad / s]")


def plot_intensity(data, i, ax, iax, iax2):
    ax[iax][iax2].errorbar(data[i]["time"], data[i]["intensity"], yerr=Ierr, fmt="o", capsize=5)
    ax[iax][iax2].title.set_text("Graph of intensity as a function of time")
    ax[iax][iax2].set_xlabel("time [s]")
    ax[iax][iax2].set_ylabel("Intensity [%]")


def plot_intensity_vs_omega(data, i, ax, iax, iax2):
    ax[iax][iax2].scatter(abs(data[i]["omega"]), data[i]["intensity"])
    ax[iax][iax2].title.set_text(f"Graph of intensity $I$ as a function of $\omega$")
    ax[iax][iax2].set_xlabel(f"|$\omega$| [rad / s]")
    ax[iax][iax2].set_ylabel("Intensity [%]")


def plot_rel_diff_intensity(data, i, ax, iax, iax2):
    ax[iax][iax2].scatter(data[i]["time"], data[i]["intensity_rel_diff"])
    ax[iax][iax2].title.set_text("Graphs of rel. diff. intensity as a function of time")
    ax[iax][iax2].set_xlabel("time [s]")
    ax[iax][iax2].set_ylabel(f"$\Delta ~ Intensity_{{rel}}$ [%]")


def plot_run(data, i):
    fig, ax = plt.subplots(1, 4, figsize=(48, 12))
    fig.suptitle(f"All relevant graphs for {i}th run")

    plot_omega(data, i, ax, 0, 0)
    plot_intensity(data, i, ax, 0, 1)
    plot_intensity_vs_omega(data, i, ax, 1, 0)
    plot_rel_diff_intensity(data, i, ax, 1, 1)

    plt.show()


def calculate_phi(data, i, I0):
    high_phi = lambda x: np.arccos(np.sqrt(x / I0))
    low_phi = lambda x: np.arccos(1)
    helper_func = lambda x: high_phi(x) if I0 > x else low_phi(x)
    #
    # data[i]["phi"] = np.arccos(np.sqrt(data[i]["intensity"] / I0))
    data[i]["phi"] = data[i]["intensity"].apply(helper_func)


def calculate_phi_errors(data, i, I0):
    I = data[i]["intensity"]
    # data[i]["phi_err"] = Ierr / (2 * np.sqrt(I * (I0 - I)))
    data[i]["phi_err"] = 2*data[i]["phi"]*I0*Ierr  # approximation


def calculate_omega_res(data, i):
    data[i]["omega_res"] = lambda_c * c * data[i]["phi"] / (2 * pi * L * D * n)


def calculate_omega_res_err(data, i):
    # data[i]["omega_res_err"] = lambda_c * c * data[i]["phi_err"] / (2 * pi * L * D * n)
    data[i]["omega_res_err"] =  data[i]["omega_res"] * np.sqrt((data[i]["phi_err"] / data[i]["phi"])**2 + (Derr / D)**2)

def plot_omega_res(data, i, ax, iax, iax2):
    ax[iax][iax2].errorbar(data[i]["time"], data[i]["omega_res"], yerr=data[i]["omega_res_err"], fmt="o", capsize=5)
    ax[iax][iax2].title.set_text(f"Graph of calculated rotational speed $\omega_{{res}}$ as a function of time")
    ax[iax][iax2].set_xlabel("time [s]")
    ax[iax][iax2].set_ylabel(f"$\omega_{{res}}$ [rad / s]")


def plot_phi(data, i, ax, iax, iax2):
    ax[iax][iax2].errorbar(data[i]["time"], data[i]["phi"], yerr=data[i]["phi_err"], fmt="o", capsize=5)
    ax[iax][iax2].title.set_text(f"Graph of phase shift $\Phi$ as a function of time")
    ax[iax][iax2].set_xlabel("time [s]")
    ax[iax][iax2].set_ylabel(f"$\Phi$ [rad]")


def plot_run_res(data, i):
    fig, ax = plt.subplots(2, 2, figsize=(24, 24))
    fig.suptitle(f"Graphs for one complete measurement using fibre-cables setup (run={i})", fontsize=40)

    ax[0][0].set_ylim([-1, 17])
    ax[1][0].set_ylim([-1, 17])
    ax[0][1].set_ylim([0.83, 0.93])
    ax[1][1].set_ylim([-0.01, 0.33])

    plot_omega(data, i, ax, 0, 0)
    plot_intensity(data, i, ax, 0, 1)
    plot_omega_res(data, i, ax, 1, 0)
    plot_phi(data, i, ax, 1, 1)

    plt.savefig(f"results/res{i}.png")
    plt.show()


def weighted_mean_squared_error(data, i):
    data_relevant = pd.DataFrame.copy(data[i][data[i]["omega_res"] > 1])
    y_true = data_relevant["omega"]
    y_pred = data_relevant["omega_res"]
    y_errors = data_relevant["omega_res_err"]
    weights = 1 / np.square(y_errors)

    return np.sum(weights * np.square(y_true - y_pred)) / np.sum(weights)



def calculate_I0(data, i):
    data[i]["intensity_diff"] = data[i]["intensity"].diff()
    diff_threshold = data[i]["intensity_diff"].std()
    mask = (data[i]["intensity_diff"].abs() < diff_threshold) & (data[i]["intensity_diff"].shift(-1).abs() < diff_threshold)
    average_intensity = data[i].loc[mask, "intensity"].mean()
    return average_intensity + Ierr

# consts
pi = np.pi
c = 299792458  # m / s
Ierr = 0.002

# consts laser
# R = 15 / 100  # m
lambda_c = 620 * 10**(-9)  # m
# A = 1.5*pi*R**2
L = 2  # m
D = 23.5 / 100  # m
Derr = 1 / 100  # m
n = 1.1  # number of turns


if __name__ == "__main__":
    num_runs = 3
    runs = load_data("")

    for i in range(num_runs):
        I0 = calculate_I0(runs, i)
        calculate_time(runs, i)
        calculate_omega3(runs, i)  # here error is negligible
        calculate_rel_diff_intensity(runs, i)
        calculate_phi(runs, i, I0)
        calculate_phi_errors(runs, i, I0)
        calculate_omega_res(runs, i)
        calculate_omega_res_err(runs, i)
        print(i, I0)
        print(weighted_mean_squared_error(runs, i))

        plot_run_res(runs, i)
        with open(f"results/precision{i}.txt", "w") as f:
            f.write(f"Curve evaluated precision is {weighted_mean_squared_error(runs, i)}\n")
            f.write(f"Calculated I0 = {I0}\n")



