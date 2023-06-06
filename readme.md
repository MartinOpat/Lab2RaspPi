# Optic Gyroscope experiment
- This repository contains all relevant code that was necessary to successfully setup, 
run and process.

- More detailed description of the experiment can be found [here](https://www.overleaf.com/read/ztjhhqszvwpp) .


## Light Sensor Code
- `light_sensor.py` code is based on [Official RaspberryPi Website](https://projects.raspberrypi.org/en/projects/physical-computing/10)
and [Official GPIO Zero Documentation](https://gpiozero.readthedocs.io/en/stable/api_input.html).


## SenseHat code
- `config.py`, `main.py` and `util.py` are based on [astropi-team-todo](https://github.com/MartinOpat/astropi).


## Data
- All data can be found as `.csv` files in its corresponding folder in `data/Data_{SetupType}Setup_Attempt{num}/`
- Each `.csv` file corresponds to one 40 seconds long measurement.
- `data_processing.py` files inside of each specific folder was used to process and generate figures for 
    all (and only) data from inside this folder (i.e. the data from one specific setup).
- `data_processing_main.py` is the base template for all `data_processing.py` files.
- `data_processing_statistical_tests.ipynb` jupyter notebook was used to determine if significant
    statistical difference is present between measured data points for $\omega = 0$ and $\omega \neq 0$


## Misc.
 - `data/copy_data.sh` bash script (originally run from the RaspberryPi) can be used to transfer data from the repository
    to a local machine (originally ssh-ed into the RaspberryPi).
 - `data/send_data.sh` bash script (originally run from the RaspberryPi) can be used to send data from the repository
    to a specified e-mail address. (Note: previous setup of corresponding packages is necessary)