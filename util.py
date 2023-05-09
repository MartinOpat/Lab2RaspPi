import os
import csv

dir_path = os.path.dirname(os.path.realpath(__file__))

def path_for(filename):
    return os.path.join(dir_path, filename)

def path_for_data(number):
    return path_for(f'data{number:02}.csv')

def create_csv_file(data_file, header):
    "Create a new CSV file and add the header row"
    with open(data_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)

def add_csv_data(data_file, data):
    "Add a row of data to the data_file CSV"
    with open(data_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(data)
        