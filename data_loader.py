import numpy as np
import csv
import os

def load_ratings():
    cwd = os.getcwd()
    dir_path = os.path.join(cwd, "training_set")
    for file in os.listdir(dir_path):
        if file.endswith(".txt"): #sanity check
            filepath = os.path.join(dir_path, file)
            with open(filepath) as csvfile:
                reader = csv.reader(csvfile, delimiter = ",")
                for row in reader:
                    print(row)

        break
            

load_ratings()

