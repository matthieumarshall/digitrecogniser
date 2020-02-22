import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

here = os.path.dirname(os.path.abspath(__file__))
path_to_train = os.path.join(here, "..", "data", "train.csv")
data = pd.read_csv(path_to_train)
print(data.head())


def print_digit(row_number, data):
    row = data.iloc[row_number, 1:].values
    row = row.reshape(28, 28).astype('uint8')
    plt.imshow(row)
    plt.show()

print_digit(2, data)