import numpy as np
import pandas as pd
import os
import statsmodels.api as sm
from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt


def train_model(X, y):
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing='drop')
    results = model.fit()

    print(results.summary())

    return results.summary().as_html()


# def get_data():
#     csv_file_path = '../data/df_reduced_cols.csv'
#     data = pd.read_csv(csv_file_path)
#     return data




if __name__ == '__main__':
    train_model()
