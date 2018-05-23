import numpy as np
import pandas as pd
import os
import statsmodels.api as sm
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor as RFR
import matplotlib.pyplot as plt
import time

def train_model(X, y):
    X = sm.add_constant(X)
    model = sm.OLS(y, X, missing='drop')
    results = model.fit()
    coefs = round(results.params, 2)
    pvals = round(results.pvalues, 5)
    print('coefs:', coefs.to_dict())
    print('pvals:', pvals.to_dict())

    df_results = pd.concat([coefs, pvals], axis=1)
    df_results.columns = ['Coefficients', 'P Values']
    num_rows = df_results.shape[0]
    coefs_pvals = df_results.head(num_rows)
    return coefs.to_dict(), pvals.to_dict()


def get_feat_imps(X, y):
    column_names = X.columns
    model = RFR(max_features        = 'auto',
                max_depth           = None,
                bootstrap           = True,
                min_samples_leaf    = 5,
                min_samples_split   = 10,
                n_estimators        = 100
                )
    model = model.fit(X, y)
    model_params    = model.get_params()
    feat_imps       = model.feature_importances_
    print('model_params', model_params)
    print('feat_imps', feat_imps)

    return model_params, feat_imps, column_names

def create_feat_imp_chart(X, y):
    _, imps, names = get_feat_imps(X, y)
    imps, names = zip(*sorted(zip(imps, names)))

    plt.style.use('bmh')
    plt.barh(range(len(names)), imps, align='center')
    plt.yticks(range(len(names)), names)
    plt.xlabel('Relative Importance of Features', fontsize=20)
    plt.ylabel('Features', fontsize=20)
    plt.tight_layout()
    ts = time.time()
    feat_imps_chart_url = 'static/images/feat_imps_{}.png'.format(ts)
    plt.savefig(feat_imps_chart_url)
    # plt.show()
    plt.clf()
    plt.cla()
    plt.close()
    return feat_imps_chart_url



if __name__ == '__main__':
    train_model()
