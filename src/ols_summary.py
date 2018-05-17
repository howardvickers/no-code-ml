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
    # coefs = results.params
    pvals = round(results.pvalues, 5)
    # pvals = results.pvalues
    print('coefs:', coefs.to_dict())
    print('pvals:', pvals.to_dict())

    df_results = pd.concat([coefs, pvals], axis=1)
    df_results.columns = ['Coefficients', 'P Values']
    # coefs_pvals_dict = df_results.to_dict()
    # print('coefs_pvals_dict: ', coefs_pvals_dict)
    num_rows = df_results.shape[0]
    coefs_pvals = df_results.head(num_rows)
    # print('coefs_pvals:', coefs_pvals)
    # print('df_results: ', df_results)
    # print(results.summary())
    # print('pvals: ', pvals)
    # print('coefs: ', coefs)
    # print('type(pvals): ', type(pvals))
    # print('type(coefs): ', type(coefs))
    # print('df_results: ', df_results)
    # print('type(df_results): ', type(df_results))

    # return results.summary().as_html(), coefs_pvals.to_html()
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


# def get_imps_names():
#     _, feat_imps, cols = get_feat_imps()
#
#     # col_dict = nice_column_names()
#     X_train_dot_columns = cols
#     something = [col_dict.get(x, x) for x in X_train_dot_columns]
#     imps, names = zip(*sorted(zip(feat_imps, [col_dict.get(x) for x in X_train_dot_columns])))
#     return imps, names

def create_feat_imp_chart(X, y):
    # imps, names = get_imps_names()
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
