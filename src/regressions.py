import numpy as np
import pandas as pd
import os

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression as LR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.svm import SVR as SVR
from sklearn.linear_model import ElasticNet as EN
from sklearn.ensemble import GradientBoostingRegressor as GBR


def calc_rmse(yhat, y):
    return np.sqrt(((yhat-y)**2).mean())

def eval_model(model, X_train, y_train, X_test, y_test):
    ypred = model.predict(X_test)
    ytrainpred = model.predict(X_train)
    r2score = round(r2_score(y_test, ypred), 2)
    rmse_train  = round(calc_rmse(ytrainpred, y_train), 2)
    rmse_test   = round(calc_rmse(ypred, y_test), 2)

    return rmse_train, rmse_test, r2score

def run_models(X, y, models):
    print('models within run_models:', models)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    hyperpara_dict = {  LR  : {}, # use defaults only
                        RFR : { 'randomforestregressor__max_features' : ['auto'],
                                'randomforestregressor__max_depth': [10],
                                'randomforestregressor__bootstrap': [True],
                                'randomforestregressor__min_samples_leaf': [1, 4],
                                'randomforestregressor__min_samples_split': [2, 5],
                                'randomforestregressor__n_estimators': [10],
                                },
                        GBR : { 'gradientboostingregressor__n_estimators' :     [100],
                                'gradientboostingregressor__max_depth':         [5, 10],
                                'gradientboostingregressor__min_samples_split': [3, 10],
                                'gradientboostingregressor__learning_rate':     [0.01, 0.05, 0.1],
                                'gradientboostingregressor__loss':              ['ls'],
                                },
                        KNR : { 'kneighborsregressor__n_neighbors' : [1, 5, 10],
                                'kneighborsregressor__weights': ['uniform', 'distance'],
                                },
                        SVR : { 'svr__kernel': ['linear', 'rbf'],
                                'svr__C': [0.5, 10, 20],
                                'svr__epsilon': [0.1, 5, 10, 50],
                                },
                        EN : { 'elasticnet__alpha': [1, 0.1, 0.01], # equivalent to lambda; alpha=0 means no regularization, ie linear regression
                                'elasticnet__l1_ratio': [0.5, 0.7, 0.9], # l1=1 means L1 penalty, ie Lasso (not L2/Ridge)
                                'elasticnet__max_iter': [1000],
                                }
                        }

    model_names_dict = {'Linear Regression': LR, 'Random Forest': RFR, 'Gradient Boosting': GBR, 'K Neighbors': KNR, 'S V R': SVR, 'Elastic Net': EN}

    model_comparison_dict = {}

    for model_name in models:
        model = model_names_dict[model_name]

        print('model in run_models:', model)

        # data preprocessing (removing mean and scaling to unit variance with StandardScaler)
        pipeline = make_pipeline(   StandardScaler(),
                                    model()
                                 )

        hyperparameters = hyperpara_dict[model]

        clf = GridSearchCV(pipeline, hyperparameters, cv=3) # cv=3 is same as cv=None (default)

        clf.fit(X_train, y_train)

        print('best params for {}:'.format(model), clf.best_params_)
        r2score, rmse_train, rmse_test = eval_model(clf.best_estimator_, X_train, y_train, X_test, y_test)

        model_comparison_dict[model_name] = r2score, rmse_train, rmse_test

    return model_comparison_dict

if __name__ == '__main__':
    run_models()
