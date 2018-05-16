import global_y
from flask import Flask, make_response, request, render_template, jsonify
from flask_bootstrap import Bootstrap
from flask import Markup
import flask
import io
import csv
import pandas as pd
import numpy as np
import os
from ols_summary import train_model as ols_tm
from ols_summary import create_feat_imp_chart as cfic
from regressions import run_models as rm


import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def build_regression_results_table(dict):
    top = '<table class="table table-hover"> <thead> <tr> <th scope="col">Model</th> <th scope="col">RMSE (Train)</th> <th scope="col">RMSE (Test)</th> <th scope="col">R-Squared</th> </tr> </thead> <tbody>'
    middle = '<tr> <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td> </tr>'
    bottom =   '</tbody></table>'
    mid_rows = ""
    for k, v in dict.items():
        row = middle.format(k, v[0], v[1], v[2])
        mid_rows += row
    return top+mid_rows+bottom


def build_cols_rename_table(list):
    top = '<table class="table table-hover"><thead><tr><th scope="col">Old Name</th><th scope="col">New Name</th></tr></thead><tbody>'
    middle = '<tr> <td>{}</td> <td><input type="text" name="{}" placeholder="New Column Name"></td></tr>'
    bottom =   '</tbody></table>'
    mid_rows = ""
    for col in list:
        row = middle.format(col, col)
        mid_rows += row
    return top+mid_rows+bottom

def print_head(df):
    """converts pandas 'head' to html; returns html"""
    head = df.head().to_html()
    return Markup(head)

def drop_unnamed_col(df):
    """drops any unnamed columns; returns df"""
    unnamed_cols = ['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1']
    columns = df.columns
    for col in columns:
        if col in unnamed_cols:
            df.drop(col, axis=1, inplace=True)
    return df

def compare_models(data_url, y_name, models):
    """passes list of models to compare to GridSearchCV in rm(); returns model_comparison_dict"""
    # X, y = get_X_y(data_url, y_name)
    df = pd.read_csv(data_url)
    X = df.drop(y_name, axis=1)
    y = df[y_name]
    model_comparison_dict = rm(X, y, models)
    print('X.head(), y.head(), models: ', X.head(), y.head(), models)
    return model_comparison_dict

# def get_X_y(data_url, y_name):
#     """reloads data as csv and separates into two dataframes (X, y); returns X, y"""
#     df = pd.read_csv(data_url)
#     X = df.drop(y_name, axis=1)
#     y = df[y_name]
#     return X, y

app = Flask(__name__)

@app.route('/')
def index():
    return flask.render_template('index.html')


@app.route('/ml')
def ml():
    return flask.render_template('ml.html')

# @app.route('/upload', methods=["POST"])
# def upload():
#     f = request.files['data_file']
#     if not f:
#         return "No file"
#     df = pd.read_csv(f)
#     print('df.head(): ', df.head())
#     df.to_csv('../data/df.csv')
#     head2 = print_head(df)
#     print('the head2', head2)
#     columns = list(df.columns)
#
#     return flask.jsonify(head_table2 = head2)

@app.route('/uploadcsv', methods=["POST"])
def uploadcsv():
    print('hello uploadcsv')
    print('type (request)', type (request.files['data_file']))
    f = request.files['data_file']
    if not f:
        return "No file"
    df = pd.read_csv(f)
    print('df.head(): ', df.head())
    df.to_csv('../data/df.csv')
    head1 = print_head(df)
    print('the head', head1)
    columns = list(df.columns)
    # resp = flask.make_response(jsonify(head_table = head1), 200)
    # resp.mimetype = "text/javascript"
    # return resp



    # return flask.jsonify(head_table = head1)

    # return ("{'head_table': 'testv'}")
    models = ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'K Neighbors', 'S V R', 'Elastic Net']

    return flask.render_template(
                                'ml.html',
                                firsthead = head1,
                                columns = columns,
                                models = models
                                )


# @app.route('/head', methods=["POST"])
# def head():
#     f = request.files['data_file']
#     if not f:
#         return "No file"
#     df = pd.read_csv(f)
#     df.to_csv('../data/df.csv')
#     head = print_head(df)
#     columns = list(df.columns)
#
#     return flask.render_template(
#                                 'ml.html',
#                                 firsthead = head,
#                                 columns = columns
#                                 )

@app.route('/select_cols', methods=["POST", "GET"])
def select_cols():
    if request.method == 'POST':
        y_col = ''
        df = pd.read_csv('../data/df.csv')
        head1 = print_head(df)
        all_cols = list(df.columns)
        form_results = request.form
        print('input from form (select cols): ', request.form)
        cols_to_keep = list(form_results.keys())
        cols_to_keep_less_y = cols_to_keep.copy()
        print('This is cols_to_keep:', cols_to_keep)
        print('This is cols_to_keep_less_y (still with y): ', cols_to_keep_less_y)
        for k, v in form_results.items():
            print('This is k', k)
            print('This is v', v)
            if v == 'y':
                global_y.y_col_name = k
                cols_to_keep_less_y.remove(k)
                print('This is global_y.y_col_name:', global_y.y_col_name)
        print('Now here is cols_to_keep (without "y"):', cols_to_keep_less_y)
        cols_to_drop = set(all_cols) - set(cols_to_keep)
        print('cols_to_drop:', cols_to_drop)
        df_reduced_cols = df.drop(cols_to_drop, axis=1)
        print('Here is df_reduced_cols', df_reduced_cols)
        print('df_reduced_cols.columns', df_reduced_cols.columns)
        df_X = df_reduced_cols.drop(global_y.y_col_name, axis=1)
        head = print_head(df_reduced_cols)
        df_reduced_cols.to_csv('../data/df_reduced_cols.csv')
        columns = list(df_reduced_cols.columns)
        html_cols = build_cols_rename_table(columns)
        X = df_X
        y = df[global_y.y_col_name]
        feat_imps_chart_url = cfic(X.astype(float), y)
        feat_imps_chart = '<embed class="d-block w-100" src="../{}">'.format(feat_imps_chart_url)
        # feat_imps_chart = cfic(X.astype(float), y)

        ols_results, ols_coefs_pvals = ols_tm(X.astype(float), y)
        ols_summary = Markup(ols_results)
        ols_coefs_pvals = Markup(ols_coefs_pvals)
        model_comparison_dict = {}
        models = ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'K Neighbors', 'S V R', 'Elastic Net']
        return flask.jsonify(
                                newhead = head,
                                firsthead1 = head1,
                                models = models,
                                columns = html_cols,
                                model_comparison_dict = model_comparison_dict,
                                ols_summary = ols_summary,
                                ols_results = ols_coefs_pvals,
                                feat_imps_chart = feat_imps_chart
                                )

        # return html_cols

    elif request.method=='GET':
        return "OK this is a GET method"
    else:
        return("ok")

@app.route('/change_col_names', methods=['POST'])
def change_col_names():
    df = pd.read_csv('../data/df_reduced_cols.csv')

    print('Existing column names:', list(df.columns))
    form_results = request.form
    print('request.form: ', request.form)
    for k, v in form_results.items():
        print('k:', k)
        print('v:', v)
        print('global_y.y_col_name:', global_y.y_col_name)
        df.rename(columns={k: v}, inplace=True)
        if k == global_y.y_col_name:
            global_y.y_col_name = v
    df = drop_unnamed_col(df)
    df.to_csv('../data/df_Xy_new_names.csv')
    print('New column names:', list(df.columns))
    new_head = print_head(df)
    print('new_head:', new_head)
    return jsonify(newcolhead = new_head)

@app.route('/regressions', methods=["POST"])
def regressions():
    data_url = '../data/df_Xy_new_names.csv'
    selected_models = list(request.form.keys())
    print('selected_models: ', selected_models)
    model_comparison_dict = compare_models(data_url, global_y.y_col_name, selected_models)
    print('model_comparison_dict from regressions.py:', model_comparison_dict)
    html_string = build_regression_results_table(model_comparison_dict)
    return html_string

if __name__ == "__main__":
    Bootstrap(app)
    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)
