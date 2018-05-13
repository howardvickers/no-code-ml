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
from regressions import run_models as rm

def build_regression_results_table(dict):
    top = '<table class="table table-hover"> <thead> <tr> <th scope="col">Model</th> <th scope="col">RMSE (Train)</th> <th scope="col">RMSE (Test)</th> <th scope="col">R-Squared</th> </tr> </thead> <tbody>'
    middle = '<tr> <td>{}</td> <td>{}</td> <td>{}</td> <td>{}</td> </tr>'
    bottom =   '</tbody></table>'
    mid_rows = ""
    for k, v in dict.items():
        row = middle.format(k, v[0], v[1], v[2])
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

def compare_models(data_url, y, models):
    """passes list of models to compare to GridSearchCV in rm(); returns model_comparison_dict"""
    # X, y = get_X_y(data_url, y)
    X = 0
    y = 0
    models = []
    model_comparison_dict = rm(X, y, models)
    return model_comparison_dict

def get_X_y(data_url, y):
    """reloads data as csv and separates into two dataframes (X, y); returns X, y"""
    df = pd.read_csv(data_url)
    X = df.drop(y, axis=1)
    y = df[y]
    return X, y

app = Flask(__name__)

@app.route('/upload.html')
def upload():
    return flask.render_template('upload.html')

@app.route('/head', methods=["POST"])
def head():
    f = request.files['data_file']
    if not f:
        return "No file"
    df = pd.read_csv(f)
    df.to_csv('../data/df.csv')
    head = print_head(df)
    columns = list(df.columns)

    return flask.render_template(
                                'head.html',
                                head = head,
                                columns = columns
                                )

@app.route('/select_cols', methods=["POST", "GET"])
def select_cols():
    if request.method == 'POST':
        y_col = ''
        df = pd.read_csv('../data/df.csv')
        all_cols = list(df.columns)
        form_results = request.form
        print('input from form (select cols): ', request.form)
        cols_to_keep = list(form_results.keys())
        x_cols = cols_to_keep.copy()
        print('This is cols_to_keep:', cols_to_keep)
        print('This is x_cols: ', x_cols)
        for k, v in form_results.items():
            print('This is k', k)
            print('This is v', v)
            if v == 'y':
                x_cols.remove(k)
                y_col = k
                print('This is y_col', y_col)
        print('Now here is cols_to_keep', cols_to_keep)
        print('Now here is x_cols', x_cols)
        print('This is again y_col', y_col)
        cols_to_drop = set(all_cols) - set(cols_to_keep)
        new_df = df.drop(cols_to_drop, axis=1)
        print('Here is new_df', new_df)
        print('new_df.columns', new_df.columns)
        df_X = new_df.drop(y_col, axis=1)
        head = print_head(new_df)
        new_df.to_csv('../data/df_reduced_cols.csv')
        columns = list(new_df.columns)
        X = df_X
        y = df[y_col]
        ols_results = ols_tm(X.astype(float), y)
        ols_summary = Markup(ols_results)
        model_comparison_dict = {}
        models = ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'K Neighbors', 'S V R', 'Elastic Net']
        return flask.render_template(
                                'cols4.html',
                                head = head,
                                models = models,
                                columns = columns,
                                model_comparison_dict = model_comparison_dict,
                                ols_summary = ols_summary
                                )
    elif request.method=='GET':
        return "OK this is a GET method"
    else:
        return("ok")

@app.route('/regressions', methods=["POST"])
def regressions():
    # X_url = '../data/df_X_new_names.csv'
    # y_url = '../data/df_y_new_names.csv'
    # X_url = 0
    # y_url = 0
    selected_models = list(request.form.keys())
    print('selected_models: ', selected_models)
    # model_comparison_dict = compare_models(X_url, y_url, selected_models)
    model_comparison_dict = {'SVR': [1, 2, 3], 'Random Forest': [11, 22, 33]}
    html_string = build_regression_results_table(model_comparison_dict)
    return html_string

@app.route('/ols', methods=["POST"])
def ols():
    pass

@app.route('/process', methods=['POST'])
def process():
    # if request.method=='POST':
    df = pd.read_csv('../data/df_reduced_cols.csv')

    existing_cols = list(df.columns)
    print('existing columns:', existing_cols)

    form_results = request.form
    print('request.form: ', request.form)
    for k, v in form_results.items():
        print('k:', k)
        print('v:', v)
        df.rename(columns={k: v}, inplace=True)
    df = drop_unnamed_col(df)
    new_columns = list(df.columns)
    print('new columns:', new_columns)

    new_head = print_head(df)
    df.to_csv('../data/df_new_col_names.csv')
    print('new_head', new_head)
    return jsonify(result = new_head)

if __name__ == "__main__":
    Bootstrap(app)
    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)
