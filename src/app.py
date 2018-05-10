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

def print_head(df):
    head = df.head().to_html()
    return Markup(head)

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

        return flask.render_template(
                                'cols4.html',
                                head = head,
                                columns = columns,
                                ols_summary = ols_summary
                                )
    elif request.method=='GET':
        return "OK this is a GET method"
    else:
        return("ok")

@app.route('/rename_cols', methods=["POST"])
def rename_cols():
    return flask.render_template(
                            'cols4.html',
                            head = head,
                            columns = columns
    )


@app.route('/ols', methods=["POST"])
def ols():
    pass

@app.route('/process', methods=['POST'])
def process():
    # if request.method=='POST':
    df = pd.read_csv('../data/df_reduced_cols.csv')
    # df.reset_index(drop=True)
    existing_cols = list(df.columns)
    print('existing columns:', existing_cols)

    form_results = request.form
    print('request.form: ', request.form)
    for k, v in form_results.items():
        print('k:', k)
        print('v:', v)
        df.rename(columns={k: v}, inplace=True)

    new_columns = list(df.columns)
    print('new columns:', new_columns)

    new_head = print_head(df)
    df.to_csv('../data/df_reduced_cols.csv')
    print('new_head', new_head)
    return jsonify(new_head = new_head)
        # return jsonify({'test' : new_head})

    # elif request.method=='GET':
    #     return "OK this is another GET method"
    # else:
    #     return("ok")

if __name__ == "__main__":
    Bootstrap(app)
    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)
