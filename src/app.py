from flask import Flask, make_response, request, render_template, jsonify
from flask_bootstrap import Bootstrap
from flask import Markup
import flask
import io
import csv
import pandas as pd
import numpy as np
import os
from flask import send_file

def print_head(df):
    """converts pandas 'head' to html; returns html"""
    head = df.head().to_html()
    return Markup(head)

app = Flask(__name__)

@app.route('/downloadcsv')
def downloadcsv():
    return send_file('../data/ready_to_import.csv',
                     mimetype='text/csv',
                     attachment_filename='ready_to_import.csv',
                     as_attachment=True)

@app.route('/')
def index():
    return flask.render_template('index.html')

@app.route('/ml')
def ml():
    return flask.render_template('upload.html')

def rename_col_types(df):
    cols_types = df.dtypes.to_dict()
    for k, v in cols_types.items():
        if v == 'int64':
            cols_types[k] = ['Numeric', '', '', '']
        elif v == 'float64':
            cols_types[k] = ['Numeric', '', '', '']
        else:
            cols_types[k] = ['Text', '', '', '']
    return cols_types

def append_cols_types(cols_types, coefs_dict, pvals_dict):
    for k, v in cols_types.items():
        for key, value in coefs_dict.items():
            if k == key:
                cols_types[k][2] = value
        for key, value in pvals_dict.items():
            if k == key:
                cols_types[k][3] = value
                cols_types[k] = v
    return cols_types

@app.route('/uploadcsv', methods=["POST"])
def uploadcsv():
    print('type (request)', type (request.files['data_file']))
    f = request.files['data_file']
    if not f:
        return "No file"
    df = pd.read_csv(f)
    print('df.head(): ', df.head())
    df.to_csv('../data/df.csv', index=False)
    head1 = print_head(df)
    columns = list(df.columns)
    cols_types = rename_col_types(df)
    coefs_dict = {}
    pvals_dict = {}
    append_cols_types(cols_types, coefs_dict, pvals_dict)
    print('cols_types:', cols_types)
    return flask.render_template(
                                'upload.html',
                                firsthead = head1,
                                cols_types = cols_types,
                                columns = columns
                                )

@app.route('/demo', methods=["GET"])
def demo():
    f = '../data/member_list.csv'
    if not f:
        return "No file"
    df = pd.read_csv(f)
    print('df.head(): ', df.head())
    df.to_csv('../data/df.csv', index=False)
    head1 = print_head(df)
    columns = list(df.columns)
    cols_types = rename_col_types(df)
    return flask.render_template(
                                'upload.html',
                                firsthead = head1,
                                cols_types = cols_types,
                                columns = columns
                                )

@app.route('/match_cols', methods=["POST", "GET"])
def match_cols():
    if request.method == 'POST':
        db_fields = ['db_first', 'db_last', 'db_state', 'db_email']
        cols_to_keep = []
        df = pd.read_csv('../data/df.csv')
        head1 = print_head(df)
        all_cols = list(df.columns)
        form_results = request.form
        field_match_dict = {}
        print('input from form (match cols): ', request.form)
        for k, v in form_results.items():
            print('This is k : v', k, ':', v)
            if v in db_fields:
                field_match_dict[k] = v
                cols_to_keep.append(k)
        print('cols_to_keep: ', cols_to_keep)
        cols_to_drop = set(all_cols) - set(cols_to_keep)
        print('cols_to_drop:', cols_to_drop)
        df_reduced_cols = df.drop(cols_to_drop, axis=1)
        print('Here is df_reduced_cols', df_reduced_cols)
        print('df_reduced_cols.columns', df_reduced_cols.columns)
        for old, new in field_match_dict.items():
            df_reduced_cols.rename(columns={old: new}, inplace=True)

        head_red = print_head(df_reduced_cols)
        df_reduced_cols.to_csv('../data/ready_to_import.csv')
        print('head_red: ', head_red)
        return flask.jsonify(
                                firsthead1 = head1,
                                newcolhead = head_red
                                )

    elif request.method=='GET':
        return "This is a GET method"
    else:
        return("ok")



if __name__ == "__main__":
    Bootstrap(app)
    app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False)
