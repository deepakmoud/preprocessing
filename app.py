from flask.helpers import send_file

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from os import path
import re
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
# Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

#------------------------------ Saving dataset---------------------------------
# this is the path to save dataset for preprocessing
pathfordataset = "static/data-preprocess/"
pathfordatasetNew = "data-preprocess/new"   
app.config['DFPr'] = pathfordataset
app.config['DFPrNew'] = pathfordatasetNew
#------------------------------ Saving dataset for Linear regression-------------------------------------------
# this is the path to save dataset for single variable LR
pathforonevarLR = "static/Regression/onevarLR"
pathforonevarLRplot = "Regression/onevarLR/plot"
app.config['LR1VAR'] = pathforonevarLR
app.config['LR1VARplot'] = pathforonevarLRplot

#------------------------------ Saving image for K means-------------------------------------------
# this is the path to save figure of K menas
pathforelbowplot = "kmeans/plot"
#pathforonevarLRplot = "Regression/onevarLR/plot"
#app.config['LR1VAR'] = pathforonevarLR
app.config['elbowplot'] = pathforelbowplot
#print(app.config['elbowplot'])

# for index page
#------------------------------ Launcing undex page-------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

#------------------------------Data Preprocessing-------------------------------------------
# for data preprocessing

@app.route('/preprocessing')
def preprocessing():
    return render_template('preprocessing/preprocessing.html')


@app.route('/preprocessing/preprocessing' , methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        my_dataset = request.files['my_dataset']
        my_model_name = request.form['name_of_model']
        data_std = request.form['flexRadioDefault']
        dataset_path = os.path.join(pathfordataset, secure_filename(my_dataset.filename))
        my_dataset.save(dataset_path)
        get_dastaset = os.path.join(app.config['DFPr'],secure_filename(my_dataset.filename))
        df = pd.read_csv(get_dastaset)
        df = df.fillna(method='ffill')
        col_no = df.shape[1]

        cols = df.columns
        num_cols = df._get_numeric_data().columns
        cat_col=list(set(cols) - set(num_cols)) 


        temp = 0
        labelencoder = LabelEncoder()
        # taking care of cataagorical data
        for i in df.columns:
            for x in cat_col:
                if x==i:
                    df.iloc[:, temp] = labelencoder.fit_transform(df.iloc[:, temp])
            temp = temp + 1
        
        # taking care of missing data
        imputer = SimpleImputer(missing_values=np.NAN, strategy='mean', fill_value=None, verbose=1, copy=True)
        imputer = imputer.fit(df.iloc[:, 0:col_no])
        df.iloc[:, 0:col_no] = imputer.transform(df.iloc[:, 0:col_no])

        # standerization
        
        if data_std == "yes":
            sc_X = StandardScaler()
            df = sc_X.fit_transform(df)
        trained_dataset = pd.DataFrame(df)
        trained_dataset.to_csv("static/data-preprocess/new/trained_dataset.csv")

        return render_template('/preprocessing/preprocessing_output.html', model_name=my_model_name, data_shape=trained_dataset.shape, table=trained_dataset.head(5).to_html(classes='table table-striped table-dark table-hover x'), dataset_describe=trained_dataset.describe().to_html(classes='table table-striped table-dark table-hover x'), )
#------------------------------Download Dataset-------------------------------------------
@app.route('/downloadNewDataset')
def download_file():
    path1 = "static/data-preprocess/new/trained_dataset.csv"
    return send_file(path1,as_attachment=True)

#------------------------------Download Model-------------------------------------------
@app.route('/downloadmodel')
def download_model():
    path1 = "static/data-preprocess/model/model.pkl"
    return send_file(path1,as_attachment=True)

#------------------------------About us-------------------------------------------
@app.route('/aboutusnew')
def aboutusnew():
    return render_template('aboutusnew.html')
#------------------------------Supervised machine Learning-------------------------------------------


if __name__ == '__main__':
    app.run(debug=True)
    
# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response
app.config["CACHE_TYPE"] = "null"