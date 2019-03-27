import os, sys, shutil, time

from flask import Flask, request, jsonify, render_template,send_from_directory
import pandas as pd
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import urllib.request
import json



app = Flask(__name__)



@app.route('/')
def root():
    return render_template('index.html')



@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')



@app.route('/result.html', methods = ['POST'])
def predict():
    ad=pd.read_csv("agri_ds.csv")
    y = ad['crop'].values
    X = ad.drop('crop', axis=1).values 
    knn = KNeighborsClassifier(n_neighbors=6)
    knn.fit(X,y)
    

    if request.method == 'POST':
        se = request.form['season']
        ph = request.form['pH']
        te = request.form['temperature']
        hu = request.form['humidity']
        ra = request.form['rainfall']
        yi = request.form['yield']
        wa = request.form['water']
        data=np.array([se,ph,te,hu,ra,yi,wa])
        
        db=data.reshape(1,-1)
        #print(db)
        my_prediction = knn.predict(db)



    return render_template('result.html', prediction = my_prediction)


if __name__ == '__main__':
    app.run(debug = True)
