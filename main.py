# -*- coding: utf-8 -*-
"""
@author: Anthony
"""

from flask import Flask,request,render_template
import pickle
import pandas as pd

app = Flask(__name__)

@app.route("/",)
def index():
    return render_template("index.html")

@app.route("/api/predict",methods=['POST'])
def predict():
    
    x=request.get_json()
    StateName=x['stateName']
    Area=float(x['cropArea'])
    RainFallInfo=float(x['totalRainFall'])
    districtName=x['districtName']
    season=x['cropSeason']
    crop=x['cropName']
    
    trained_model = pickle.load(open('static/pythonFiles/trainedModel.pkl', 'rb'))
    test_data=pd.read_csv('static/pythonFiles/test_model.csv')
    test_data=test_data.iloc[:,1:]
        
    StateName="StateName_"+StateName
    districtName="DistrictName_"+districtName
    season="Season_"+season
    crop="Crop_"+crop
    
    test_data['Area']=Area
    test_data['RainFallInfo']=RainFallInfo
    if StateName in test_data:
        test_data[StateName]=1
    if districtName in test_data:
        test_data[districtName]=1
    if season in test_data:
        test_data[season]=1
    if crop in test_data:
        test_data[crop]=1
    a=trained_model.predict(test_data)
    b=','.join(str(x) for x in a)
    return b

if __name__ ==  '__main__':
    app.run(host="localhost",port=9999,debug=True,use_reloader=False)
