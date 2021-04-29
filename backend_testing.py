# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 15:18:54 2020

@author: Anthony
"""

import pandas as pd
import pickle

trained_model = pickle.load(open('trainedModel.pkl', 'rb'))

ss_X= pickle.load(open('transformer.pkl', 'rb'))

test_data=pd.read_csv('test_model.csv')

test_data=test_data.iloc[:,1:]

Area=1254.00
RainFallInfo=2535.8
StateName="Andaman and Nicobar Islands"
districtName="NICOBARS"
season="Kharif"
crop="Arecanut"

StateName="StateName_"+StateName
districtName="DistrictName_"+districtName
season="Season_"+season
crop="Crop_"+crop


test_data[StateName]=1
test_data[districtName]=1
test_data[season]=1
test_data[crop]=1
test_data['Area']=Area
test_data['RainFallInfo']=RainFallInfo

ft=test_data.iloc[:,0:2].values
ft = ss_X.transform(ft.reshape(-1,2))
test_data.loc[:,0:2]=ft

score=trained_model.predict(test_data)

print(score)
