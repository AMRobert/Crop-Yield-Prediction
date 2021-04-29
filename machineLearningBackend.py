# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 18:09:45 2020

@author: Anthony
"""
import pandas as pa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import pickle
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

cropDataSet = pa.read_csv('crop_production.csv')

rainFallDataSet = pa.read_csv('rainfallDataSet.csv')

mergedDataSet = pa.merge(cropDataSet,rainFallDataSet,on=['StateName','Year'])

mergedDataSet['Production'] = mergedDataSet['Production']/mergedDataSet['Area']

columnNames=['StateName','DistrictName','Year','Season','Crop','Area','RainFallInfo','Production']

mergedDataSet = mergedDataSet.reindex(columns=columnNames)

mergedDataSet=mergedDataSet.dropna(axis=0)

mergedDataSetOneHotEncoder = pa.get_dummies(mergedDataSet, columns=['StateName',"DistrictName","Crop","Season"])
mergedDataSetFinal=mergedDataSetOneHotEncoder.loc[:, mergedDataSetOneHotEncoder.columns != 'Production']

mergedDataSetFinal=mergedDataSetFinal.drop(['Year'],axis=1)

Y=mergedDataSet['Production']

X_train, X_test, Y_train, Y_test = train_test_split(mergedDataSetFinal, Y, test_size = 0.3, random_state = 0)


decisionTreeRegressor = DecisionTreeRegressor()
trainedModel = decisionTreeRegressor.fit(X_train,Y_train)
Y_pred=trainedModel.predict(X_test)
r2=r2_score(Y_test,Y_pred)

with open('trainedModel.pkl','wb') as f:
        pickle.dump(trainedModel,f)
