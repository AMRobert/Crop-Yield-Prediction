#Importing the Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Data Preprocessing

df=pd.read_csv("AgricultureInKarnataka.csv")

df.describe()

df['Crop'].unique()

options = ['Rice','Paddy']

df1 = df.loc[df['Crop'].isin(options)]

df1.describe()

df1['State_Name'].unique()

options = ['Andhra Pradesh','Karnataka','Kerala','Puducherry','Tamil Nadu','Telangana']

df2 = df1.loc[df1['State_Name'].isin(options)]

options = ['Kharif     ', 'Rabi       ', 'Summer     ',
       'Autumn     ', 'Winter     ']

df3 = df2.loc[df2['Season'].isin(options)]

df3.sort_values("Crop_Year", axis = 0, ascending=True,inplace=True)

One_hot_code_Season = pd.get_dummies(df3['Season'])

One_hot_code_Crop = pd.get_dummies(df3['Crop'])

One_hot_code_State_Name = pd.get_dummies(df3['State_Name'])

Crop = df3.drop(['State_Name','District_Name','Crop_Year','Season','Crop','Area'], axis=1)

df4 = pd.concat([df3, One_hot_code_State_Name], axis=1, sort=False)

df5 = pd.concat([df4, One_hot_code_Season], axis=1, sort=False)

df5 = df5.drop(['Crop_Year','State_Name','District_Name','Season','Crop','Production'], axis=1)

df6 = pd.concat([df5, One_hot_code_Crop], axis=1, sort=False)

df7 = pd.concat([df6, Crop], axis=1, sort=False)

df7.dropna(subset=['Production'],inplace=True)

df7 = df7.drop(['Paddy','Rice'],axis = 1)

df_head = df7.head()
print(df_head)
#Data Visualization

import seaborn as sns

sns.boxplot(x=df7['Area'])

sns.boxplot(x=df7['Production'])

fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(df7['Area'], df7['Production'])
ax.set_xlabel('Area')
ax.set_ylabel('Production')
plt.show()

sns.set_style('whitegrid')
plt.figure(figsize=(14,8))
sns.heatmap(df7.corr(), annot = True, cmap='coolwarm',linewidths=.1)
plt.show()

#Normalizing the data

df7['Area'] = np.log(df7['Area'])
df7['Production'] = np.log(df7['Production'])

#Feature Extraction

X = df7[['Area']]
y = df7['Production']

#Splitting the dataset into training set and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
print(X_train)

#Simple Linear Regression

from sklearn.linear_model import LinearRegression

Linear_regression = LinearRegression()
Linear_regression.fit(X_train,y_train)

#Evaluating Simple Linear Regression model

from sklearn.metrics import mean_squared_error

y_pred = Linear_regression.predict(X_test)
MSE = mean_squared_error(y_test, y_pred)
print(MSE)

#Plotting the graph

plt.scatter(X_test,y_test,color ='b')
plt.plot(X_test,y_pred,color = 'k')
plt.xlabel('Area')
plt.ylabel('Production')
plt.title('Test Set Yield Prediction')
plt.show()

#Multiple Linear Regression

X1 = df7.iloc[:, 0:11].values
y1 = df7.iloc[:, -1].values

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.30, random_state = 0)

Multiple_Linear_regression = LinearRegression()
Multiple_Linear_regression.fit(X1_train,y1_train)

#Saving and loading the model

import pickle

pickle.dump(Multiple_Linear_regression,open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

#Evaluating the model

y_pred1 = Multiple_Linear_regression.predict(X1_test)
MSE1 = mean_squared_error(y1_test, y_pred1)
print(MSE1)

print(Multiple_Linear_regression.intercept_)

print(Multiple_Linear_regression.coef_)
list(zip(df7.columns,Multiple_Linear_regression.coef_))

from sklearn import metrics

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y1_test, y_pred1)))

from sklearn.metrics import r2_score
score = r2_score(y1_test,y_pred1)
print(score)