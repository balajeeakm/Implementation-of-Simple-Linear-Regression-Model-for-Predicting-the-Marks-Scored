# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the standard Libraries
2. set variables for assigning dataset values.
3. import linear regression from sklearn
4. assign the points for representing in the graph
5. Predict the regression for the marks by using the representation of the graph.
6. hence we obtained the linear regression for the given dataset.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: BALAJEE  K.S
RegisterNumber:  212222080009
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
df=pd.read_csv('/content/exp-2.csv')
df.head(10)
plt.scatter(df['study hours(x)'],df['score(y)'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
lr.predict(x_test.iloc[0].values.reshape(1,1))
plt.scatter(df['study hours(x)'],df['score(y)'])
plt.xlabel('study hours(x)')
plt.ylabel('score(y)')
plt.plot(x_train,lr.predict(x_train),color='blue')
lr.coef_
lr.intercept_
*/
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
dataset:
![Screenshot (428)](https://github.com/balajeeakm/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131589871/add71f63-fcb1-442c-9dac-2cf8b1fa25ef)
graphical representation of dataset:
![Screenshot (429)](https://github.com/balajeeakm/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131589871/6da5a3e4-cda6-43f5-98c7-9e9ec07214d1)
![Screenshot (430)](https://github.com/balajeeakm/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131589871/c8c4e02b-ea63-4c0d-b021-e6acb2735f93)
The line of regression:
![Screenshot (431)](https://github.com/balajeeakm/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131589871/00a473e5-85fc-446e-90df-1fd88af849c5)
slope and intercept:
![Screenshot (432)](https://github.com/balajeeakm/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/131589871/112e0b6b-0dc9-4641-b885-e087f1e4fefd)







## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
