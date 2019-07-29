#import  libraries and functions
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.linear_model import LogisticRegression

bank_full = pd.read_csv('./bank/bank.csv')
# print(bank_full.head())
# print(bank_full.info())


from sklearn.preprocessing import LabelEncoder
number=LabelEncoder()
bank_full['y']=number.fit_transform(bank_full['y'].astype('str'))
print(bank_full)
bank_full=bank_full.apply(LabelEncoder().fit_transform)

# create a model
# Assign input features to object 'X'.
X = bank_full.drop('y',axis=1)
#Assign output variable to object 'y'.
y = bank_full['y']

#Instantiate and fit the model.
LogRegModel = LogisticRegression()
LogRegModel.fit(X, y)


client = [[0 ,  11   ,10   ,     1    ,      0     ,   0  ,   1475   ,     0  ,   0   ,     0  , 18  ,   10    ,    75  ,       0   ,           0     ,      0]]
y_pred = LogRegModel.predict(client)
if y_pred:
	print("Good client")
else: 
	print("Not Good Client")


