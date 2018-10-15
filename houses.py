#lighting the hearth
import pandas
import numpy
import matplotlib.pyplot
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet


#prep
HousingDataset = fetch_california_housing()
XTrain,XTest,YTrain,YTest= train_test_split(HousingDataset['data'],HousingDataset['target'],random_state=0)
CrossValidationScore = []

'''
#visualize data by each attribute
HousingDataframe = pandas.DataFrame(XTrain,columns=HousingDataset.feature_names)
grr = pandas.scatter_matrix(HousingDataframe,c=YTrain,figsize=(15,15),marker='o',hist_kwds={'bins':20},s=60,alpha=0.8)
'''

#LR
LR = LinearRegression()
LR.fit(XTrain,YTrain)

PredictTrain = LR.predict(XTrain)
PredictTest = LR.predict(XTest)

print("Fit a model X_train, and calculate mean-square-error with Y_train:", numpy.mean((YTrain - LR.predict(XTrain)) ** 2))

matplotlib.pyplot.figure()
matplotlib.pyplot.title("Linear Regression")
matplotlib.pyplot.scatter(PredictTrain, PredictTrain-YTrain,c='b',s=40,alpha=0.5)
matplotlib.pyplot.scatter(PredictTest, PredictTest-YTest,c='g',s=40,alpha=0.5)
matplotlib.pyplot.hlines(y=0,xmin=0,xmax=10)


#Ride
R = Ridge()
R.fit(XTrain,YTrain)

PredictTrain = R.predict(XTrain)
PredictTest = R.predict(XTest)

print("Fit a model X_train, and calculate mean-square-error with Y_train:", numpy.mean((YTrain - LR.predict(XTrain)) ** 2))

matplotlib.pyplot.figure()
matplotlib.pyplot.title("Ridge")
matplotlib.pyplot.scatter(PredictTrain, PredictTrain-YTrain,c='b',s=40,alpha=0.5)
matplotlib.pyplot.scatter(PredictTest, PredictTest-YTest,c='g',s=40,alpha=0.5)
matplotlib.pyplot.hlines(y=0,xmin=0,xmax=10)


#Lasso
L = Lasso()
L.fit(XTrain,YTrain)

PredictTrain = L.predict(XTrain)
PredictTest = L.predict(XTest)

print("Fit a model X_train, and calculate mean-square-error with Y_train:", numpy.mean((YTrain - LR.predict(XTrain)) ** 2))

matplotlib.pyplot.figure()
matplotlib.pyplot.title("Lasso")
matplotlib.pyplot.scatter(PredictTrain, PredictTrain-YTrain,c='b',s=40,alpha=0.5)
matplotlib.pyplot.scatter(PredictTest, PredictTest-YTest,c='g',s=40,alpha=0.5)
matplotlib.pyplot.hlines(y=0,xmin=0,xmax=10)


#ElasticNet
EN = ElasticNet()
EN.fit(XTrain,YTrain)

PredictTrain = EN.predict(XTrain)
PredictTest = EN.predict(XTest)

print("Fit a model X_train, and calculate mean-square-error with Y_train:", numpy.mean((YTrain - LR.predict(XTrain)) ** 2))
matplotlib.pyplot.figure()
matplotlib.pyplot.title("ElasticNet")
matplotlib.pyplot.scatter(PredictTrain, PredictTrain-YTrain,c='b',s=40,alpha=0.5)
matplotlib.pyplot.scatter(PredictTest, PredictTest-YTest,c='g',s=40,alpha=0.5)
matplotlib.pyplot.hlines(y=0,xmin=0,xmax=10)

#show
matplotlib.pyplot.show()
