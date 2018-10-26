#lighting the hearth
import pandas
import numpy
import matplotlib.pyplot
import seaborn
from sklearn.datasets import fetch_california_housing
from sklearn.neighbors import KNeighborsClassifier
import sklearn.model_selection
import sklearn.linear_model
import sklearn.metrics
import sklearn.preprocessing


#0: 
#prep
HousingDataSet = fetch_california_housing()
HousingDataFrame = pandas.DataFrame(HousingDataSet.data,columns=HousingDataSet.feature_names)
HousingDataTarget = pandas.Series(HousingDataSet.target)

Scaler = sklearn.preprocessing.StandardScaler()
Scaler.fit(HousingDataFrame)
HousingDataFrameScaled = Scaler.transform(HousingDataFrame)

dataTrain, dataTest, targetTrain, targetTest = sklearn.model_selection.train_test_split(HousingDataFrame,HousingDataTarget,random_state=253)

seaborn.set()
matplotlib.pyplot.figure(0,figsize=(20, 50))
CrossValidationScore = []


#1: 
#univariate distribution
i = 0
for HousingNames in HousingDataFrame.columns:
    i += 1
    
    matplotlib.pyplot.subplot(HousingDataFrame.columns.size, 3, i)
    matplotlib.pyplot.plot(HousingDataFrame[HousingNames].value_counts().sort_index(), color = matplotlib.pyplot.cm.brg(i * 256 / 8))
    matplotlib.pyplot.xlabel(HousingNames)
    matplotlib.pyplot.ylabel("Frequency")
matplotlib.pyplot.figure(1)
matplotlib.pyplot.plot(HousingDataTarget.value_counts().sort_index())


#2: 
#feature dependency
matplotlib.pyplot.figure(2,figsize=(20,50))
i=0
for HousingNames in HousingDataFrame.columns:
    i += 1
    
    matplotlib.pyplot.subplot(HousingDataFrame.columns.size, 3, i)
    matplotlib.pyplot.plot(HousingDataFrame[HousingNames],HousingDataTarget,'.',color = matplotlib.pyplot.cm.brg(i * 256 / 8))
    matplotlib.pyplot.xlabel(HousingNames)
    matplotlib.pyplot.ylabel("Home Value")

matplotlib.pyplot.figure(3,figsize=(20,10))

matplotlib.pyplot.subplot(121)
matplotlib.pyplot.title("Data (unscaled)")
matplotlib.pyplot.xlabel("Latitude")
matplotlib.pyplot.ylabel("Home value")
matplotlib.pyplot.plot(HousingDataFrameScaled['Latitude'],HousingDataTarget,'.')


#
#3: 
#evaluating regression with SKLearn
#LR
modelLinearRegression = sklearn.linear_model.LinearRegression()

scoreLinearRegression = sklearn.model_selection.cross_val_score(modelLinearRegression, HousingDataFrame,HousingDataTarget)
scoreLinearRegressionScaled = sklearn.model_selection.cross_val_score(modelLinearRegression, HousingDataFrameScaled,HousingDataTarget)

print("Linear Regression accuracy: %0.2f (+- %0.2f)",(scoreLinearRegression.mean(),scoreLinearRegression.std()*2))
print("Linear Regression scaled accuracy: %0.2f (+- %0.2f)",(scoreLinearRegressionScaled.mean(),scoreLinearRegressionScaled.std()*2))


#Ride
modelRidge = sklearn.linear_model.Ridge()

scoreRidge = sklearn.model_selection.cross_val_score(modelRidge, HousingDataFrame,HousingDataTarget)
scoreRidgeScaled = sklearn.model_selection.cross_val_score(modelRidge, HousingDataFrameScaled,HousingDataTarget)

print("Ridge accuracy: %0.2f (+- %0.2f)",(scoreRidge.mean(),scoreRidge.std()*2))
print("Ridge scaled accuracy: %0.2f (+- %0.2f)",(scoreRidgeScaled.mean(),scoreRidgeScaled.std()*2))


#Lasso
modelLasso = sklearn.linear_model.Lasso()

scoreLasso = sklearn.model_selection.cross_val_score(modelLasso, HousingDataFrame,HousingDataTarget)
scoreLassoScaled = sklearn.model_selection.cross_val_score(modelLasso, HousingDataFrameScaled,HousingDataTarget)

print("Lasso accuracy: %0.2f (+- %0.2f)",(scoreLasso.mean(),scoreLasso.std()*2))
print("Lasso scaled accuracy: %0.2f (+- %0.2f)",(scoreLassoScaled.mean(),scoreLassoScaled.std()*2))


#ElasticNet
modelElasticNet = sklearn.linear_model.ElasticNet()

scoreElasticNet = sklearn.model_selection.cross_val_score(modelElasticNet, HousingDataFrame,HousingDataTarget)
scoreElasticNetScaled = sklearn.model_selection.cross_val_score(modelElasticNet, HousingDataFrameScaled,HousingDataTarget)

print("Elastic Net accuracy: %0.2f (+- %0.2f)",(scoreElasticNet.mean(),scoreElasticNet.std()*2))
print("Elastic Net scaled accuracy: %0.2f (+- %0.2f)",(scoreElasticNetScaled.mean(),scoreElasticNetScaled.std()*2))


#
#4:
#finding the best parameters with gridsearchcv



#show
matplotlib.pyplot.show()
