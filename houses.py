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
HousingDataSetScaled = Scaler.transform(HousingDataFrame)
HousingDataFrameScaled = pandas.DataFrame(HousingDataSetScaled, columns = HousingDataSet.feature_names)

dataTrain, dataTest, targetTrain, targetTest = sklearn.model_selection.train_test_split(HousingDataFrame,HousingDataTarget,random_state=253)

seaborn.set()
matplotlib.pyplot.figure(0,figsize=(20, 50))
CrossValidationScore = []


#
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


#
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
matplotlib.pyplot.title("Data (scaled)")
matplotlib.pyplot.xlabel("Latitude")
matplotlib.pyplot.ylabel("Home value")
matplotlib.pyplot.plot(HousingDataFrameScaled['Latitude'],HousingDataTarget,'.')


#
#evaluating regression with SKLearn
#LR
modelLinearRegression = sklearn.linear_model.LinearRegression()

scoreLinearRegression = sklearn.model_selection.cross_val_score(modelLinearRegression, HousingDataFrame,HousingDataTarget)
scoreLinearRegressionScaled = sklearn.model_selection.cross_val_score(modelLinearRegression, HousingDataSetScaled,HousingDataTarget)

print("Linear Regression accuracy: ",scoreLinearRegression.mean(),"(+/-",scoreLinearRegression.std()*2,")")
print("Linear Regression scaled accuracy: ",scoreLinearRegressionScaled.mean(),"(+/-",scoreLinearRegressionScaled.std()*2,")")

crossValidationLinearRegression = sklearn.model_selection.GridSearchCV(sklearn.linear_model.LinearRegression(), param_grid = {
    'fit_intercept': [False, True],
    'alpha': numpy.linspace(0, 10, 100)
})
crossValidationLinearRegression.fit(HousingDataSetScaled, HousingDataTarget)

print("BEST FIT: ",str(crossValidationRidge.best_estimator_))
print("BEST SCORE: ",crossValidationRidge.best_score_)

gridSearchLinearRegression = pandas.DataFrame(crossValidationLinearRegression.cv_results)
gridSearchLinearRegression

matplotlib.pyplot.figure(4, figsize = (20, 10))

matplotlib.pyplot.subplot(121)
matplotlib.pyplot.title("Linear Regression Alpha Parameter Optimization\nwith Intercept Fitting")
matplotlib.pyplot.xlabel("Alpha Value")
matplotlib.pyplot.ylabel("Mean CV Test Score")
matplotlib.pyplot.plot(gridSearchLinearRegression.loc[gs_LinearRegression.param_fit_intercept == True, ['param_alpha']],
         gridSearchLinearRegression.loc[gridSearchLinearRegression.param_fit_intercept == True, ['mean_test_score']],
         '.',
         label = "fit_intercept == True")

matplotlib.pyplot.subplot(122)
matplotlib.pyplot.title("Linear Regression Alpha Parameter Optimization\nwithout Intercept Fitting")
matplotlib.pyplot.xlabel("Alpha Value")
matplotlib.pyplot.ylabel("Mean CV Test Score")
matplotlib.pyplot.plot(gridSearchLinearRegression.loc[gridSearchLinearRegression.param_fit_intercept == False, ['param_alpha']],
         gridSearchLinearRegression.loc[gridSearchLinearRegression.param_fit_intercept == False, ['mean_test_score']],
         '.',
         label = "fit_intercept == False",
         color = matplotlib.pyplot.cm.brg(128))


#Ridge
modelRidge = sklearn.linear_model.Ridge()

scoreRidge = sklearn.model_selection.cross_val_score(modelRidge, HousingDataFrame,HousingDataTarget)
scoreRidgeScaled = sklearn.model_selection.cross_val_score(modelRidge, HousingDataSetScaled,HousingDataTarget)

print("Ridge accuracy: ",scoreRidge.mean(),"(+/- ",scoreRidge.std()*2,")")
print("Ridge scaled accuracy: ",scoreRidgeScaled.mean(),"(+/- ",scoreRidgeScaled.std()*2,")")

crossValidationRidge = sklearn.model_selection.GridSearchCV(sklearn.linear_model.Ridge(), param_grid = {
    'fit_intercept': [False, True],
    'alpha': numpy.linspace(0, 10, 100)
})
crossValidationRidge.fit(HousingDataSetScaled, HousingDataTarget)

print("BEST FIT: ",str(crossValidationRidge.best_estimator_))
print("BEST SCORE: ",crossValidationRidge.best_score_)

gridSearchRidge = pandas.DataFrame(crossValidationRidge.cv_results)
gridSearchRidge

matplotlib.pyplot.figure(4, figsize = (20, 10))

matplotlib.pyplot.subplot(121)
matplotlib.pyplot.title("Ridge Alpha Parameter Optimization\nwith Intercept Fitting")
matplotlib.pyplot.xlabel("Alpha Value")
matplotlib.pyplot.ylabel("Mean CV Test Score")
matplotlib.pyplot.plot(gridSearchRidge.loc[gs_ridge.param_fit_intercept == True, ['param_alpha']],
         gridSearchRidge.loc[gridSearchRidge.param_fit_intercept == True, ['mean_test_score']],
         '.',
         label = "fit_intercept == True")

matplotlib.pyplot.subplot(122)
matplotlib.pyplot.title("Ridge Alpha Parameter Optimization\nwithout Intercept Fitting")
matplotlib.pyplot.xlabel("Alpha Value")
matplotlib.pyplot.ylabel("Mean CV Test Score")
matplotlib.pyplot.plot(gridSearchRidge.loc[gridSearchRidge.param_fit_intercept == False, ['param_alpha']],
         gridSearchRidge.loc[gridSearchRidge.param_fit_intercept == False, ['mean_test_score']],
         '.',
         label = "fit_intercept == False",
         color = matplotlib.pyplot.cm.brg(128))


#Lasso
modelLasso = sklearn.linear_model.Lasso()

scoreLasso = sklearn.model_selection.cross_val_score(modelLasso, HousingDataFrame,HousingDataTarget)
scoreLassoScaled = sklearn.model_selection.cross_val_score(modelLasso, HousingDataSetScaled,HousingDataTarget)

print("Lasso accuracy: ",scoreLasso.mean(),"(+/- ",scoreLasso.std()*2,")")
print("Lasso scaled accuracy: ",scoreLassoScaled.mean(),"(+/- ",scoreLassoScaled.std()*2,")")

crossValidationLasso = sklearn.model_selection.GridSearchCV(sklearn.linear_model.Lasso(), param_grid = {
    'fit_intercept': [False, True],
    'alpha': numpy.linspace(0, 10, 100)
})
crossValidationLasso.fit(HousingDataSetScaled, HousingDataTarget)

print("BEST FIT: ",str(crossValidationLasso.best_estimator_))
print("BEST SCORE: ",crossValidationLasso.best_score_)

gridSearchLasso = pandas.DataFrame(crossValidationLasso.cv_results)
gridSearchLasso

matplotlib.pyplot.figure(4, figsize = (20, 10))

matplotlib.pyplot.subplot(121)
matplotlib.pyplot.title("Lasso Alpha Parameter Optimization\nwith Intercept Fitting")
matplotlib.pyplot.xlabel("Alpha Value")
matplotlib.pyplot.ylabel("Mean CV Test Score")
matplotlib.pyplot.plot(gridSearchLasso.loc[gs_Lasso.param_fit_intercept == True, ['param_alpha']],
         gridSearchLasso.loc[gridSearchLasso.param_fit_intercept == True, ['mean_test_score']],
         '.',
         label = "fit_intercept == True")

matplotlib.pyplot.subplot(122)
matplotlib.pyplot.title("Lasso Alpha Parameter Optimization\nwithout Intercept Fitting")
matplotlib.pyplot.xlabel("Alpha Value")
matplotlib.pyplot.ylabel("Mean CV Test Score")
matplotlib.pyplot.plot(gridSearchLasso.loc[gridSearchLasso.param_fit_intercept == False, ['param_alpha']],
         gridSearchLasso.loc[gridSearchLasso.param_fit_intercept == False, ['mean_test_score']],
         '.',
         label = "fit_intercept == False",
         color = matplotlib.pyplot.cm.brg(128))


#ElasticNet
modelElasticNet = sklearn.linear_model.ElasticNet()

scoreElasticNet = sklearn.model_selection.cross_val_score(modelElasticNet, HousingDataFrame,HousingDataTarget)
scoreElasticNetScaled = sklearn.model_selection.cross_val_score(modelElasticNet, HousingDataSetScaled,HousingDataTarget)

print("Elastic Net accuracy: ",scoreElasticNet.mean(),"(+/- ",scoreElasticNet.std()*2,")")
print("Elastic Net scaled accuracy: ",scoreElasticNetScaled.mean(),"(+/-",scoreElasticNetScaled.std()*2,")")

crossValidationElasticNet = sklearn.model_selection.GridSearchCV(sklearn.linear_model.ElasticNet(), param_grid = {
    'fit_intercept': [False, True],
    'alpha': numpy.linspace(0, 10, 100)
})
crossValidationElasticNet.fit(HousingDataSetScaled, HousingDataTarget)

print("BEST FIT: ",str(crossValidationElasticNet.best_estimator_))
print("BEST SCORE: ",crossValidationElasticNet.best_score_)

gridSearchElasticNet = pandas.DataFrame(crossValidationElasticNet.cv_results)
gridSearchElasticNet

matplotlib.pyplot.figure(4, figsize = (20, 10))

matplotlib.pyplot.subplot(121)
matplotlib.pyplot.title("Elastic Net Alpha Parameter Optimization\nwith Intercept Fitting")
matplotlib.pyplot.xlabel("Alpha Value")
matplotlib.pyplot.ylabel("Mean CV Test Score")
matplotlib.pyplot.plot(gridSearchElasticNet.loc[gs_ElasticNet.param_fit_intercept == True, ['param_alpha']],
         gridSearchElasticNet.loc[gridSearchElasticNet.param_fit_intercept == True, ['mean_test_score']],
         '.',
         label = "fit_intercept == True")

matplotlib.pyplot.subplot(122)
matplotlib.pyplot.title("Elastic Net Alpha Parameter Optimization\nwithout Intercept Fitting")
matplotlib.pyplot.xlabel("Alpha Value")
matplotlib.pyplot.ylabel("Mean CV Test Score")
matplotlib.pyplot.plot(gridSearchElasticNet.loc[gridSearchElasticNet.param_fit_intercept == False, ['param_alpha']],
         gridSearchElasticNet.loc[gridSearchElasticNet.param_fit_intercept == False, ['mean_test_score']],
         '.',
         label = "fit_intercept == False",
         color = matplotlib.pyplot.cm.brg(128))

#show
matplotlib.pyplot.show()
