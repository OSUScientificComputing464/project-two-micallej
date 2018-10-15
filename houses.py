#lighting the hearth
import pandas
import numpy
import matplotlib.pyplot
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

#prep
HousingDataset = fetch_california_housing()
XTrain,XTest,YTrain,YTest= train_test_split(HousingDataset['data'],HousingDataset['target'],random_state=0)

Neighbors = numpy.arange(1,15,2)

#visualize data by each attribute
HousingDataframe = pandas.DataFrame(XTrain,columns=HousingDataset.feature_names)
grr = pandas.scatter_matrix(HousingDataframe,c=YTrain,figsize=(15,15),marker='o',hist_kwds={'bins':20},s=60,alpha=0.8)

'''
#grid search with cross-validation
CrossValidationScore = []

for i in Neighbors:
	knn = KNeighborsClassifier(n_neighbors=i)
	Scores = CrossValidationScore(knn,XTrain, YTrain)
	CrossValidationScore.append(numpy.mean(Scores))
	
print("best cross-validation score: {:.3f}".format(numpy.max(CrossValidationScore)))
BestNNeighbors = Neighbors[numpy.argmax(CrossValidationScore)]
print("best n_neighbors: {:.3f}".format(knn.score(XTest,YTest)))

knn = KNeighborsClassifier(n_neighbors=BestNNeighbors)
knn.fit(XTrain,YTrain)
print("test-set score: {:.3f}".format(knn.score(XTest,YTest)))
'''
#show
matplotlib.pyplot.show()
