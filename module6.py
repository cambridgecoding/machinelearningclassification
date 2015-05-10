import csv
import numpy
import scipy
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn import neighbors
import knnplots
from sklearn.naive_bayes import GaussianNB

from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV


#Code common to all modeles from module 3 onwards
##NB. The X and yTransformed variables come from the preprocessing in the previous module.
fileName = "wdbc.csv"
fileOpen = open(fileName, "rU")
csvData = csv.reader(fileOpen)
dataList = list(csvData)
dataArray =  numpy.array(dataList)
X = dataArray[:,2:32].astype(float)
y = dataArray[:, 1]
le = preprocessing.LabelEncoder()
le.fit(y)
yTransformed = le.transform(y)
XTrain, XTest, yTrain, yTest = train_test_split(X, yTransformed)

knnK3 = neighbors.KNeighborsClassifier(n_neighbors = 3)
knnK15 = neighbors.KNeighborsClassifier(n_neighbors = 15)
nbmodel = GaussianNB()

