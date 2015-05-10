import csv
import numpy
import scipy
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn import metrics

#Import for Module 5
from sklearn.naive_bayes import GaussianNB

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


