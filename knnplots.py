import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors
import numpy as np

def plotvector(XTrain, yTrain, XTest, yTest, weights, upperLim = 310):
    results = []
    for n in range(1, upperLim, 4):
        clf = neighbors.KNeighborsClassifier(n_neighbors = n, weights = weights)
        clf = clf.fit(XTrain, yTrain)
        preds = clf.predict(XTest)
        accuracy = clf.score(XTest, yTest)
        results.append([n, accuracy])
    results = np.array(results)
    return(results)

def plotaccuracy(XTrain, yTrain, XTest, yTest, upperLim):
    pltvector1 = plotvector(XTrain, yTrain, XTest, yTest, weights = "uniform")
    pltvector2 = plotvector(XTrain, yTrain, XTest, yTest, weights = "distance")
    line1 = plt.plot(pltvector1[:,0], pltvector1[:,1], label = "uniform")
    line2 = plt.plot(pltvector2[:,0], pltvector2[:,1],  label = "distance")
    plt.legend(loc=3)
    plt.ylim(0.5, 1)
    plt.title("Accuracy with Increasing K")
    plt.show()



def decisionplot(XTrain, yTrain, n_neighbors, weights):
    h = .02  # step size in the mesh
    Xtrain = XTrain[:, :2] # we only take the first two features.
    # Create color maps
    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
    cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(Xtrain, yTrain)
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = Xtrain[:, 0].min() - 1, Xtrain[:, 0].max() + 1
    y_min, y_max = Xtrain[:, 1].min() - 1, Xtrain[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),  np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap = cmap_light)
    # Plot also the training points
    plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c = yTrain, cmap = cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("2-Class classification (k = %i, weights = '%s')" % (n_neighbors, weights))
    plt.show()