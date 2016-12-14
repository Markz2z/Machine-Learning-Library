from numpy import *
import operator
import matplotlib
import os

# inX:input vector for classifying
# dataSet:input training set
# labels:label vector
# k:account of the most nearest neighbors
def trainingKNN(inX, dataSet, labels, k):
    #compute the distance
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5

    #select the nearest k points
    sortedDisIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDisIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    #sort all points
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]