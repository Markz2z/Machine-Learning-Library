from math import log
import operator


def calculateShannonEntropy(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        curLabel = featVec[-1]
        if curLabel not in labelCounts:
            labelCounts[curLabel] = 1
        else:
            labelCounts[curLabel] += 1
    shannoEntropy = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannoEntropy -= prob * log(prob, 2)
    return shannoEntropy


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            vec = featVec[:axis]
            vec.extend(featVec[axis+1:])
            retDataSet.append(vec)
    return retDataSet


def chooseBestFeature(dataSet):
    features = len(dataSet[0]) - 1
    baseEntropy = calculateShannonEntropy(dataSet)
    largestEntropyReduce = 0.0
    bestFeature = 0
    for i in range(features):
        featureList = [example[i] for example in dataSet]
        uniqueLabels = set(featureList)
        entropy = 0.0
        for val in uniqueLabels:
            subDataSet = splitDataSet(dataSet, i, val)
            prob = len(subDataSet) / float(len(dataSet))
            entropy += prob * calculateShannonEntropy(subDataSet)
        if (baseEntropy - entropy) > largestEntropyReduce:
            largestEntropyReduce = baseEntropy - entropy
            bestFeature = i
    return bestFeature


def majority(classList):
    classCnt = {}
    for vote in classList:
        if vote not in classCnt.keys():
            classCnt[vote] = 0
        classCnt[vote] += 1
    sortedClassCnt = sorted(classCnt.iteritems(), key=operator.itemgetter(1), reverse=True)
    print classCnt
    return sortedClassCnt[0][0]


def createTree(dataSet, labels_):
    labels = labels_[:]
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majority(classList)
    bestFeatIdx = chooseBestFeature(dataSet)
    bestFeatLabel = labels[bestFeatIdx]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeatIdx])
    featValues = [example[bestFeatIdx] for example in dataSet]
    uniqueVals = set(featValues)
    for val in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][val] = createTree(splitDataSet(dataSet, bestFeatIdx, val), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDic = inputTree[firstStr]
    featIdx = featLabels.index(firstStr)
    for key in secondDic.keys():
        if testVec[featIdx] == key:
            if type(secondDic[key]).__name__ == 'dict':
                classLabel = classify(secondDic[key], featLabels, testVec)
            else:
                classLabel = secondDic[key]
    return classLabel


dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
labelList = ['no surfacing', 'flippers']
tree = createTree(dataSet, labelList)
print classify(tree, labelList, [0, 1])

