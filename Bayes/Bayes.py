from numpy import ones, log


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec


def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWord2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word : %s is not in my vocabulary!" % word
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p2 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    print "p1:"
    print p1
    print "p2:"
    print p2
    if p1 > p2:
        return 1
    else:
        return 0


def BayesClassify(testSample):
    # 1.load train data and generate the vocabulary
    trainDataSet, trainDataCategory = loadDataSet()
    vocabularyList = createVocabList(trainDataSet)

    # 2.generate the string format vector to bit vector according to the vocabulary
    trainDataVecs = []
    for trainData in trainDataSet:
        trainDataVecs.append(setOfWord2Vec(vocabularyList, trainData))
    testBitVec = setOfWord2Vec(vocabularyList, testSample)
    print testBitVec

    # 3.train the naive bayes model
    p0Vec, p1Vec, pAbusive = trainNB0(trainDataVecs, trainDataCategory)
    print p0Vec
    print p1Vec

    # 4.select the label, which is the highest score of computing result list
    result = classifyNB(testBitVec, p0Vec, p1Vec, pAbusive)
    if result == 1:
        print "label is 1"
    else:
        print "label is 0"


testVec = ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please']
BayesClassify(testVec)