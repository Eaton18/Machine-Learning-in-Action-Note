import numpy as np
import operator
import matplotlib.pyplot as plt
import os


def createDataSet():
    """
    create data sets
    :return:
    """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """

    :param inX: the input vector to classify
    :param dataSet: full matrix of training examples
    :param labels: a vector of labels
    :param k: the number of nearest neighbors to use in the voting
    :return:
    """
    dataSetSize = dataSet.shape[0]  # get training examples counts
    # calculate Euclidian distance
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDisttances = sqDiffMat.sum(axis=1)
    distances = sqDisttances ** 0.5
    sortedDistIndicies = distances.argsort()

    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()  # get the number of lines in the file
    numberOfLines = len(arrayOLines)  # prepare matrix to return
    returnMat = np.zeros((numberOfLines, 3))  # prepare labels return
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))

        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    """
      norm_value = (oldValue-min) / (max-min)
    :param dataSet:
    :return:
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    # element wise divide, not Matrix division
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def datingClassTest():
    """
    test classification results
    :return:
    """
    hoRatio = 0.08  # hold out 10%
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    # print(errorCount)


def classifyPerson():
    """
    Dating site predictor function
    :return:
    """
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')

    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person:", resultList[classifierResult - 1])


def img2vector(filename):
    """
    converting images into test vectors
    :param filename:
    :return:
    """
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    """

    :return:
    """
    hwLabels = []
    trainingFileList = os.listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)

    testFileList = os.listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d,the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total rate is:%f" % (errorCount / float(mTest)))


if __name__ == '__main__':
    print(">> Test createDataSet()")
    group, labels = createDataSet()
    print(f"group:\n{group}")
    print(f"labels: {labels}")

    print(">> Test classify0()")
    res_class = classify0([0, 0], group, labels=labels, k=3)
    print(f"class is: {res_class}")

    # Example: improving matches from a dating site with kNN
    datingDataMat, datingLabels = file2matrix(filename="datingTestSet2.txt")
    print(f"dataingDataMat: \n{datingDataMat}")
    print(f"datingLabels: \n{datingLabels}")
    # plot res
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * np.array(datingLabels), 15.0 * np.array(datingLabels))
    plt.show()

    normMat, ranges, minVals = autoNorm(datingDataMat)
    print(f"normMat: \n{normMat}")
    print(f"ranges: {ranges}")
    print(f"minVals: {minVals}")

    # evaluation
    datingClassTest()
    # predictor system
    classifyPerson()

    # Example: a handwriting recognition system
    print(">> Handwriting recognition")
    handwritingClassTest()
