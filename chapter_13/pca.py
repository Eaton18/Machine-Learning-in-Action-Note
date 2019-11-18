import numpy as np
from matplotlib import pyplot


def loadDataSet(fileName, delim='\t'):
    """

    :param fileName:
    :param delim:
    :return: (list), item is data row []
    """
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float, line)) for line in stringArr]
    return np.mat(datArr)


def pca(dataMat, topNfeat=9999999):
    """

    :param dataMat:
    :param topNfeat: (int), feature count
    :return: lowDDataMat, reconMat
    """
    meanVals = np.mean(dataMat, axis=0)  # cal mean matrix
    meanRemoved = dataMat - meanVals  # #remove mean
    covMat = np.cov(meanRemoved, rowvar=False)  # covariance matrix
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # cal eigenvalue and eigenvector
    eigValInd = np.argsort(eigVals)  # sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat + 1):-1]  # cut off unwanted dimensions
    redEigVects = eigVects[:, eigValInd]  # reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects  # transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat = np.shape(datMat)[1]
    for i in range(numFeat):
        meanVal = np.mean(datMat[np.nonzero(~np.isnan(datMat[:, i].A))[0], i])
        datMat[np.nonzero(np.isnan(datMat[:, i].A))[0], i] = meanVal
    return datMat


if __name__ == '__main__':
    data_mat = loadDataSet('testSet.txt')
    print(data_mat)
    print(np.shape(data_mat))
    lowDMat, reconMat = pca(data_mat, 1)
    print(np.shape(lowDMat))

    # # plot result
    # fig = pyplot.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(data_mat[:, 0].flatten().A[0], data_mat[:, 1].flatten().A[0], marker='^', s=20)
    # ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=10, c='red')
    # pyplot.show()

    data_mat = replaceNanWithMean()
    mean_vals = np.mean(data_mat, axis=0)
    mean_removed = data_mat - mean_vals
    cov_mat = np.cov(mean_removed, rowvar=False)
    eigVals, eigVects = np.linalg.eig(np.mat(cov_mat))
    print(eigVals)
