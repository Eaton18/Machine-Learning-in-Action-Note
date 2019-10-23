import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    """
    general function to parse tab -delimited floats
    :param fileName:
    :return:
    """
    dataMat = []  # assume last column is target value
    fr = open(fileName)
    for line in fr.readlines():  # for each line
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))  # map all elements to float()
        dataMat.append(fltLine)
    return dataMat


def distEclud(vecA, vecB):
    """
    distance func, using Euclidean distance.
    :param vecA:
    :param vecB:
    :return:
    """
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))  # la.norm(vecA-vecB)


def randCent(dataSet, k):
    """
    init K points randomly
    :param dataSet:
    :param k:
    :return:
    """
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k, n)))  # create centroid mat
    for j in range(n):  # create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = np.mat(minJ + rangeJ * np.random.rand(k, 1))
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    """

    :param dataSet:
    :param k:
    :param distMeas:
    :param createCent:
    :return:
    """
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))  # create mat to assign data points
    # to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # for each data point assign it to the closest centroid
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        print(centroids)
        for cent in range(k):  # recalculate centroids
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster
            centroids[cent, :] = np.mean(ptsInClust, axis=0)  # assign centroid to mean

    return centroids, clusterAssment


def biKmeans(dataSet, k, distMeas=distEclud):
    """

    :param dataSet:
    :param k:
    :param distMeas:
    :return:
    """
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))  # 存储每个点的簇以及分配结果
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]  # 初始簇
    centList = [centroid0]  # create a list with one centroid, 簇列表
    for j in range(m):  # calc initial Error for each point
        clusterAssment[j, 1] = distMeas(np.mat(centroid0), dataSet[j, :]) ** 2
    while (len(centList) < k):
        lowestSSE = np.inf  # init SSE, set as np.inf
        for i in range(len(centList)):  # for every centroid
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0],
                               :]  # get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)  # k=2, kMeans
            sseSplit = sum(splitClustAss[:, 1])  # compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:  # judge the error
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # new cluster and split cluster
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)  # change 1 to 3,4, or whatever
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]  # replace a centroid with two best centroids
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0],
        :] = bestClustAss  # reassign new clusters, and SSE
    return np.mat(centList), clusterAssment


def distSLC(vecA, vecB):  # Spherical Law of Cosines
    """
    calculate the distance in miles for two points on the earth’s surface.
    :param vecA:
    :param vecB:
    :return:
    """
    a = np.sin(vecA[0, 1] * np.pi / 180) * np.sin(vecB[0, 1] * np.pi / 180)
    b = np.cos(vecA[0, 1] * np.pi / 180) * np.cos(vecB[0, 1] * np.pi / 180) * np.cos(
        np.pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return np.arccos(a + b) * 6371.0  # pi is imported with numpy


def clusterClubs(numClust=5):
    """
    
    :param numClust: cluster number
    :return:
    """
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        # get the fourth and fifth fields, which contain the latitude and longitude
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = np.mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    # draw
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]  # 创建矩形
    # 创建不同标记图案
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')  # 导入地图
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[np.nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle,
                    s=90)
    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()


if __name__ == '__main__':
    datMat = np.mat(loadDataSet('testSet.txt'))
    print(">> Utilities function")
    print(f"min in dim 0: {min(datMat[:, 0])}")
    print(f"min in dim 1: {min(datMat[:, 1])}")
    print(f"max in dim 0: {max(datMat[:, 0])}")
    print(f"max in dim 1: {max(datMat[:, 1])}")
    print(f"random centroid:\n{randCent(datMat, 2)}")
    print(f"Euclidean distance: {distEclud(datMat[0], datMat[1])}")

    print(">> kMeans()")
    myCentroids, clustAssing = kMeans(datMat, 4)
    print(f"myCentroids:\n {myCentroids}")
    print(f"clustAssing:\n {clustAssing}")

    print(">> bi-kMeans()")
    datMat3 = np.mat(loadDataSet("testSet2.txt"))
    centList, myNewAssments = biKmeans(datMat3, 3)
    print(f"centList: \n{centList}")

    print(">> Portland example:")
    clusterClubs(5)
