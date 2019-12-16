import numpy as np


def loadExData():
    return [[0, 0, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 1, 0, 0]]


def ecludSim(inA, inB):
    return 1.0 / (1.0 + np.linalg.norm(inA - inB))


def pearsSim(inA, inB):
    if len(inA) < 3: return 1.0
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=0)[0][1]


def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = np.linalg.norm(inA) * np.linalg.norm(inB)
    return 0.5 + 0.5 * (num / denom)


def standEst(dataMat, user, simMeas, item):
    """

    :param dataMat: data matrix
    :param user: user id
    :param simMeas: function to calculate similarity
    :param item: production id
    :return:
    """
    n = np.shape(dataMat)[1]  # production count
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        overLap = np.nonzero(np.logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = np.nonzero(dataMat[user, :].A == 0)[1]
    if len(unratedItems) == 0:
        return ('you rated everything')
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]


def svdEst(dataMat, user, simMeas, item):
    n = np.shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = np.linalg.svd(dataMat)
    Sig4 = np.mat(np.eye(4) * Sigma[:4])
    xformedItems = dataMat.T * U[:, :4] * Sig4.I

    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item: continue
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


if __name__ == '__main__':
    Data = loadExData()
    U, sigma, VT = np.linalg.svd(Data)
    sigma3 = np.mat([[sigma[0], 0, 0], [0, sigma[1], 0], [0, 0, sigma[1]]])
    ary = U[:, :3] * sigma3 * VT[:3, :]
    # print(ary)
    print("========================================")
    myMat = np.mat(loadExData())
    print(ecludSim(myMat[:, 0], myMat[:, 4]))
    print(ecludSim(myMat[:, 0], myMat[:, 0]))
    print(cosSim(myMat[:, 0], myMat[:, 4]))
    print(cosSim(myMat[:, 0], myMat[:, 0]))
    print(pearsSim(myMat[:, 0], myMat[:, 4]))
    print(pearsSim(myMat[:, 0], myMat[:, 0]))
    print("========================================")
    myMat = np.mat(loadExData())
    myMat[0, 1] = myMat[0, 0] = myMat[1, 0] = myMat[2, 0] = 4
    myMat[3, 3] = 2
    print(myMat)
    cos_res = recommend(myMat, 2, simMeas=cosSim)
    eclud_res = recommend(myMat, 2, simMeas=ecludSim)
    pers_res = recommend(myMat, 2, simMeas=pearsSim)
    print(f"cos:{cos_res}")
    print(f"eclud:{eclud_res}")
    print(f"pers:{pers_res}")
    print("========================================")
    myMat = np.mat(loadExData())
    myMat[0, 1] = myMat[0, 0] = myMat[1, 0] = myMat[2, 0] = 4
    myMat[3, 3] = 2
    print(myMat)
    cos_res = recommend(myMat, 1, simMeas=cosSim, estMethod=svdEst)
    print(f"cos:{cos_res}")
