import numpy as np
from sklearn.preprocessing import OneHotEncoder
from astropy import stats
#本文件内都是通用的数据处理&评估函数

#将x稳健标准化，采用的方法是使用双权标度，在方差较大的情况下合理估计标准化
#具体数学公式我也看不懂
def normalize_biweight(x, eps=1e-10):
    median = np.median(x)#中位数
    scale = stats.biweight.biweight_scale(x)
    if np.std(x) < 1e+2 or np.isnan(scale) or scale < 1e-4:
        norm =  (x-np.mean(x))/np.std(x)
    else:
        norm = (x - median) / (scale + eps)
    return norm

#对每一列进行标准化
def normalize(x):
    norm = lambda x: (x-np.mean(x))/np.std(x)
    return np.apply_along_axis(norm, 0, x)

#数据处理，将df中的数据进行处理，离散值转换成独热码，连续值不变
def data_preprocess(df):

    def _encoding(i):
        if df.iloc[:,i].dtype == 'O' or df.iloc[:, i].dtype.name == 'category':
            tempX = df.iloc[:, i].values.reshape(-1, 1)
            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(tempX)
            out = enc.transform(tempX).toarray()
        else:
            out = df.iloc[:, i].values.reshape(-1, 1)
        return out
    p = df.shape[1]
    X_encode = [_encoding(i) for i in np.arange(p)]
    return X_encode

#对图G中的预测结果进行评估，边的两个方向一起被计算，trueG中某一条边不会出现两个方向都有
#TP 真阳 真实和预测都为1（两个方向）
#TN 真阴 真实和预测都为0（两个方向）
#FP 假阳 真实0 预测1（一个方向，另外一个方向TN）
#FN 假阴 真实1 预测0（一个方向，另外一个方向TN）
#FD 两个方向一个FP 另外一个FN
#MD 一个FP 一个TP
#FPMD 两个FP
#precise 准确率 TP/(TP + FP + FD)
#recall 召回率，TP/所有边  recall1 召回率，(TP + FN + FD)
#SHD 结构汉明距离，所有边预测和真实不同的个数(不论哪一个方向)/2
def evaluate_binary(trueG, estG):
    TP, TN, FP, FN, FD, MD, FPMD = 0, 0, 0, 0, 0, 0, 0
    n_node = trueG.shape[0]
    for i in range(1, n_node):
        for j in range(i):
            if trueG[i, j] == 1 and trueG[j, i] == 0 and estG[i, j] == 1 and \
                    estG[j, i] == 0:
                TP += 1
            if trueG[i, j] == 0 and trueG[j, i] == 1 and estG[i, j] == 0 and \
                    estG[j, i] == 1:
                TP += 1
            if trueG[i, j] == 0 and trueG[j, i] == 0 and estG[i, j] == 0 and \
                    estG[j, i] == 0:
                TN += 1
            if trueG[i, j] == 0 and trueG[j, i] == 0 and estG[i, j] == 1 and \
                    estG[j, i] == 0:
                FP += 1
            if trueG[i, j] == 0 and trueG[j, i] == 0 and estG[i, j] == 0 and \
                    estG[j, i] == 1:
                FP += 1
            if trueG[i, j] == 1 and trueG[j, i] == 0 and estG[i, j] == 0 and \
                    estG[j, i] == 0:
                FN += 1
            if trueG[i, j] == 0 and trueG[j, i] == 1 and estG[i, j] == 0 and \
                    estG[j, i] == 0:
                FN += 1
            if trueG[i, j] == 1 and trueG[j, i] == 0 and estG[i, j] == 0 and \
                    estG[j, i] == 1:
                FD += 1
            if trueG[i, j] == 0 and trueG[j, i] == 1 and estG[i, j] == 1 and \
                    estG[j, i] == 0:
                FD += 1
            if trueG[i, j] == 0 and trueG[j, i] == 1 and estG[i, j] == 1 and \
                    estG[j, i] == 1:
                MD += 1
            if trueG[i, j] == 1 and trueG[j, i] == 0 and estG[i, j] == 1 and \
                    estG[j, i] == 1:
                MD += 1
            if trueG[i, j] == 0 and trueG[j, i] == 0 and estG[i, j] == 1 and \
                    estG[j, i] == 1:
                FPMD += 1
    if (TP + FP + FD)>0:
        Precision = TP / (TP + FP + FD)
    else:
        Precision = 0.0
    Recall = TP / sum(sum(trueG))

    if (TP + FN + FD) > 0:
        Recall1 = TP / (TP + FN + FD)
    else:
        Recall1 = 0.0
    #对于二维矩阵np.transpose就是转置
    SHD = sum(sum((trueG != estG) | np.transpose((trueG != estG)))) / 2
    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 'FD': FD, 'MD': MD,
            'FPMD': FPMD,'Precision': Precision, 'Recall': Recall,
            'Recall_NOMD': Recall1, 'SHD': SHD}

#图G中的预测结果进行评估，跟上一个函数很相似，但是它不区分方向，相当于把G看成无向图
def skeleton_metrics(trueG, estG):
    TP,TN,FP,FN = 0,0,0,0
    n = trueG.shape[0]
    for i in range(n):
        for j in range(i):
            if trueG[i, j] == 1 or trueG[j, i] == 1:
                if estG[i, j] != 0 or estG[j, i] != 0:
                    TP += 1
                else:
                    FN += 1
            else:
                if estG[i, j] == 0 and estG[j, i] == 0:
                    TN += 1
                else:
                    FP += 1
    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}

#ij两个结点在skeleton图上是否是联通的（不考虑边的方向）
def check_connect_skel(skeleton, i, j):
    depth = set.union(set(np.where(skeleton[:,i]==1)[0]),
                      set(np.where(skeleton[i,:]==1)[0]))
    checked = depth
    while depth:
        if j in depth:
            return True
        next = {}
        for k in depth:
            next = set.union(next, set.union(set(np.where(skeleton[:,k]==1)[0]),
                                             set(np.where(skeleton[k,:]==1)[0])))
        depth = set.difference(next, checked)
        checked = set.union(checked, depth)
    return False

#返回是否存在一个路径从fr到to(考虑边的方向)
def reachable(dag, fr, to):
    depth = set(np.where(dag[fr,:]==1)[0])
    checked = depth
    while depth:
        if to in depth:
            return True
        next = set()
        for k in depth:
            next = set.union(next, set(np.where(dag[k,:]==1)[0]))
        depth = set.difference(next, checked)
        checked = set.union(checked, depth)
    return False
