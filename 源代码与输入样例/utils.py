### 本文件：通用的数据处理与评估函数
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from astropy import stats


# 非线性转化，返回 x 的归一化双权值（转换器用的函数）
# 采用的方法是使用双权标度，在方差较大的情况下合理估计标准化；如果分布的标准差太小，则用标准化的标准差代替
def normalize_biweight(x, eps=1e-10):
    median = np.median(x)                      # 计算分布的中位数
    scale = stats.biweight.biweight_scale(x)   # 返回分布的双权重比例。双权标度是确定分布标准差的稳健统计
    if np.std(x) < 1e+2 or np.isnan(scale) or scale < 1e-4:  # np.std() 是得到 x 标准差, np.isnan() 判断是否非数字
        norm =  (x-np.mean(x))/np.std(x)       # 标准化（太小或者非数字时就用个体标准差/样本标准差代替双权值）
    else:
        norm = (x - median) / (scale + eps)    # 否则，返回（x-中位数）/（双权标准差+eps）
    return norm


# 将 x 中的数据标准差标准化
def normalize(x):
    norm = lambda x: (x-np.mean(x))/np.std(x)  # 标准化标准差
    return np.apply_along_axis(norm, 0, x)     # 用 lambda 表达式代替函数，将 x 中的数据标准差标准化


# 数据处理，将 df 中的数据进行处理，离散值转换成独热码，连续值不变
# 参数:
#   dataframe 是带行列名的二维矩阵，iloc 可以通过 0，1，2 的索引访问 df 元素 
#   iloc 可以通过位置索引访问 df 元素 
#   loc 可以通过列名，行名访问 df 元素  
#   'category'是一种类型，'O'是 o 不是 0，表示 object
def data_preprocess(df):
    def _encoding(i):
        if df.iloc[:,i].dtype == 'O' or df.iloc[:, i].dtype.name == 'category':  # 每行第 i 个输入数据是分类有关的
            tempX = df.iloc[:, i].values.reshape(-1, 1)   # 把数据变成只有 1 列的，行自动计算（reshape 中的 -1 表示该维度的大小自动计算）
            enc = OneHotEncoder(handle_unknown='ignore')  # 创建一个对象，把每一行数据编成独热码，参数表示遇到没标识过的时不报错
            enc.fit(tempX)                                # 加入训练数据（也就是告诉这个对象有哪些类别）
            out = enc.transform(tempX).toarray()          # 得到 tempX 的独热码
        else:
            out = df.iloc[:, i].values.reshape(-1, 1)     # 不是则直接变成 1 列，不编码
        return out
    p = df.shape[1]                                       # 得到 df 的列数 p
    X_encode = [_encoding(i) for i in np.arange(p)]       # 每一列编码一次
    return X_encode                                       # 返回编码


# trueG 是真实值，estG是估计值
# 对图 G 中的预测结果进行评估，边的两个方向一起被计算，trueG 中某一条边不会出现两个方向都有
# 作用是计算准确率、召回率等，与下面的函数不同的是，这应是一种二分评估
def evaluate_binary(trueG, estG):
    # TP,TN,FP,FN 中 T,F 代表 true, false; P,N 代表 postive, negative
    # TP：被模型预测为正类的正样本（真阳）真实和预测都为 1（两个方向）
    # TN：被模型预测为负类的负样本（真阴）真实和预测都为 0（两个方向）
    # FP：被模型预测为正类的负样本（假阳）真实 0 预测 1（一个方向，另外一个方向TN）
    # FN：被模型预测为负类的正样本（假阴）真实 1 预测 0（一个方向，另外一个方向TN）
    # FD：预测错误的相反正样本，即 01 预测为 10，10 预测为 01（两个方向一个FP，另外一个FN）
    # MD：01，10 都预测为 11（一个FP 一个TP）
    # FPMD：00 预测为 11（两个FP）
    # TP 与 TN 是预测正确的
    # precise 准确率 TP/(TP + FP + FD)
    # recall 召回率，TP/所有边  recall1 召回率，(TP + FN + FD)
    # SHD 结构汉明距离，所有边预测和真实不同的个数(不论哪一个方向)/2
    TP, TN, FP, FN, FD, MD, FPMD = 0, 0, 0, 0, 0, 0, 0
    n_node = trueG.shape[0]    # 行数
    for i in range(1, n_node):
        for j in range(i):
            # trueG中 10 01 代表正样本，00 代表负样本
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
        Precision = TP / (TP + FP + FD)  # 计算精确率
    else:
        Precision = 0.0
    Recall = TP / sum(sum(trueG))        # 计算总召回率

    if (TP + FN + FD) > 0:
        Recall1 = TP / (TP + FN + FD)
    else:
        Recall1 = 0.0
    SHD = sum(sum((trueG != estG) | np.transpose((trueG != estG)))) / 2 # transpose()是转置，也就是计算所有 trueG[i,j][j,i]!=estG[i,j][j,i] 的和
    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 'FD': FD, 'MD': MD,
            'FPMD': FPMD,'Precision': Precision, 'Recall': Recall,
            'Recall_NOMD': Recall1, 'SHD': SHD}

# 图 G 中的预测结果进行评估，计算被预测正确或错误的正负样本
# 跟上一个函数很相似，但是它不区分方向，相当于把 G 看成无向图；即：这里只要 [i,j] 或 [j,i] 满足一个就可以
def skeleton_metrics(trueG, estG):
    # TP,TN,FP,FN 中 T,F 代表 true, false; P,N 代表 postive, negative
    #TP：被模型预测为正类的正样本
    #TN：被模型预测为负类的负样本
    #FP：被模型预测为正类的负样本
    #FN：被模型预测为负类的正样本
    TP,TN,FP,FN = 0,0,0,0
    n = trueG.shape[0]
    for i in range(n):
        for j in range(i):
            # 按照定义算
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


# 在 i 的范围内检查 j 是否已经存在，skeleton 是一个邻接矩阵
# 即：检查 ij 两个结点在 skeleton 图上是否是联通的（不考虑边的方向）
def check_connect_skel(skeleton, i, j):
    depth = set.union(set(np.where(skeleton[:,i]==1)[0]),    # 将第 i 行或第 i 列中为 1 的行索引索引加入集合
                      set(np.where(skeleton[i,:]==1)[0]))
    checked = depth
    while depth:
        if j in depth:
            return True
        next = {}
        for k in depth:
            next = set.union(next, set.union(set(np.where(skeleton[:,k]==1)[0]),
                                            set(np.where(skeleton[k,:]==1)[0])))
        depth = set.difference(next, checked)    # 获得有差异的集合，即还没被检查过的
        checked = set.union(checked, depth)      # 更新已经检查过的
    return False

# 返回是否存在一个路径从 fr 到 to (考虑边的方向)
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
