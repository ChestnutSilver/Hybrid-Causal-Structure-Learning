from pygam import LogisticGAM, LinearGAM, s, f
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.metrics import log_loss
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
from sklearn.model_selection import StratifiedKFold, KFold


# 建立逻辑回归模型将多分类任务转为二分类任务，并将输入的 train 放入模型
# 返回 test 的对数损失值和模型有效自由度的均值
def GamClsModel(sfun=None,
              n_jobs=1,         # 并行作业数，使用的是默认值 1
              X_train=None,
              y_train=None,
              X_test=None,
              y_test=None
              ):
    base_model = LogisticGAM(sfun) # 建立逻辑模型
    ovr_classifier = OneVsRestClassifier(base_model, n_jobs=n_jobs) # 转为二分类任务
    ovr_classifier.fit(X_train, y_train) # 输入参数
    prob_pred = ovr_classifier.predict_proba(X_test) # 预测
    # 计算模型的对数似然（log likelihood）
    edofs = [est.statistics_['edof'] for est in ovr_classifier.estimators_]
    ll = -log_loss(y_true=y_test, y_pred=prob_pred, normalize=False) # 得到损失率
    return ll, np.mean(edofs) # mean 为去平均值


# 建立线性回归模型，利用输入的 train 得到 testX 对数预测值，与 testY 相减得到差，
# 利用高斯核进行核密度估计，最后返回差值，核密度估计的和，模型的有效自由度
def GamRegModel(sfun=None,
                X_train=None,
                y_train=None,
                X_test=None,
                y_test=None
                ):

    gam = LinearGAM(sfun).fit(X_train, y_train)
    y_pred = gam.predict(X_test)
    residule = y_test - y_pred

    kde = gaussian_kde(residule)               # 进行高斯核密度分布估计
    logprob = np.log(kde.evaluate(residule))   # 类似于取值并取对数

    return residule, np.sum(logprob), gam.statistics_['edof']


class ModelWrapper(object):
    def __init__(self,
                X,
                Y,                             # X,Y 皆为测试集
                para=None,                     # 参数类型
                train_test_split_ratio=0.0,
                cv_split=5,                    # 交叉验证的次数
                ll_type='local'                # koe 估算 kde: 'local' 或者 'global'
                ):
        """
        :param X: X 是一个 pandas 数据框架（data frame）
        :param Y: Y 是一个 pandas 数据序列（data series）
        DEFAULT_GAM_PARA = {
            "spline_order": 10,
            "lam": 0.6,
            "n_jobs": 1,
            "use_edof": True,
        }
        """
        spline_order = para['spline_order']
        lam = para['lam']
        n_jobs = para['n_jobs']
        use_edof = para['use_edof']  # 是否需要计算自由度

        self.X = X
        self.train_test_split_ratio = train_test_split_ratio
        p = X.shape[1]
        cols = list(X.columns)
        if (X[cols[0]].dtypes == 'O' or X[cols[0]].dtypes == 'bool'
                or X[cols[0]].dtype.name == 'category'):  # 判断是否是离散特征
            sfun = f(0, lam=lam)  # 每一个特征都对应一个 item，sfun 是所有 item 的集合
        else:
            sfun = s(0, spline_order=spline_order)

        for i in range(1, p):
            if (X[cols[i]].dtypes == 'O' or X[cols[i]].dtypes == 'bool'
                    or X[cols[i]].dtype.name == 'category'):
                sfun = sfun + f(i, lam=lam)
            else:
                sfun = sfun + s(i, spline_order=spline_order)
        self.Y = Y
        self.sfun = sfun
        self.n_jobs = n_jobs
        self.use_edof = use_edof
        self.cv_split = cv_split
        self.ll_type = ll_type

    def fit(self):
        n_split = self.cv_split
        ll_type = self.ll_type
        total_ll = 0
        total_num = 0
        total_edof = 0
        # 分情况使用逻辑/线性回归。如果数据类型是离散变量用逻辑回归，否则用线性回归。
        if (self.Y.dtypes == 'O' or self.Y.dtypes == 'bool'
                or self.Y.dtype.name == 'category'):
            # 下面，将标签标准化；例：1, 2, 6, 7 ---> 0, 1, 2, 3
            le = preprocessing.LabelEncoder()
            le.fit(self.Y)
            self.Y = le.transform(self.Y)   # 将 self.Y 标签化为 0 ~ len(self.Y)-1 之间的数字
            if n_split == 0:  # 划分次数为 0 不需要划分
                sumll, edof = GamClsModel(sfun=self.sfun, n_jobs=self.n_jobs,
                                          X_train=self.X, y_train=self.Y,
                                          X_test=self.X, y_test=self.Y)
                total_ll += sumll           # 损失
                total_num += len(self.Y)    # 预测的样本数
                total_edof += edof          # 自由度
            else:
                skf = StratifiedKFold(n_splits=n_split) # 等比例划分
                skf.get_n_splits(self.X, self.Y)
                # 随机将 self.X, self.Y 划分，实现交叉验证
                # 具体就是将将训练集等比例划分为 n_split 份，然后每次取一份，把这一份当成训练集放到测试集上，
                # 进行 n_split 次后得到分类率的平均值，作为该模型平均值
                for train_ind, test_ind in skf.split(self.X, self.Y):
                    X_train, X_test = self.X.iloc[train_ind], self.X.iloc[test_ind]
                    y_train, y_test = self.Y[train_ind], self.Y[test_ind]
                    sumll, edof = GamClsModel(
                        sfun=self.sfun, n_jobs=self.n_jobs,
                        X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test)
                    total_ll += sumll
                    total_num += len(y_test)
                    total_edof += edof
            if self.use_edof:
                return total_ll/total_num, total_edof/n_split
            else:
                return total_ll/total_num, 0
        else:
            residule = np.array([])
            if n_split == 0:  # 划分次数为 0 则不需要划分
                presidule, sumll, edof = GamRegModel(
                    sfun=self.sfun, X_train=self.X,
                    y_train=self.Y, X_test=self.X, y_test=self.Y)
                residule = np.append(residule, presidule)

                total_ll += sumll
                total_num += len(self.Y)
                total_edof += edof
            else:
                # 具体就是将将训练集比例划分为 n_split 份，然后每次取一份，把这一份当成训练集放到测试集上，
                # 进行 n_split 次后得到分数的平均值，作为该模型平均值
                # 注意，与上面不同的是 KFold 不是等比例划分
                kf = KFold(n_splits=n_split)  # 随机比例划分
                kf.get_n_splits(self.X)
                residule = np.array([])
                for train_ind, test_ind in kf.split(self.X):
                    X_train, X_test = self.X.iloc[train_ind], self.X.iloc[test_ind]
                    y_train, y_test = self.Y[train_ind], self.Y[test_ind]
                    presidule, sumll, edof = GamRegModel(
                        sfun=self.sfun, X_train=X_train,
                        y_train=y_train, X_test=X_test, y_test=y_test)
                    residule = np.append(residule, presidule)

                    total_ll += sumll
                    total_num += len(y_test)
                    total_edof += edof

            # 有 local，则直接返回利用线性回归得到的训练值与测试值只差的高斯核密度分布之和；没有 local，则还需要进行平均值
            # self.use_edof 代表返不返回平均有效自由度
            if ll_type == 'local':
                if self.use_edof:
                    return total_ll / total_num, total_edof/n_split
                else:
                    return  total_ll / total_num, 0
            else:
                # residule 是所有折的 ytest-ypre 的集合，相当于把函数内的所有偏差综合在一起进行高斯核密度估计
                # 也就是说在 global 模式下，不采用每一折的高斯核密度估计进行取均值，而是重新计算整个高斯核密度估计
                kde = gaussian_kde(residule)              # 高斯核密度分布
                logprob = np.log(kde.evaluate(residule))  # 取对数
                if self.use_edof:
                    return np.mean(logprob), total_edof/n_split
                else:
                    return np.mean(logprob), 0
