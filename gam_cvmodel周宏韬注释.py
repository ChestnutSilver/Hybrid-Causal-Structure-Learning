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

#多分类逻辑回归，返回对数损失值和平均自由度
#sfun

#njobs 我的理解是并行作业数，应该没啥太大的用处，使用的是默认值1
'''
The number of jobs to use for the computation: the n_classes one-vs-rest problems are computed in parallel.
None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.
'''
def GamClsModel(sfun=None,
               n_jobs=1,
               X_train=None,
               y_train=None,
               X_test=None,
               y_test=None
               ):
    base_model = LogisticGAM(sfun)#逻辑回归
    ovr_classifier = OneVsRestClassifier(base_model, n_jobs=n_jobs)#多分类器
    ovr_classifier.fit(X_train, y_train)
    prob_pred = ovr_classifier.predict_proba(X_test)
    #  calculate log likelihood of the model
    edofs = [est.statistics_['edof'] for est in ovr_classifier.estimators_]
    ll = -log_loss(y_true=y_test, y_pred=prob_pred, normalize=False)
    return ll, np.mean(edofs)

#广义线性回归
#返回 偏差矩阵和核密度估计的对数和，有效自由度
def GamRegModel(sfun=None,
                X_train=None,
                y_train=None,
                X_test=None,
                y_test=None
                ):

    gam = LinearGAM(sfun).fit(X_train, y_train)
    y_pred = gam.predict(X_test)
    residule = y_test - y_pred

    kde = gaussian_kde(residule)#高斯核密度估计
    logprob = np.log(kde.evaluate(residule))

    return residule, np.sum(logprob), gam.statistics_['edof']


class ModelWrapper(object):
    def __init__(self,
                 X,
                 Y,
                 para=None,
                 train_test_split_ratio=0.0,
                 cv_split=5,#交叉验证的折数
                 ll_type='local'  # 'local' or 'global', kow to estimate kde
                 ):
        """
        :param X: X is a pandas data frame
        :param Y: Y is a pandas data series
        DEFAULT_GAM_PARA = {
            "spline_order": 10,
            "lam": 0.6,
            "n_jobs": 1,
            "use_edof": True,
        }
        """
        spline_order = para['spline_order']
        lam = para['lam']#前两个是s和f函数的参数没看懂是什么意思，具体可以在官网的连接中查看，连接在下面
        n_jobs = para['n_jobs']
        use_edof = para['use_edof']#是否需要计算自由度？，true

        self.X = X
        self.train_test_split_ratio = train_test_split_ratio
        p = X.shape[1]
        cols = list(X.columns)
        if (X[cols[0]].dtypes == 'O' or X[cols[0]].dtypes == 'bool'
                or X[cols[0]].dtype.name == 'category'):#判断是否是离散特征
            #这里的s函数和f函数在官网的定义https://pygam.readthedocs.io/en/latest/api/terms.html?highlight=pygam.s#pygam.terms.s
            #具体是啥没搞懂，每一个特征都对应一个item，sfun是所有item的集合
            sfun = f(0, lam=lam)
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
        #如果Y是离散变量，则我们做逻辑回归
        if (self.Y.dtypes == 'O' or self.Y.dtypes == 'bool'
                or self.Y.dtype.name == 'category'):
            #下面三行是将标签标准化，比如原来是1，2，6，7--->0,1,2,3
            le = preprocessing.LabelEncoder()
            le.fit(self.Y)
            self.Y = le.transform(self.Y)
            #如果不进行k则交叉验证
            if n_split == 0:
                sumll, edof = GamClsModel(sfun=self.sfun, n_jobs=self.n_jobs,
                                          X_train=self.X, y_train=self.Y,
                                          X_test=self.X, y_test=self.Y)
                total_ll += sumll#损失
                total_num += len(self.Y)#预测的样本数
                total_edof += edof#自由度
            #进行k折交叉验证，和上面一摸一样
            else:
                skf = StratifiedKFold(n_splits=n_split)
                skf.get_n_splits(self.X, self.Y)
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
        #跟上面一模一样只是换了一个回归方法
        else:
            residule = np.array([])
            if n_split == 0:
                presidule, sumll, edof = GamRegModel(
                    sfun=self.sfun, X_train=self.X,
                    y_train=self.Y, X_test=self.X, y_test=self.Y)
                residule = np.append(residule, presidule)

                total_ll += sumll
                total_num += len(self.Y)
                total_edof += edof
            else:
                kf = KFold(n_splits=n_split)
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

            if ll_type == 'local':
                if self.use_edof:
                    return total_ll / total_num, total_edof/n_split
                else:
                    return  total_ll / total_num, 0
            else:
                #residule是所有折的ytest-ypre的集合，相当于把函数内的所有偏差综合在一起进行高斯核密度估计
                #也就是说在global模式下，不采用每一折的高斯核密度估计进行取均值，而是重新计算整个高斯核密度估计
                kde = gaussian_kde(residule)
                logprob = np.log(kde.evaluate(residule))
                if self.use_edof:
                    return np.mean(logprob), total_edof/n_split
                else:
                    return np.mean(logprob), 0
