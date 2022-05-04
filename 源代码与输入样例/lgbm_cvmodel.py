import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.metrics import log_loss
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
from sklearn.model_selection import StratifiedKFold, KFold


class LgbClsModel(object):
    def __init__(self,
                is_unbalance='true',       # 用于 binary 分类，使用列集中的 pos/neg 分数
                boosting='gbdt',           # 提升类型：传统的梯度提升决策树——决策树类型
                num_leaves=31,             # 每个基学习器的最大叶子节点
                feature_fraction=0.5,      # 特征分数/子特征处理列采样
                learning_rate=0.05,        # 梯度下降的步长。常用 0.1, 0.001, 0.003
                num_boost_round=20,        # 迭代次数
                num_class=2,               # 只用于 multiclass 分类
                early_stopping_round=3,    # 如果一次验证数据的一个度量在最近的 round 中没有提高，模型将停止训练
                bagging_fraction=0.5,      # 每次迭代时用的数据比例，建树的样本采样比例
                bagging_freq=20            # bagging 的次数(与 bagging_fraction 同时设置)
                ):
        # lgb 模型参数配置
        self.parameters = {
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'is_unbalance': is_unbalance,
            'boosting': boosting,
            'num_leaves': num_leaves,
            'feature_fraction': feature_fraction,
            'learning_rate': learning_rate,
            'num_class': num_class,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'verbose': -1    # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
        }
        self.num_boost_round = num_boost_round
        self.early_stopping_round = early_stopping_round
        self.model = None

    # lgb 模型训练，得出预测结果
    def fit(self, X_train=None, y_train=None, X_test=None, y_test=None):
        lgb_train = lgb.Dataset(X_train, y_train)
        test_data = lgb.Dataset(X_test, label=y_test)
        # 训练
        self.model = lgb.train(
            self.parameters,
            lgb_train,
            valid_sets=test_data,
            verbose_eval=False,
            num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_round
        )
        # 预测数据集
        y_pred = self.model.predict(X_test)
        # 对数损失是逻辑模型的负对数可能性
        return -log_loss(y_true=y_test, y_pred=y_pred, normalize=False)


class LgbRegModel(object):
    def __init__(self,
                boosting='gbdt',           # 提升类型：传统的梯度提升决策树——决策树类型
                num_leaves=31,             # 每个基学习器的最大叶子节点
                feature_fraction=0.5,      # 特征分数/子特征处理列采样
                learning_rate=0.05,        # 梯度下降的步长
                num_boost_round=20,        # 迭代次数
                bandwidth=0.4,
                kernel='gaussian',
                early_stopping_round=3,    # 如果一次验证数据的一个度量在最近的 round 中没有提高，模型将停止训练
                bagging_fraction=0.5,      # 每次迭代时用的数据比例,建树的样本采样比例
                bagging_freq=20            # bagging 的次数(与 bagging_fraction 同时设置)
                ):
        # lgb 模型参数配置
        self.parameters = {
            'objective': 'regression',     # 目标函数
            'boosting': boosting,
            'num_leaves': num_leaves,
            'feature_fraction': feature_fraction,
            'learning_rate': learning_rate,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'verbose': -1                  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
        }
        self.early_stopping_round = early_stopping_round
        self.num_boost_round = num_boost_round
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.model = None

    # lgb 模型训练，得出预测结果
    def fit(self, X_train=None, y_train=None, X_test=None, y_test=None):
        lgb_train = lgb.Dataset(X_train, y_train)
        test_data = lgb.Dataset(X_test, label=y_test)
        # 训练
        self.model = lgb.train(
            self.parameters,
            lgb_train,
            valid_sets=test_data,
            verbose_eval=False,
            num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_round
        )

        # 预测数据集
        y_pred = self.model.predict(X_test)
        residule = y_test - y_pred

        kde = gaussian_kde(residule)
        logprob = np.log(kde.evaluate(residule))

        return residule, np.sum(logprob)


# 模型封装: Lightgbm 基于决策树算法的梯度提升算法
class ModelWrapper(object):
    def __init__(self,
                X,
                Y,               # pandas 数据序列
                para=None,       # 参数类型
                cv_split=5,      # 交叉验证次数（可能需要0次）
                ll_type='local'  # koe 估算 kde: 'local' 或者 'global'
                ):
        """
        :param X:
        :param Y: Y 是一个 pandas 数据序列（data series）
        :param is_unbalance: 是否使用列集中的pos/neg分数
        :param boosting: 传统的梯度提升决策树——决策树类型
        :param num_leaves: 每个基学习器的最大叶子节点
        :param feature_fraction: 特征分数/子特征处理列采样
        :param learning_rate: 梯度下降的步长
        :param num_boost_round: 迭代次数
        """
        boosting = para['boosting']
        num_leaves = para['num_leaves']
        feature_fraction = para['feature_fraction']
        learning_rate = para['learning_rate']
        num_boost_round = para['num_boost_round']

        self.X = X
        self.Y = Y
        if (Y.dtypes == 'O' or Y.dtypes == 'bool' or
                Y.dtype.name == 'category' or Y.dtypes == 'int'): # 模型参数定义，数据类型为布尔型、int 型、类属
            num_class = len(Y.unique())
            self.pred_model = LgbClsModel(
                boosting=boosting,
                num_leaves=num_leaves,
                feature_fraction=feature_fraction,
                learning_rate=learning_rate,
                num_boost_round=num_boost_round,
                num_class=num_class
            )
        else: # 模型参数定义：其他数据类型
            self.pred_model = LgbRegModel(
                boosting=boosting,
                num_leaves=num_leaves,
                feature_fraction=feature_fraction,
                learning_rate=learning_rate,
                num_boost_round=num_boost_round
            )
        self.fited = False
        self.cv_split = cv_split
        self.ll_type = ll_type

    def fit(self):
        n_split = self.cv_split
        ll_type = self.ll_type
        total_ll = 0
        total_num = 0
        if (self.Y.dtypes == 'O' or self.Y.dtypes == 'bool'
                or self.Y.dtype.name == 'category'): # 模型训练预测：数据类型为布尔型、int 型、类属
            le = preprocessing.LabelEncoder()
            le.fit(self.Y)
            self.Y = le.transform(self.Y)
            if n_split == 0: # 交叉验证次数为 0
                sumll = self.pred_model.fit(X_train=self.X, y_train=self.Y,
                                            X_test=self.X, y_test=self.Y)
                total_ll += sumll
                total_num += len(self.Y)
            else:
                skf = StratifiedKFold(n_splits=n_split)
                skf.get_n_splits(self.X, self.Y)
                for train_ind, test_ind in skf.split(self.X, self.Y):
                    X_train, X_test = self.X[train_ind], self.X[test_ind]
                    y_train, y_test = self.Y[train_ind], self.Y[test_ind]
                    sumll = self.pred_model.fit(X_train=X_train, y_train=y_train,
                                                X_test=X_test, y_test=y_test)
                    total_ll += sumll
                    total_num += len(y_test)
            return total_ll/total_num, 0
        else: # 模型训练预测：其他数据类型
            residule = np.array([])
            if n_split == 0: # 交叉验证次数为 0
                presidule, sumll = self.pred_model.fit(X_train=self.X, y_train=self.Y, X_test=self.X, y_test=self.Y)
                residule = np.append(residule, presidule)
                total_ll += sumll
                total_num += len(self.Y)
            else:
                kf = KFold(n_splits=n_split)
                kf.get_n_splits(self.X)
                total_num = 0
                for train_ind, test_ind in kf.split(self.X):
                    X_train, X_test = self.X[train_ind], self.X[test_ind]
                    y_train, y_test = self.Y[train_ind], self.Y[test_ind]
                    presidule, sumll = self.pred_model.fit(
                        X_train=X_train, y_train=y_train,
                        X_test=X_test, y_test=y_test)
                    residule = np.append(residule, presidule)

                    total_ll += sumll
                    total_num += len(y_test)
            if ll_type=='local':
                return total_ll/total_num, 0
            else:
                kde = gaussian_kde(residule)
                logprob = np.log(kde.evaluate(residule))
            return np.mean(logprob), 0
