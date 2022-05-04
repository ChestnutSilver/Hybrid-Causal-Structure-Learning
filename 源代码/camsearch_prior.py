import itertools
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings("ignore")


def greedy_edgeadding(df, X_encode, selMat, maxNumParents,base_model_para,
                      alpha=0.0,
                      cv_split=5, ll_type='local',
                      prior_adj=None, prior_anc=None,
                      score_type='bic', debug=False):
    # 首先在 selMat 中对先验知识进行编码，去除无向边
    if prior_adj is not None:
        if prior_adj.shape != selMat.shape:
            raise ValueError("the shape of prior_adj is not same as selMat")
        selMat = selMat & (prior_adj >= 0)
        must_have = np.argwhere(prior_adj > 0)    # 返回 prior_adj 中满足条件的位置索引
    else: # 这是默认情况
        must_have = []

    if prior_anc is not None:
        if prior_anc.shape != selMat.shape:
            raise ValueError("the shape of prior_anc is not same as selMat")
        selMat = selMat & (prior_anc >= 0)
        not_prior_anc = prior_anc < 0
        np.fill_diagonal(not_prior_anc, False)    # 填充not_prior_anc 对角线
    else:
        not_prior_anc = np.zeros(selMat.shape, dtype=bool)

    path = np.zeros(selMat.shape)
    np.fill_diagonal(path, 1)                     # 填充对角线
    Adj = np.zeros(selMat.shape)

    # 计算分数
    ScoreMatComputer = ScoreMatCompute(
        df, X_encode, selMat,
        maxNumParents=maxNumParents,
        cv_split=cv_split,
        ll_type=ll_type,
        score_type=score_type,
        debug=debug,
        base_model_para=base_model_para)

    # scoreMat 初始化
    scoreMat, scoreNodes = ScoreMatComputer.initialScoreMat()
    # 贪心地添加边
    while np.max(scoreMat) > -float('inf'):
        diff = scoreMat-np.transpose(scoreMat)
        # 两个"-inf"的不同之处在于 set to -inf
        diff[np.isnan(diff)] = -float('inf')
        # "non -inf"和"-inf"的不同之处在于 set to 0 来避免错误
        # 在上一步中没有对称纯化（none symmetric puring）时赋值
        diff[np.isinf(diff)] = 0.0
        weighted_gain_diff = (1 - alpha) * scoreMat + alpha * diff

        # 如果有一些边必须加上
        if len(must_have) > 0:    # 如果有必须加的，就 add
            mind = np.argmax([weighted_gain_diff[tuple(i)] for i in must_have])   # argmax 找某个维度的最大值，每次加最大的
            row_index, col_index = must_have[mind][0], must_have[mind][1]
            must_have = np.delete(must_have, mind, axis=0)
        else:
            # 找到最好的边
            # np.unravel_index 返回的是拉成一维之后的索引
            # 这里找的是最大值在原来中的索引
            row_index, col_index = np.unravel_index(
                weighted_gain_diff.argmax(), weighted_gain_diff.shape)

        # 需要考虑是否添加 (row_index, col_index)
        # 将避免原因顺序 (cause order)
        # 更新有向无环路径
        t_path = path.copy()
        t_path[row_index, col_index] = 1
        DescOfNewChild = np.append(np.where(t_path[col_index,:]==1), col_index)   # 父节点；添加边
        AncOfNewParent = np.append(np.where(t_path[:,row_index]==1), row_index)   # np.where 返回的是满足条件的坐标
        # itertools.product 用来求笛卡尔积，这里用来组合出所有的元素位置 
        # 找到所有已经够存在路径的
        for element in list(itertools.product(AncOfNewParent, DescOfNewChild)):
            t_path[element] = 1

        # 如果有一些避免，那么改变不包括边缘和为"-inf"设置分数
        if np.any(not_prior_anc & (t_path == 1)):
            # not_prior_anc 的情况下，如果已经存在路径了，删掉他
            scoreMat[row_index, col_index] = -float('inf')
            continue
        else:
            if debug:
                print(f"before add the edge ({row_index, col_index}), the "
                      f"score of {col_index} is {scoreNodes[col_index]}")

            scoreNodes[col_index] = (scoreNodes[col_index] +
                                    scoreMat[row_index, col_index])

            if debug:
                print(f"after the score is {scoreNodes[col_index]}")
                print(scoreNodes)

            ScoreMatComputer.set_scoreNodes(scoreNodes)
            scoreMat[row_index, col_index] = -float('inf')
            scoreMat[col_index, row_index] = -float('inf')
            Adj[row_index, col_index] = 1
            path = t_path.copy()
            scoreMat[np.transpose(path) == 1] = -float('inf')
            # 更新 scoreMat；算分
            ScoreMatComputer.set_scoreMat(scoreMat)
            scoreMat, scoreNodes = ScoreMatComputer.scoreupdate(
                Adj=Adj, j=col_index)
    return Adj


# 计算没有模型的每个变量的对数似然
# param x_col:pandas 的一个数据系列
# 返回：每个变量的对数似然(log likelihood)
def compute_init_ll(x_col, bandwidth=1.0, kernel='gaussian'):
    # 就是文献中初始得分的定义
    if x_col.dtypes == 'O' or x_col.dtypes == 'bool':
        # 离散的通过经验概率
        prob_dic = x_col.value_counts(normalize=True).to_dict()   # value_counts 统计出现频率
        prob_list = x_col.replace(prob_dic)
        return np.mean(np.log(prob_list))
    else:
        # 连续的通过核密度估计
        data_x = x_col.values
        kde = gaussian_kde(data_x)
        logprob = np.log(kde.evaluate(data_x))
        """
        kde = KernelDensity(bandwidth=bandwidth, kernel=kernel)
        kde.fit(data_x[:, None])

        # score_samples 返回的是 log of the probability density
        logprob = kde.score_samples(data_x[:, None])
        """
        return np.mean(logprob)


class ScoreMatCompute(object):
    def __init__(self, X, X_encode, selMat, maxNumParents, base_model_para,
                cv_split=5,
                ll_type='local', score_type='bic', debug=False):
        # 初始的设置，复制
        self.X = X
        self.X_encode = X_encode
        self.selMat = selMat
        self.p = selMat.shape[0]
        self.maxNumParents = maxNumParents
        self.valid_pair = np.argwhere(selMat)
        self.scoreMat = np.ones(selMat.shape) * (-float('inf'))
        # scoreNodes 是每个变量的对数 log(p(x))
        self.scoreNodes = (self.X).apply(compute_init_ll, axis=0).values
        self.score_type = score_type
        self.debug = debug
        self.Dn = X.shape[0]
        self.pn = X.shape[1]
        self.bicterm = np.log(self.Dn) / self.Dn / 2
        self.base_model_para = base_model_para
        base_model = base_model_para['base_model']
        if self.debug:
            print("score of each variable without add any edge")
            print(self.scoreNodes)

        # 分类选取交叉检验
        if base_model == 'lgbm':
            from lgbm_cvmodel import ModelWrapper
        elif base_model == 'gam':
            from gam_cvmodel import ModelWrapper
        else:
            raise NotImplementedError(
                f"currently we only support 'lgbm' and 'gam'.")
        self.ModelWrapper = ModelWrapper
        self.cv_split = cv_split
        self.ll_type = ll_type

    def set_scoreMat(self, scoreMat):
        self.scoreMat = scoreMat

    def set_scoreNodes(self, scoreNodes):
        self.scoreNodes = scoreNodes

    def _compute_ll(self, x):
        Y = self.X.iloc[:, x[1]]
        # isinstance 判断是否是已知类型
        if isinstance(self.X_encode, list):
            model = self.ModelWrapper(X=self.X_encode[x[0]], Y=Y,
                                      cv_split=self.cv_split,
                                      ll_type=self.ll_type,
                                      para=self.base_model_para
                                      )
        elif isinstance(self.X_encode, pd.DataFrame):
            X_input = self.X_encode.iloc[:, [x[0]]]
            model = self.ModelWrapper(X=X_input, Y=Y, cv_split=self.cv_split,
                ll_type=self.ll_type,para=self.base_model_para)
        else:
            raise ValueError("The type of X_encode must be list of numpy "
                            "array or pandas DataFrame")

        ll, edof = model.fit()
        if edof == 0:
            edof = 1
        #print(ll, edof, self.bicterm, x[0], x[1])
        if self.score_type == 'll':
            self.scoreMat[x[0], x[1]] = ll
        elif self.score_type == 'bic':
            self.scoreMat[x[0], x[1]] = ll - edof*self.bicterm
        elif self.score_type == 'aic':
            self.scoreMat[x[0], x[1]] = ll - edof/self.Dn

    def initialScoreMat(self):
        np.apply_along_axis(self._compute_ll, axis=1, arr=self.valid_pair)  # apply_along_axie 将某个函数沿坐标轴作用与 arr
        # 当前，self.scoreMat 是每个模型的分数
        if self.debug:
            print("score of each variable when adding the first edge")
            print(self.scoreMat)
        self.scoreMat = self.scoreMat - self.scoreNodes
        # 当前，self.scoreMat是添加边与不添加边的分数差异(adding the edge and not add the edge)
        if self.debug:
            print("score improve of each variable when adding the first edge")
            print(self.scoreMat)
        return self.scoreMat, self.scoreNodes

    def _update_ll(self, x):
        Y = self.X.iloc[:, x[-1]]
        if isinstance(self.X_encode, list):
            X_input = np.concatenate([self.X_encode[i] for i in x[:-1]],
                                    axis=1)
        # np.concatenate 用于合并数组
        elif isinstance(self.X_encode, pd.DataFrame):
            X_input = self.X_encode.iloc[:, x[:-1]]
        else:
            raise ValueError("The type of X_encode must be list of numpy "
                            "array or pandas DataFrame")
        model2 = self.ModelWrapper(X=X_input, Y=Y, cv_split=self.cv_split,
                                  ll_type=self.ll_type,
                                  para=self.base_model_para)

        ll, edof = model2.fit()
        if edof==0:
            edof = len(x)-2
        if self.score_type == 'll':
            self.scoreMat[x[-2], x[-1]] = ll
        elif self.score_type == 'bic':
            self.scoreMat[x[-2], x[-1]] = ll - edof*self.bicterm
        elif self.score_type == 'aic':
            self.scoreMat[x[-2], x[-1]] = ll - edof/self.Dn

    def _fillninf(self, x):
        self.scoreMat[x[-2], x[-1]] = -float('inf')

    def scoreupdate(self, Adj, j):
        existingParOfJ = np.where(Adj[:, j] == 1)[0]
        # 找有连接的
        notAllowedParOfJ = np.setdiff1d(
            np.where(self.scoreMat[:, j] == -float('inf'))[0],
            np.append(existingParOfJ, [j]))
        if len(existingParOfJ) + len(notAllowedParOfJ) < self.p:
            # 获取 undecided candidate 的索引
            toUpdate = np.setdiff1d(np.arange(self.p), np.concatenate(
                (existingParOfJ, notAllowedParOfJ, [j])))
            # np.setdiff1d 作差集
            update_need = np.concatenate(
                (
                np.tile(existingParOfJ, (len(toUpdate), 1)),  # existingParOfJ
                toUpdate.reshape(-1, 1),        # 添加的 candidate
                np.tile(j, (len(toUpdate), 1))  # 目标
                )
                , axis=1)
            # 这里，reshape 弄成 1 列
            # 最后一个 np.tile()，这里是 j 沿 y 轴复制 len，x 轴不变
            if update_need.shape[0] > 0:        # 需要更新的情况
                if len(existingParOfJ) < self.maxNumParents:
                    np.apply_along_axis(self._update_ll, axis=1,
                                        arr=update_need)
                else:
                    np.apply_along_axis(self._fillninf, axis=1,
                                        arr=update_need)
                if self.debug:
                    print("the score matrix after adding an edge")
                    print(self.scoreMat)
                self.scoreMat[:, j] = self.scoreMat[:, j] - self.scoreNodes[j]
                if self.debug:
                    print(
                        "score improve of each variable when adding an edge")
                    print(self.scoreMat)
        return self.scoreMat, self.scoreNodes
