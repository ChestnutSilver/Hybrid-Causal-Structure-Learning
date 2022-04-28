import numpy as np
import math
import random
from utils import normalize
from scipy.spatial.distance import pdist
from scipy.linalg import cholesky, solve_triangular
from rpy2.robjects.numpy2ri import numpy2rpy

import rpy2.robjects.packages as rpackages
momentchi2 = rpackages.importr('momentchi2')

#第一步骨架学习的混合类型随机化CIT，对应论文3.2部分
#请看类定义

#random Fourier features随机傅里叶特征

#x是某一个或者或一组特征的所有样本的值，离散变量以独热吗的形式传递进来
#num_f=10 不知道 傅里叶特征数？
#这个函数通过RFF方法计算出RFF的具体值，送到core函数进行核函数计算
#输出是一个矩阵 大小 样本数*numf
def rff_mixed(x, num_f=10):

    #使一维向量reshape成二维矩阵
    if len(x.shape) == 1:
        x = x.reshape(-1,1)

    cat_idx = []#个人觉得是离散变量discrete
    
    for i in range(x.shape[1]):
        #如果这一列只有两个不同的值（独热吗），cat_idx增加列索引
        if len(np.unique(x[:,i])) == 2:
            cat_idx.append(i)
            
    #提取出来所有的离散变量
    x_disc = x[:, cat_idx]
    #x_cont是xdisc的补集，即所有的连续变量
    x_cont = x[:, np.setdiff1d(range(x.shape[1]), cat_idx)]

    #离散变量的个数？，连续变量的个数
    n_disc, n_cont = np.sum(x_disc[0, :]), x_cont.shape[1]
    
    #按照比例分配特征数
    num_f_cont = int(n_cont/(n_disc+n_cont)*num_f)
    num_f_disc = num_f - num_f_cont

    if x_cont.shape[1] != 0:
        #样本数和特征数
        r, c = x_cont.shape
        r1 = min(r, 500)
        
        #pdist的功能 计算矩阵X样本之间（m*n）的欧氏距离(2-norm) ，返回值为 Y (m*m)为压缩距离元组或矩阵。
        #也就是计算前r1个样本的欧式距离中位数
        sigma = np.median(pdist(x_cont[:r1, :], "euclidean"))
        if sigma == 0 or np.isnan(sigma):
            sigma = 1
        #w，b是满足一定分布的随机值，w的形状是连续rff特征数量*连续变量的个数
        w = (1 / sigma) * np.random.normal(0, 1, size=num_f_cont * c).reshape(num_f_cont, c)
        b = 2 * math.pi * np.random.uniform(0, 1, size=num_f_cont)
        #用b作为元素构造矩阵，每一列是相同的连续rff特征数量个值，一共有样本数个，只是为了方便计算
        b = np.tile(b.reshape(-1, 1), (1, r))
        #RFF算法的z(x)计算结果
        #与这篇博客的公式形式类似，略有不同
        #https://blog.csdn.net/a358463121/article/details/111541560
        cont_feat = np.sqrt(2) * np.cos(np.dot(w, x_cont.T) + b).T

    #与上面的基本一样区别在于sigma的计算
    if x_disc.shape[1] != 0:
        r, c = x_disc.shape
        r1 = min(r, 500)

        sigma = 1

        w = (1 / sigma) * np.random.normal(0, 1, size=num_f_disc * c).reshape(num_f_disc, c)
        b = 2 * math.pi * np.random.uniform(0, 1, size=num_f_disc)
        b = np.tile(b.reshape(-1, 1), (1, r))
        disc_feat = np.sqrt(2) * np.cos(np.dot(w, x_disc.T) + b).T
    
    #将两个特征进行连接
    
    if x_cont.shape[1]!=0 and x_disc.shape[1] != 0:
        return np.concatenate([cont_feat,disc_feat],axis=1)
    elif x_cont.shape[1] != 0:
        return cont_feat
    else:
        return disc_feat


#rit核函数，没有z这个附加条件
#four_x 特征x的RFF 大小 样本数*numf
#four_y 特征y的RFF
#r是样本数
#num_f RFF中包含的特征数
def RIT_core(four_x, four_y, r, num_f):
    #cov会把一行当成一个特征，一列当成一个样本，所以需要进行转置
    #cov(a,b)会把ab的所有特征都两两求协方差，所以大小是 2numf x 2numf
    #计算出特征之间的协方差，Cxy的大小为num_f x num_f
    Cxy = np.cov(four_x.T,four_y.T)[:-num_f, -num_f:]
    #标准差，r是自由度？
    Sta = r*np.sum(Cxy*Cxy)
    
    #每一个样本都减去特征的均值
    res_x = four_x - np.tile(np.mean(four_x,0),(r,1))
    res_y = four_y - np.tile(np.mean(four_y,0),(r,1))

    #没有看太懂，计算出样本之间的协方差？
    #d是所有的二元对，x的某一个特征，y的某一个特征
    d = np.array([(x, y) for x in range(four_x.shape[1]) for y in range(four_y.shape[1])])
    res = res_x[:,d[:,1]]*res_y[:,d[:,0]]
    Cov = 1/r * np.dot(res.T,res)

    eig_d = np.linalg.eig(Cov)#返回特征值和特征向量
    #晒选出是正实数的特征值
    eig_d = eig_d[0][eig_d[0].imag==0]
    eig_d = eig_d[eig_d.real > 0]
    eig_d = eig_d.real
    
    #看不懂
    try:
        p = 1- momentchi2.lpb4(numpy2rpy(eig_d), Sta.item())[0]
    except:
        p = 1 - momentchi2.hbe(numpy2rpy(eig_d), Sta.item())[0]

    p = max(p,np.exp(-40))
    return p

#rcit核函数
def RCIT_core(four_x, four_y, four_z, r, num_f, num_f2):
    Cxy = np.cov(four_x.T,four_y.T)[:-num_f2, -num_f2:]
    Cxz = np.cov(four_x.T,four_z.T)[:-num_f, -num_f:]
    Czy = np.cov(four_z.T,four_y.T)[:-num_f2, -num_f2:]

    Czz = np.cov(four_z.T)
    #cholesky分解
    Lzz = cholesky(Czz + np.eye(num_f)*(1e-10), lower=True)
    #解决ax=b方程中的x，（假定a是一个上/下三角矩阵）
    A = solve_triangular(Lzz, Cxz.T, lower=True)
    e_x_z = np.dot(four_z, solve_triangular(Lzz.T, A, lower=False))

    A = solve_triangular(Lzz, Czy, lower=True)
    B = solve_triangular(Lzz.T, A, lower=False)
    e_y_z = np.dot(four_z, B)

    res_x = four_x - e_x_z
    res_y = four_y - e_y_z

    Cxy_z = Cxy - np.dot(Cxz, B)
    Sta = r*np.sum(Cxy_z*Cxy_z)
    #剩下的不走与原来相同，上面的看不太懂了，语句的含义都已经注释，但是实际的意义不明
    d = np.array([(x, y) for x in range(four_x.shape[1]) for y in range(four_y.shape[1])])
    res = res_x[:,d[:,1]]*res_y[:,d[:,0]]
    Cov = 1/r * np.dot(res.T,res)

    eig_d = np.linalg.eig(Cov)
    eig_d = eig_d[0][eig_d[0].imag==0]
    eig_d = eig_d[eig_d.real > 0]
    eig_d = eig_d.real

    try:
        p = 1- momentchi2.lpb4(numpy2rpy(eig_d), Sta.item())[0]
    except:
        p = 1 - momentchi2.hbe(numpy2rpy(eig_d), Sta.item())[0]

    p = max(p,np.exp(-40))
    return p

#随机独立性条件检验，这个类的功能就是做独立性检验，返回一个p值，
#如果p值大于alpha，则接受独立假设，说明输入的数据是独立的则说明两个数据没有边
#参数:
#X_encode,经过离散值独热吗处理的df,他是一个列表，有df的列数个矩阵，每个矩阵对应一个特征
#down(是否下采样)
#num_f,num_f2 模型固有参数

class RCITIndepTest(object):
    def __init__(self, suffStat,  down=False, num_f=100, num_f2=10):
        #n是样本数，c是特征数，r是处理后的最终样本数
        self.n, self.c = suffStat[0].shape[0], len(suffStat)
        #进行下采样处理
        if not down:
            self.suffStat = suffStat
            self.r = self.n
        else:
            #idx是保留的样本索引
            #也就是说他会控制样本数不超过特征数（独热吗算一个特征）的100倍
            self.idx = random.sample(range(self.n), self.c * 100) \
                if self.c * 100 < self.n  else np.array(range(self.n))
            self.suffStat = []
            for i in range(len(suffStat)):
                self.suffStat.append(suffStat[i][self.idx])
            self.r = len(self.idx)
        # keep the fft transmation
        self.fft_feature_f2 = {}#将计算好的RFF存在里面，避免重复计算
        self.fft_feature_f = {}
        self.num_f = num_f
        self.num_f2 = num_f2

    #KRR核岭回归，返回估计的p值
    #xy是需要检验的边，z是x为起点的所有不是x--y的边的索引，具体形式见skelprune.py，kwarg是空的
    #以下的判断if语句的作用都是判别某一组特征的RFF是否已经被计算过
    def fit(self, x, y, z=None, **kwargs):
        if z is None or len(z) == 0:
            if x not in self.fft_feature_f2:
                self.fft_feature_f2[x] = normalize(
                    rff_mixed(self.suffStat[x], num_f=self.num_f2))
            if y not in self.fft_feature_f2:
                self.fft_feature_f2[y] = normalize(
                    rff_mixed(self.suffStat[y], num_f=self.num_f2))
            return RIT_core(four_x=self.fft_feature_f2[x],
                            four_y=self.fft_feature_f2[y],
                            r=self.r, num_f=self.num_f2)
        else:
            # print(x, y, z)
            if x not in self.fft_feature_f2:
                self.fft_feature_f2[x] = normalize(
                    rff_mixed(self.suffStat[x], num_f=self.num_f2))
            y = frozenset([y] + z)#不可修改的集合
            if y not in self.fft_feature_f2:
                #数组拼接函数，把y里的所有特征的具体值全部拿进来
                suffStaty = np.concatenate([self.suffStat[i]
                                            for i in y], axis=1)
                self.fft_feature_f2[y] = normalize(
                    rff_mixed(suffStaty, num_f=self.num_f2))
            setz = frozenset(z)
            if setz not in self.fft_feature_f:
                suffStatz = np.concatenate([self.suffStat[i]
                                            for i in setz], axis=1)
                self.fft_feature_f[setz] = normalize(
                    rff_mixed(suffStatz, num_f=self.num_f))
            return RCIT_core(four_x=self.fft_feature_f2[x],
                             four_y=self.fft_feature_f2[y],
                             four_z=self.fft_feature_f[setz],
                             r=self.r, num_f=self.num_f, num_f2=self.num_f2)
