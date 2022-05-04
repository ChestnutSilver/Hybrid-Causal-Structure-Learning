import numpy as np
import itertools
from utils import reachable


# 框架学习：PC-stable 算法 [ 在因果马尔可夫假设和因果忠实性假设下学习有向无环图 (DAGs) ]
# 迭代地扩大条件集的大小，如果节点之间的对应变量通过 MRCIT（条件）独立，则丢弃节点之间的边
# 参数含义:
#   indepTest 条件独立性检验类
#   labels range(p), p是 df 的列数, df 是 alram_simulate.csv
#   m_max model_para['step1_maxr']=1
#   alpha 显著性标准 0.05 对应 95% 置信水平
def skeleton(indepTest, labels, m_max, alpha=0.05, priorAdj=None, **kwargs):
    # 创建一个 p*p 大小的矩阵，矩阵内的元素都为 None
    sepset = [[None for i in range(len(labels))] for i in range(len(labels))]
    # 把有向图 G 的有向边变成无向边，形成完整无向图；如果需要研究边 i--j，则为真
    G = [[True for i in range(len(labels))] for i in range(len(labels))]
    # 把对角线元素去掉，去掉自己到的边
    for i in range(len(labels)): G[i][i] = False
    done = False     # done 标志
    ord = 0
    n_edgetests = {} # 第ord次循环进行过的独立检验次数

    # 这里的循环 mmax=1，只执行一遍
    while done != True and any(G) and ord <= m_max:
        ord1 = ord + 1
        n_edgetests[ord1] = 0
        done = True
        G1 = G.copy()
        # 提取出所有边
        ind = [(i, j)
              for i in range(len(G))
              for j in range(len(G[i]))
              if G[i][j] == True
              ]
        # 遍历所有边
        for x, y in ind:
            if priorAdj is not None:
                if priorAdj[x,y]==1 or priorAdj[y,x]==1:
                    continue

            # 并行测试 MRCIT 条件是否独立
            if G[y][x] == True:             # 如果反向边存在
                # nbrs 是 x 为起点的所有不是 x--y 的边的索引
                nbrs = [i for i in range(len(G1)) if G1[x][i] == True and i != y]
                if len(nbrs) >= ord:        # 如果上面的边数大于等于ord
                    if len(nbrs) > ord:
                        done = False

                    # itertools.combinations(nbrs, ord) 的作用是返回 nbrs 的大小为 ord 的所有子集
                    # nbrs_S 也就是所有以 x 为起点的 ord 条边的终点索引
                    for nbrs_S in set(itertools.combinations(nbrs, ord)):
                        n_edgetests[ord1] = n_edgetests[ord1] + 1
                        pval = indepTest.fit(x, y, list(nbrs_S), **kwargs)
                        if pval >= alpha:
                            G[x][y] = G[y][x] = False
                            sepset[x][y] = set(nbrs_S)    # 将没有通过检验的一组数据存在 sepset
                            break
        ord += 1

    return {'sk': np.array(G),'sepset': sepset,}

# 剪枝算法：
# 利用 MRCITs 测试每个父子对条件作用对所有其他直接原因的独立性，对避免选择难以确定的阈值而新增的"多余"的边进行修剪
# 参数含义:
#   indepTest 条件独立性检验类
#   dag 无向图
#   m_max model_para['step3_maxr']=3
#   alpha 显著性标准 0.05 对应 95% 置信水平
#   priorAdj 预处理返回的矩阵
def pruning(indepTest, dag, m_max, alpha=0.05, priorAdj=None, **kwargs):

    for r in range(1, m_max):
        dag1 = dag.copy()
        edges = np.where(dag == 1)              # edges 是一个两个元素的元组，edges[0]是所有的行索引，[1]是列索引

        for k in range(len(edges[0])):
            xi, xj = edges[0][k], edges[1][k]   # 取出 xi--xj 这条边

            if priorAdj is not None:
                if priorAdj[xi,xj] == 1 or priorAdj[xj,xi] ==1:
                    continue
                if priorAdj[xi,xj] == -1:
                    dag1[xi, xj] = 0
                    continue

            ifdelete = dag.copy()               # 删掉这条边的 dag
            ifdelete = dag.copy()               
            ifdelete[xi, xj] = 0

            # 增枝：在 DAG 条件约束下，会贪婪地添加边
            considerz = []
            # parent 是 ifdelete 里面所有以 xi 为终点的边的起点的索引
            for parent in list(np.where(ifdelete[:, xi] == 1)[0]):
                # 如果这个起点仍然和 xj 联通
                if reachable(ifdelete, parent, xj): considerz.append(parent)
            for parent in list(np.where(ifdelete[:, xj] == 1)[0]):
                if reachable(ifdelete, parent, xi): considerz.append(parent)

            considerz = list(set(considerz))    # 去掉重复元素

            # 剪枝：通过 MRCITs 以测试每个父子对条件作用对所有其他直接原因的独立性，如果相应的父子是条件独立的，则删除“多余”边
            if len(considerz) > r:
                if len(considerz) == 1:
                    z = considerz[0]
                    pvalue = indepTest.fit(xi, xj, z, **kwargs)
                    if pvalue > alpha:
                        dag1[xi, xj] = 0
                        continue
                else:
                    for nbrs_z in set(itertools.combinations(considerz, r)):
                        pvalue = indepTest.fit(xi, xj, list(nbrs_z), **kwargs)
                        if pvalue > alpha:
                            dag1[xi, xj] = 0
                            break
        dag = dag1.copy()
    return dag
