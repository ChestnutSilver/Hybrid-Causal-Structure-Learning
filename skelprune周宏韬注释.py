import numpy as np
import itertools
from utils import reachable

#骨架学习函数
#参数含义
#indepTest 条件独立性检验类
#labels range(p),p是df的列数 df是alram_simulate.csv
#m_max model_para['step1_maxr']=1
#alpha 显著性标准0.05对应95%置信水平
#priorAdj 不知道
#**kwargs 似乎是空的
def skeleton(indepTest, labels, m_max, alpha=0.05, priorAdj=None, **kwargs):
    #创建一个p*p大小的矩阵，矩阵内的元素都为None
    sepset = [[None for i in range(len(labels))] for i in range(len(labels))]
    
    #同上，形成完全无向图，如果需要研究边 i--j，则为真
    # form complete undirected graph, true if edge i--j needs to be investigated
    G = [[True for i in range(len(labels))] for i in range(len(labels))]

    #把对角线元素去掉，去掉自己到的边
    for i in range(len(labels)): G[i][i] = False

    # done flag
    done = False

    ord = 0
    n_edgetests = {}#个人认为这个变量存储的是不是第ord次循环进行过的独立检验次数
    
    #这里的循环因为mmax=1，只会执行一遍
    while done != True and any(G) and ord <= m_max:
        ord1 = ord + 1
        n_edgetests[ord1] = 0
        done = True
        G1 = G.copy()
        
        #提取出所有边
        ind = [(i, j)
               for i in range(len(G))
               for j in range(len(G[i]))
               if G[i][j] == True
               ]
        #遍历所有边
        for x, y in ind:
            if priorAdj is not None:
                if priorAdj[x,y]==1 or priorAdj[y,x]==1:
                    continue

            if G[y][x] == True:#如果反向边存在
                #nbrs是x为起点的所有不是x--y的边的索引
                nbrs = [i for i in range(len(G1)) if G1[x][i] == True and i != y]
                
                if len(nbrs) >= ord:#如果上面的边数大于等于ord
                    if len(nbrs) > ord:
                        done = False
                    #itertools.combinations(nbrs, ord)的作用是返回nbrs的大小为ord的所有子集
                    #nbrs_S也就是所有以x为起点的ord条边的终点索引
                    for nbrs_S in set(itertools.combinations(nbrs, ord)):
                        n_edgetests[ord1] = n_edgetests[ord1] + 1
                        pval = indepTest.fit(x, y, list(nbrs_S), **kwargs)
                        if pval >= alpha:
                            G[x][y] = G[y][x] = False
                            sepset[x][y] = set(nbrs_S)#将没有通过检验的一组数据存在sepset
                            break
        ord += 1

    return {'sk': np.array(G),'sepset': sepset,}

#裁剪函数
#参数含义
#indepTest 条件独立性检验类
#dag 无向图
#m_max model_para['step3_maxr']=3
#alpha 显著性标准0.05对应95%置信水平
#priorAdj 不知道 预处理返回的矩阵
#**kwargs 似乎是空的
def pruning(indepTest, dag, m_max, alpha=0.05, priorAdj=None, **kwargs):

    for r in range(1, m_max):
        dag1 = dag.copy()
        edges = np.where(dag == 1)#edges是一个两个元素的元组，edges[0]是所有的行索引，[1]是列索引

        for k in range(len(edges[0])):
            xi, xj = edges[0][k], edges[1][k]#取出xi--xj这条边
            
            #不懂这个priorAdj是什么
            if priorAdj is not None:
                if priorAdj[xi,xj] == 1 or priorAdj[xj,xi] ==1:
                    continue
                if priorAdj[xi,xj] == -1:
                    dag1[xi, xj] = 0
                    continue

            ifdelete = dag.copy()#删掉这条边的dag
            ifdelete[xi, xj] = 0

            considerz = []
            #parent是ifdelete里面所有以xi为终点的边的起点的索引
            for parent in list(np.where(ifdelete[:, xi] == 1)[0]):
                #如果这个起点仍然和xj联通
                if reachable(ifdelete, parent, xj): considerz.append(parent)
            for parent in list(np.where(ifdelete[:, xj] == 1)[0]):
                if reachable(ifdelete, parent, xi): considerz.append(parent)

            considerz = list(set(considerz))#去掉重复元素

            #继续做独立性检验，与前面skelton里的部分非常相似
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
