import pandas as pd
import numpy as np
import time
from utils import data_preprocess, evaluate_binary, \
    normalize_biweight, skeleton_metrics
from camsearch_prior import greedy_edgeadding
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn_pandas import DataFrameMapper
from skelprune import skeleton, pruning
from RCIT import RCITIndepTest


# 先验知识编码
def prior_knowledge_encode(feature_names,
                          source_nodes=None, direct_edges=None,
                          not_direct_edges=None, happen_before=None):
    """
    :param feature_names: list of string, list of feature names
    :param source_nodes: list of string, list of source node
    :param direct_edges: dictionary, {start1:end1, start2:end2}
    :param not_direct_edges: dictionary, {start1:end1, start2:end2}
    :param happen_before: dictionary,
            {node: [ac1, ac2, ac3], node2: [ac1, ac2, ac3]}
    :return:
    """
    p = len(feature_names)
    feature2index = {}
    for i, feature in enumerate(feature_names): # 枚举特征的序号和名字，并储存（建立名字和数字的索引）
        feature2index[feature] = i              # feature2index 中存的是随机变量

    # 生成 p*p 初始化为 0 的矩阵
    # 类似邻接矩阵的数据结构
    prior_adj = np.zeros((p, p))                # prior_adj 表示有向图中是否有 start->end 的边
    prior_anc = np.zeros((p, p))

    # 设置源节点（source_nodes），源节点没有任何祖先（ancestor）
    if source_nodes: # 自行添加参数，加上后执行
        # 空的就创建，不空就去重
        source_nodes = set(source_nodes)
        for s_node in source_nodes:
            if s_node not in feature2index:
                raise ValueError(
                    f"the feature: {s_node} you provide in the source_nodes "
                    f"is not in the column names")
            prior_adj[:, feature2index[s_node]] = -1  # 该节点（特征名字）的列全部变为 -1
            prior_anc[:, feature2index[s_node]] = -1
            # 无连接
    else: # 默认情形
        source_nodes = set()

    # 根据先验知识设置直接边缘（direct_edges）
    if direct_edges: # 自行添加参数，加上后执行
        set_direct_edges = set()
        for start in direct_edges:
            if start not in feature2index:
                raise ValueError(
                    f"the feature: {start} you provide in the direct_edges"
                    f" is not in the column names")
            for end in direct_edges[start]:
                if end not in feature2index:
                    raise ValueError(
                        f"the feature: {end} you provide in the "
                        f"direct_edges is not in the column names")
                if end in source_nodes:
                    raise ValueError(
                        f"The prior knowledge you provide is conflict you"
                        f" claim the feature {end} is a source node but "
                        f"there is an edge to it.")
                ind_s = feature2index[start]        # 得到该点的序号
                ind_e = feature2index[end]          # 得到指向的点的序号
                prior_adj[ind_s, ind_e] = 1         # 表明两点单向可达
                set_direct_edges.add((start, end))  # 加入到有向图中
                # 建立连接
    else:
        set_direct_edges = set()

    # 根据先验知识设置确定的无直接边缘（not_direct_edges）
    if not_direct_edges: # 自行添加参数，加上后执行
        for start in not_direct_edges:
            if start not in feature2index:
                raise ValueError(
                    f"the feature: {start} you provide in the "
                    f"not_direct_edges is not in the column names")
            for end in not_direct_edges[start]:
                if end not in feature2index:
                    raise ValueError(
                        f"the feature: {end} you provide in the "
                        f"not_direct_edges is not in the column names")
                if (start, end) in set_direct_edges:
                    raise ValueError(
                        f"The prior knowledge you provide is conflict please "
                        f"check the existence of edge {(start, end)}")
                ind_s = feature2index[start]        # 得到点的索引
                ind_e = feature2index[end]          # 得到可到达的点的索引
                prior_adj[ind_s, ind_e] = -1        # 双向可达的状态

    # 根据先验知识设置确定的无祖先（no ancestor）
    if happen_before: # 自行添加参数，加上后执行
                      # 应是 {feature2index: [feature2index, feature2index, feature2index]} 的形式
        for late in happen_before:
            if late not in feature2index:
                raise ValueError(
                    f"the feature: {late} you provide in order information "
                    f"is not in the column names")
            for anc in happen_before[late]:
                if anc not in feature2index:
                    raise ValueError(
                        f"the feature: {anc} you provide in order information "
                        f"is not in the column names")
                if (late, anc) in set_direct_edges:
                    raise ValueError(
                        f"The prior knowledge you provide is conflict "
                        f"please check the existence of edge ({late, anc})")

                ind_s = feature2index[late]
                ind_e = feature2index[anc]
                prior_adj[ind_s, ind_e] = -1  # 双向可达状态
                prior_anc[ind_s, ind_e] = -1  # 两点间的关系状态

    return prior_adj, prior_anc


def data_processing(df, cat_index, normalize='biweight'):
    columns = df.columns
    if normalize == 'biweight':  # 双权
        BiweightScaler = FunctionTransformer(normalize_biweight)   # 创建一个转换器，转换方式为 normalize_biweight 函数，预期得到一个可以获得归一化的双权值
        standardize = [(col, None) if col in cat_index else        # 遍历每一列，如果此列在测试中(col在cat_index中)则不做双权标准化(col, None)，否则做
                      ([col], BiweightScaler) for col in columns]
        x_mapper = DataFrameMapper(standardize)                    # 变成转换器可处理的格式，DataFrame 是二维数据结构，类似表格
        df = x_mapper.fit_transform(df).astype('float32')          # 转换
        df = pd.DataFrame(df, columns=columns)                     # 转换为具有列标签和行标签的格式，行标签默认为索引，列标签为 columns
    elif normalize == 'standard':  # 过程同上，但转化方法不同
        standardize = [(col, None) if col in cat_index else
                      ([col], StandardScaler()) for col in columns]
        x_mapper = DataFrameMapper(standardize)
        df = x_mapper.fit_transform(df).astype('float32')
        df = pd.DataFrame(df, columns=columns)
    else:
        raise NotImplementedError(
            f"currently we only support 'biweight' and 'standard'.")
    # 编码
    df[cat_index] = df[cat_index].astype(object)
    X_encode = data_preprocess(df)  # 将处理好的数据编码，方式为编成独热码
    return df, X_encode


def mixed_causal(df, X_encode,model_para, base_model_para,
                prior_adj=None, prior_anc=None, selMat=None):
    # 导入模型参数
    step1_maxr = model_para['step1_maxr']
    step3_maxr = model_para['step3_maxr']
    maxNumParents = model_para['maxNumParents']
    num_f = model_para['num_f']
    num_f2 = model_para['num_f2']
    cv_split = model_para['cv_split']
    ll_type = model_para['ll_type']
    alpha = model_para['alpha']
    downsampling= model_para['downsampling']
    indep_pvalue = model_para['indep_pvalue']
    base_model = base_model_para['base_model']
    score_type = model_para['score_type']

    p = df.shape[1]  # 返回数据的列数

    #######################################################################
    # Step 1.使用 pc 算法进行框架学习
    indepTest = RCITIndepTest(suffStat=X_encode, down=downsampling, # 独立性测试 MRCIT
                              num_f=num_f, num_f2=num_f2)
    if selMat is None:  # 此处默认没有，如果已经有了这个 skeleton 的话可以跳过，没学习 skeleton 就学习，统计学习时间
        t1 = time.time()
        skel = skeleton(indepTest, labels=range(p),
                        m_max=step1_maxr, alpha=indep_pvalue,
                        priorAdj=prior_adj,
                        )
        selMat = skel['sk']                     # selMat 为学习到的 Skelton

        step1_train_time = time.time() - t1     # 得到 step1_train_time，框架学习训练时间
    else:
        step1_train_time = 0

    #######################################################################
    # step 2: 基于贪心搜索算法，创建 dag
    # 选择交叉检验的方式
    if base_model == "lgbm":  # lgbm 模型用编码
        X = X_encode
    elif base_model =="gam":  # gam 模型直接用数据
        X = df
    else:
        raise NotImplementedError(
            f"currently we only support 'lgbm' and 'gam'.")
    # 贪心搜索
    t2 = time.time()
    dag2 = greedy_edgeadding(df, X, selMat,
                            maxNumParents=maxNumParents,
                            alpha=alpha,
                            cv_split=cv_split,
                            ll_type=ll_type,
                            base_model_para=base_model_para,
                            prior_adj=prior_adj,
                            prior_anc=prior_anc,
                            score_type = score_type,
                            )
    step2_train_time = time.time() - t2

    ######################################################################
    # step 3: 通过条件独立性（conditional independence test）测试去除边缘
    # 剪枝
    t3 = time.time()
    dag = pruning(indepTest, dag2, m_max=step3_maxr,
                  alpha=indep_pvalue, priorAdj=prior_adj)

    step3_train_time = time.time() - t3

    return(selMat, dag2, dag, step1_train_time,
          step2_train_time, step3_train_time)


def evaluate(trueG, skel_bool, dag2, dag):
    skel = skel_bool.astype('int')
    # skelton 修改类型
    if skel.shape != trueG.shape != dag.shape:  # 保证邻接矩阵的尺寸相同
        raise ValueError(f"the shape of true adjacency matrix and the "
                        f"predicted skeleton and dag is not same!")
    # 分别评估 3 种
    skl_result = pd.DataFrame(skeleton_metrics(trueG, skel), index=[0])
    # print(skl_result)
    dag2_result = pd.DataFrame(evaluate_binary(trueG, dag2), index=[0]) # 将召回率精确率等数据
    dag_result = pd.DataFrame(evaluate_binary(trueG, dag), index=[0])
    return skl_result, dag2_result, dag_result
