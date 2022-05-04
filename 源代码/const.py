DEFAULT_MODEL_PARA = {
    'step1_maxr': 1,  # step 1 的最大环境设置
    'step3_maxr': 3,  # step 3 的最大环境设置
    'num_f': 100,     # 条件集的随机 fft 特征数
    'num_f2': 10,     # 测试变量的随机 fft 特征数
    'indep_pvalue': 0.05,  # 独立测试的阀值
    'alpha': 0.0,
    'll_type': 'local',    # koe估算kde: 'local' 或者 'global'
    'cv_split': 5,         # 交叉验证次数（可以为 0 次）
    'downsampling': False, # 是否需要在 MRCIT 中进行下采样
    'maxNumParents': 10,   # NumParents 最大值
    'score_type': 'bic'    # 分数类型: 'bic'、'll'、'aic'；LH: 对数似然
}

# 检查 Lightgbm 中的参数，Lightgbm 基于决策树算法的梯度提升算法(lgb)
DEFAULT_LGBM_PARA = {
    'boosting': 'gbdt',        # 提升类型：传统的梯度提升决策树——决策树类型
    'num_leaves': 31,          # 每个基学习器的最大叶子节点
    'feature_fraction': 0.5,   # 特征分数/子特征处理列采样
    'learning_rate': 0.05,     # 梯度下降的步长。常用 0.1, 0.001, 0.003
    'num_boost_round': 200,    # 迭代次数
}

# 检查 Pygam 中的参数
DEFAULT_GAM_PARA = {
    "spline_order": 10,  # 样条函数
    "lam": 0.6,          # 分层
    "n_jobs": 1,         # 个数
    "use_edof": True,    # 使用edof标志
}
