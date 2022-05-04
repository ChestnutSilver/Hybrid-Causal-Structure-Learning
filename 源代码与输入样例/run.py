# coding: utf-8
import pandas as pd
import numpy as np
from mixed_causal import mixed_causal, \
    prior_knowledge_encode, data_processing, evaluate
import easydict

from const import DEFAULT_MODEL_PARA, DEFAULT_LGBM_PARA, DEFAULT_GAM_PARA


args = easydict.EasyDict({
    'data_file': 'alram_simulate.csv',  # 数据文件的位置和名字
    'cat_index': ['2','3','4','7','8','10','11','13','16','17','18','19','21','22','27','33','36','37'],
    'true_G':'alarm.csv',  # 实际图（true graph）文件的位置和名字
    'model_para': {'step1_maxr': 1, 'step3_maxr': 3, 'num_f': 100,
                  'num_f2': 20, 'indep_pvalue': 0.05, 'downsampling': False,
                  'cv_split': 5, 'll_type': 'local', 'alpha': 0.0,
                  'maxNumParents': 10, 'score_type': 'bic'
                  },
    # 可用于测试步骤 2 在不同设置下的性能；即:
    # 如果已经有了骨架文件"alaram_simulate_skl.csv"
    # 然后可以使用它来避免再次运行步骤 1
    'skl_file': "",
    'base_model':'lgbm',  # 'lgbm' or 'gam'
    'base_model_para': {},
    'source_nodes': [],
    'direct_edges': {},
    'not_direct_edges': {},
    'happen_before': {},
})


def check_model_para(model_para, base_model, base_model_para):
    model_para_out = {}
    for para in DEFAULT_MODEL_PARA:  # 自己设置的参数
        if para in model_para:
            model_para_out[para] = model_para[para]
        else:                        # 使用默认参数
            model_para_out[para] = DEFAULT_MODEL_PARA[para]
    base_model_para_out = {}
    base_model_para_out['base_model'] = base_model
    if base_model=='lgbm':   # 如果用模型 lgbm_cvmodel 的话导入该模型参数，可以自己设置，可以默认
        for para in DEFAULT_LGBM_PARA:
            if para in base_model_para:
                base_model_para_out[para] = base_model_para[para]
            else:
                base_model_para_out[para] = DEFAULT_LGBM_PARA[para]
    elif base_model=='gam':  # 如果用模型 gam_cvmodel 的话导入该模型参数，可以自己设置，可以默认
        for para in DEFAULT_GAM_PARA:
            if para in base_model_para:
                base_model_para_out[para] = base_model_para[para]
            else:
                base_model_para_out[para] = DEFAULT_GAM_PARA[para]
    else:  # 两个都不是抛出异常
        raise NotImplementedError(
            f"currently we only support 'lgbm' and 'gam'.")
    return model_para_out, base_model_para_out


if __name__ == '__main__':
    # df = pd.read_csv("C:\\Users\\lenovo\\Desktop\\AAAI2022-HCM-main\\alram_simulate.csv") # 读取表格文件
    df = pd.read_csv(args.data_file) # 读取表格文件

    if args.skl_file == "":          # 如果没有骨架文件，就不用
        selMat = None
    else:                            # 如果有骨架文件，读入
        selMat = pd.read_csv(args.skl_file, header=None).values > 0
    # csv(comma-separated values),hearder=None 取消表头 
    # 读取 csv 文件
    print(df.columns)
    print(df.columns.values)

    # 导入模型及其参数
    model_para_out, base_model_para_out = check_model_para(        
        args.model_para, args.base_model, args.base_model_para)

    # 数据预处理，也就是编码，根据需要可对数据进行双权'biweight'和标准'standard'编码
    df, X_encode = data_processing(df, args.cat_index, normalize='biweight')

    # 根据自己填入的参数告诉模型哪些特征是有关系的，可以是单向的，也可以是双向的，返回邻接矩阵
    prior_adj, prior_anc = prior_knowledge_encode(
        feature_names=df.columns, source_nodes=args.source_nodes,
        direct_edges=args.direct_edges, not_direct_edges=args.not_direct_edges)

    # 运用文中算法得到关系的有向图和运行时间
    selMat, dag2, dag, step1_time, step2_time, step3_time = mixed_causal(
        df, X_encode, model_para= model_para_out,
        prior_adj=prior_adj, prior_anc=prior_anc,
        base_model_para=base_model_para_out, selMat=selMat)
    print(step1_time, step2_time, step3_time)

    # 将获得的结果存起来，参数分别是文件名字，要保持的矩阵，分隔符
    np.savetxt(args.data_file[:-4]+'_skl.csv', selMat, delimiter=",")
    np.savetxt(args.data_file[:-4]+'_dag2.csv', dag2, delimiter=",")
    np.savetxt(args.data_file[:-4]+'_dag.csv', dag, delimiter=",")

    if args.true_G != '':
        trueG = pd.read_csv(args.true_G).values  # 获得表格的所有值
        # 将结果与真实值进行评估，返回损失率，精确率，TP  TN  FP  FN 等
        skl_result, dag2_result, dag_result = evaluate(trueG, selMat, dag2,dag)
        print(skl_result)
        print(dag2_result)
        print(dag_result)
        print(trueG)
        print(selMat)
        print(dag2)
        print(dag)