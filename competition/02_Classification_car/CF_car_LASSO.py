# -*- coding : utf-8 -*-
"""
    作者:     Fargo
    版本:     1.0
    日期:     2018/09/25

    任务类型：二元分类

    交通事故理赔审核:
    在交通摩擦（事故）发生后，理赔员会前往现场勘察、采集信息，这些信息往往影响着车主是否能够得到保险公司的理赔。
    训练集数据包括理赔人员在现场对该事故方采集的36条信息，信息已经被编码，以及该事故方最终是否获得理赔。
    我们的任务是根据这36条信息预测该事故方没有被理赔的概率。

    变量名	解释
    CaseId	案例编号，没有实际意义
    Q1	理赔员现场勘察采集的信息，Q1代表第一个问题的信息。信息被编码成数字，数字的大小不代表真实的关系。
    Qk	同上，Qk代表第k个问题的信息。一共36个问题。
    Evaluation	表示最终审核结果。0表示授予理赔，1表示未通过理赔审核。在test.csv中，这是需要被预测的标签。

    评价方法为精度-召回曲线下面积(Precision-Recall AUC)，以下简称PR-AUC
    PR-AUC的取值范围是0到1。越接近1，说明模型预测的结果越接近真实结果
"""

# -*- coding: utf-8 -*-
import math

import pandas as pd
from sklearn.linear_model import LogisticRegression

# 读取数据
from sklearn.metrics import average_precision_score

train_data = pd.read_csv("./data_car/train.csv")


def split_train_test(train_data, size=0.8):
    """
        分割训练集和测试集
    """
    print('分割训练集和测试集...')

    # 默认 80%训练集 20%测试集 分割
    n_lines = train_data.shape[0]   # 数据集行数
    split_line_on = math.floor(n_lines * size)  # floor 向下取整
    train_train = train_data.iloc[ : split_line_on, : ]
    train_test = train_data.iloc[  split_line_on : , : ]

    train_train = train_train.reset_index()
    train_test = train_test.reset_index()
    return train_train,train_test


def acc_rmse(y_pred, Evaluation_test):
    pass


def run_main():
    """
        主函数
    """
    # 1.处理数据
    print('处理数据...')
    # 1.1 分割训练集 测试集
    train_train, train_test = split_train_test(train_data)
    # 删除id
    train_train.drop('CaseId', axis=1, inplace=True)
    train_test_real = train_test.drop('CaseId', axis=1)
    train_test_pred = train_test_real.drop('Evaluation', axis=1)

    # 取出训练集的y
    Evaluation_train = train_train.pop('Evaluation')
    Evaluation_test = train_test_real.pop('Evaluation')
    

    # 建立LASSO逻辑回归模型
    clf = LogisticRegression(penalty='l1', C=1.0, random_state=0, solver ='liblinear')
    clf.fit(train_train, Evaluation_train)
    y_pred = clf.predict_proba(train_test_pred)[:, 1]

    print('未通过审核的概率 :', average_precision_score( Evaluation_test,y_pred))




if __name__ == '__main__':
    run_main()



























