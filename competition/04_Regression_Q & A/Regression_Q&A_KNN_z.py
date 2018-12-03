# -*- coding : utf-8 -*-
"""
    作者:     Fargo
    版本:     1.0
    日期:     2018/09/26

    任务类型：时间序列、回归

    交通事故理赔审核:
    我们给出美国某大型问答社区从2010年10月1日到2016年11月30日，每天新增的问题的个数和回答的个数。
    任务是预测2016年12月1日到2017年5月1日，该问答网站每天新增的问题数和回答数。在本练习赛中，日期是唯一的特征。

    变量名	解释
    id	行编号
    date	年月日
    questions	当日新增的问题的数量。在预测集中，这是要被预测的数值。
    answers	当日新增的答案的数量。在预测集中，这是要被预测的数值。

    我们采用绝对百分比误差均值（MAPE）作为评价标准。 MAPE越小，说明模型预测的结果越接近真实结果
"""

# -*- coding: utf-8 -*-

import math
from datetime import date
import holidays
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


def run_main():
    """
        主函数
    """
    # 读取数据
    train = pd.read_csv('./data_Q & A./train.csv')
    test = pd.read_csv('./data_Q & A./test.csv')
    submit = pd.read_csv('./data_Q & A./sample_submit.csv')

    # 构造非线性特征
    cols_lr = ['id', 'sqrt_id']
    train['sqrt_id'] = np.sqrt(train['id'])
    test['sqrt_id'] = np.sqrt(test['id'])

    # 构造星期、月、年特征
    train['date'] = pd.to_datetime(train['date'])
    train['d_w'] = train['date'].dt.dayofweek
    train['d_m'] = train['date'].dt.month
    train['d_y'] = train['date'].dt.year

    test['date'] = pd.to_datetime(test['date'])
    test['d_w'] = test['date'].dt.dayofweek
    test['d_m'] = test['date'].dt.month
    test['d_y'] = test['date'].dt.year

    # 增加节假日特征
    train['holidays'] = 0
    for i in range(len(train)):
        if train['date'][i] in holidays.US():
            train['holidays'][i] =  1

    test['holidays'] = 0
    for i in range(len(test)):
        if test['date'][i] in holidays.US():
            test['holidays'][i] =  1

    cols_knn = ['d_w', 'd_m', 'd_y', 'holidays']


    # 根据特征['id', 'sqrt_id']，构造线性模型预测questions
    reg = LinearRegression()
    reg.fit(train[cols_lr], train['questions'])
    q_fit = reg.predict(train[cols_lr])
    q_pred = reg.predict(test[cols_lr])

    # 根据特征['id', 'sqrt_id']，构造线性模型预测answers
    reg = LinearRegression()
    reg.fit(train[cols_lr], train['answers'])
    a_fit = reg.predict(train[cols_lr])
    a_pred = reg.predict(test[cols_lr])

    # 得到questions和answers的训练误差
    q_diff = train['questions'] - q_fit
    a_diff = train['answers'] - a_fit

    # 把训练误差作为新的目标值，使用特征cols_knn，建立kNN模型
    reg = KNeighborsRegressor(n_neighbors=4, algorithm= 'brute')
    reg.fit(train[cols_knn], q_diff)
    q_pred_knn = reg.predict(test[cols_knn])

    reg = KNeighborsRegressor(n_neighbors=4, algorithm= 'brute')
    reg.fit(train[cols_knn], a_diff)
    a_pred_knn = reg.predict(test[cols_knn])

    # 输出预测结果至my_Lr_Knn_prediction.csv
    submit['questions'] = q_pred + q_pred_knn
    submit['answers'] = a_pred + a_pred_knn
    submit.to_csv('fargo_Lr_Knn_prediction.csv', index=False)

if __name__ == '__main__':
    run_main()


