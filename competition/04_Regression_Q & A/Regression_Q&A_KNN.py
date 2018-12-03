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



# 读取数据
train_data = pd.read_csv('./data_Q & A./train.csv')


def split_train_test(train_data,size=0.8):

    print('分割训练集 测试集')

    # 默认 80%训练集 20%测试集 分割
    n_lines = train_data.shape[0]   # 数据集行数
    split_line_on = math.floor(n_lines * size)  # floor 向下取整
    train_train = train_data.iloc[ : split_line_on, : ]
    train_test = train_data.iloc[  split_line_on : , : ]

    train_train = train_train.reset_index()
    train_test = train_test.reset_index()

    return train_train, train_test


def acc_MAPE(Q_test, Q_pred):
    n = len(Q_pred)

    ape = [ np.abs(Q_test[i] - Q_pred[i]) / Q_test[i] for i in range(n) ]
    mape = sum(ape) / n
    return  mape

def run_main():
    """
        主函数
    """
    # 分割训练集 测试集
    train_train, train_test = split_train_test(train_data)

    # 取出测试集 的 真实值 Q 和 A
    Q_test = train_test.pop('questions')
    A_test = train_test.pop('answers')

    # 构造星期、月、年特征
    train_train['date'] = pd.to_datetime(train_train['date'])
    train_train['d_w'] = train_train['date'].dt.dayofweek
    train_train['d_m'] = train_train['date'].dt.month
    train_train['d_y'] = train_train['date'].dt.year

    train_test['date'] = pd.to_datetime(train_test['date'])
    train_test['d_w'] = train_test['date'].dt.dayofweek
    train_test['d_m'] = train_test['date'].dt.month
    train_test['d_y'] = train_test['date'].dt.year
    # cols_knn = ['d_w', 'd_m', 'd_y']

    # 增加周末特征
    # train_train['weekend'] = 0
    # for i in range(len(train_train)):
    #     if train_train['d_w'][i] in [0,6]:
    #         train_train['weekend'][i] =  1
    # print(train_train)
    #
    # train_test['weekend'] = 0
    # for i in range(len(train_test)):
    #     if train_test['d_w'][i] in [0,6]:
    #         train_test['weekend'][i] =  1

    # 增加节假日特征1  效果不大
    train_train['holidays'] = 0
    for i in range(len(train_train)):
        if train_train['date'][i] in holidays.US():
            train_train['holidays'][i] =  1

    train_test['holidays'] = 0
    for i in range(len(train_test)):
        if train_test['date'][i] in holidays.US():
            train_test['holidays'][i] =  1
    # cols_knn = ['weekend', 'd_w', 'd_m', 'd_y','holidays']


    # 增加节假日特征2
    # D = pd.date_range('2012-11-1', periods=366,freq='d')  # 366 天
        # train_train['holidays'] = 0
    for i in range(len(train_train)-1):
        q_diff__down_day = (train_train['questions'][i] - train_train['questions'][i + 1]) / train_train['questions'][i + 1]
        q_diff__up_day = (train_train['questions'][i+1] - train_train['questions'][i]) / train_train['questions'][i]
        if q_diff__down_day > 0.6 :
            train_train['holidays'][i + 1] = str(train_train['date'].dt.month[i+1]) +'-'+ str(train_train['date'].dt.day[i+1])
        if q_diff__up_day > 0.8 :
            train_train['holidays'][i] = str(train_train['date'].dt.month[i]) +'-'+ str(train_train['date'].dt.day[i])

    # train_train['holidays'] = train_train['date'].dt.month + train_train['date'].dt.day
    # train_train['holidays'] = train_train['date'].dt.day
    print(train_train['holidays'].head(100))



    cols_knn = [ 'd_w', 'd_m', 'd_y']

    # 构造非线性特征
    cols_lr = ['id', 'sqrt_id']
    train_train['sqrt_id'] = np.sqrt(train_train['id'])
    train_test['sqrt_id'] = np.sqrt(train_test['id'])

    # 根据特征['id', 'sqrt_id']，构造线性模型预测questions
    reg = LinearRegression()
    reg.fit(train_train[cols_lr], train_train['questions'])
    q_fit = reg.predict(train_train[cols_lr])
    q_pred = reg.predict(train_test[cols_lr])

    # 根据特征['id', 'sqrt_id']，构造线性模型预测answers
    reg = LinearRegression()
    reg.fit(train_train[cols_lr], train_train['answers'])
    a_fit = reg.predict(train_train[cols_lr])
    a_pred = reg.predict(train_test[cols_lr])

    # 得到questions和answers的训练误差
    q_diff = train_train['questions'] - q_fit
    a_diff = train_train['answers'] - a_fit


    # 把训练误差作为新的目标值，使用特征cols_knn，建立kNN模型
    from sklearn.neighbors import KNeighborsRegressor
    reg = KNeighborsRegressor(n_neighbors=4, algorithm= 'brute')
    reg.fit(train_train[cols_knn], q_diff)
    q_pred_knn = reg.predict(train_test[cols_knn])

    reg = KNeighborsRegressor(n_neighbors=4, algorithm= 'brute')
    reg.fit(train_train[cols_knn], a_diff)
    a_pred_knn = reg.predict(train_test[cols_knn])

    Q_pred = q_pred + q_pred_knn
    A_pred = a_pred + a_pred_knn

    # 计算 MAPE
    print('Q的MAPE： ',acc_MAPE(Q_test,Q_pred))
    print('A的MAPE： ', acc_MAPE(A_test, A_pred))
    print('MAPE:', (acc_MAPE(Q_test,Q_pred)+ acc_MAPE(A_test, A_pred))/2)

if __name__ == '__main__':
    run_main()


