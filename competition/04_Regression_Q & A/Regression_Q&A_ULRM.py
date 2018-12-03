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
from pandas.tseries import holiday
from sklearn.linear_model import LinearRegression



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


    # 取出真实值 Q 和 A
    Q_train = train_train.pop('questions')
    A_train = train_train.pop('answers')

    Q_test = train_test.pop('questions')
    A_test = train_test.pop('answers')

    # 把date转为时间格式，得到星期，再进行独热处理
    train_train['date'] = pd.to_datetime(train_train['date'])
    train_train['dayofweek'] = train_train['date'].dt.dayofweek
    train_train = pd.get_dummies(train_train, columns=['dayofweek'])

    train_test['date'] = pd.to_datetime(train_test['date'])
    train_test['dayofweek'] = train_test['date'].dt.dayofweek
    train_test = pd.get_dummies(train_test, columns=['dayofweek']) # get_dummies 将类别变量转换成新增的虚拟变量/指示变量

    # 插入id与星期的交叉相，一共得到7项
    for i in range(7):
        train_train['id_dayofweek_%s' % i] = train_train['id'] * train_train['dayofweek_%s' % i]
        train_test['id_dayofweek_%s' % i] = train_test['id'] * train_test['dayofweek_%s' % i]
    # print(train_train.head(50))

    # 增加周末特征  引入后没有提升
    train_train['weekend'] = (train_train['id_dayofweek_0'] + train_train['id_dayofweek_6'])/ \
                             (train_train['id_dayofweek_0'] + train_train['id_dayofweek_6'])
    train_train = train_train.fillna(value=0)

    train_test['weekend'] = (train_test['id_dayofweek_0'] + train_test['id_dayofweek_6'])/ \
                             (train_test['id_dayofweek_0'] + train_test['id_dayofweek_6'])
    train_test = train_test.fillna(value=0)

    train_train['id_weekend' ] = train_train['id'] * train_train['weekend']
    train_test['id_weekend'] = train_test['id'] * train_test['weekend']


    # print(train_train['date'])

    # 增加节假日特征
    train_train['holidays'] = 0
    for i in range(len(train_train)):
        if train_train['date'][i] in holidays.US():
            train_train['holidays'][i] =  1
    train_train['id_holidays'] = train_train['id'] * train_train['holidays']
    # print(train_train.head(50))

    train_test['holidays'] = 0
    for i in range(len(train_test)):
        if train_test['date'][i] in holidays.US():
            train_test['holidays'][i] =  1
    train_test['id_holidays'] = train_test['id'] * train_test['holidays']
            # print(train_test['date'][i])


    # 去除 不需要的列
    x = ['date','id', 'index']
    train_train.drop(x,axis=1,inplace=True)
    train_test.drop(x, axis=1, inplace=True)
    print(train_train.head(50))

    # 预测Q
    reg = LinearRegression(n_jobs=10)
    reg.fit(train_train, Q_train)
    Q_pred = reg.predict(train_test)

    # 预测A
    reg = LinearRegression()
    reg.fit(train_train,A_train)
    A_pred = reg.predict(train_test)

    # 计算 MAPE
    print('Q的MAPE： ',acc_MAPE(Q_test,Q_pred))
    print('A的MAPE： ', acc_MAPE(A_test, A_pred))
    print('MAPE:', (acc_MAPE(Q_test,Q_pred)+ acc_MAPE(A_test, A_pred))/2)

if __name__ == '__main__':
    run_main()


