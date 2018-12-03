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

import holidays
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression






def run_main():
    """
        主函数
    """
    # 读取数据
    train = pd.read_csv('./data_Q & A./train.csv')
    test = pd.read_csv('./data_Q & A./test.csv')
    submit = pd.read_csv('./data_Q & A./sample_submit.csv')


    # 取出真实值 Q 和 A
    Q_train = train.pop('questions')
    A_train = train.pop('answers')


    # 把date转为时间格式，得到星期，再进行独热处理
    train['date'] = pd.to_datetime(train['date'])
    train['dayofweek'] = train['date'].dt.dayofweek
    train = pd.get_dummies(train, columns=['dayofweek'])

    test['date'] = pd.to_datetime(test['date'])
    test['dayofweek'] = test['date'].dt.dayofweek
    test = pd.get_dummies(test, columns=['dayofweek'])

    # 插入id与星期的交叉相，一共得到7项
    for i in range(7):
        train['id_dayofweek_%s' % i] = train['id'] * train['dayofweek_%s' % i]
        test['id_dayofweek_%s' % i] = test['id'] * test['dayofweek_%s' % i]
    # print(train_train.head(50))

    # 增加周末特征  引入后没有提升
    train['weekend'] = (train['id_dayofweek_0'] + train['id_dayofweek_6'])/ \
                             (train['id_dayofweek_0'] + train['id_dayofweek_6'])
    train = train.fillna(value=0)

    test['weekend'] = (test['id_dayofweek_0'] + test['id_dayofweek_6'])/ \
                             (test['id_dayofweek_0'] + test['id_dayofweek_6'])
    test = test.fillna(value=0)

    train['id_weekend' ] = train['id'] * train['weekend']
    test['id_weekend'] = test['id'] * test['weekend']


    # print(train_train['date'])

    # 增加节假日特征
    train['holidays'] = 0
    for i in range(len(train)):
        if train['date'][i] in holidays.US():
            train['holidays'][i] =  1
    train['id_holidays'] = train['id'] * train['holidays']
    # print(train_train.head(50))

    test['holidays'] = 0
    for i in range(len(test)):
        if test['date'][i] in holidays.US():
            test['holidays'][i] =  1
    test['id_holidays'] = test['id'] * test['holidays']
            # print(train_test['date'][i])


    # 去除 不需要的列
    x = ['date','id']
    train.drop(x,axis=1,inplace=True)
    test.drop(x, axis=1, inplace=True)
    print(train.head(50))

    # 预测Q
    reg = LinearRegression(n_jobs=10)
    reg.fit(train, Q_train)
    Q_pred = reg.predict(test)

    # 预测A
    reg = LinearRegression()
    reg.fit(train,A_train)
    A_pred = reg.predict(test)

    # 输出预测结果至my_LR_prediction.csv
    submit['questions'] = Q_pred
    submit['answers'] = A_pred
    submit.to_csv('Fargo_LR_prediction.csv', index=False)

if __name__ == '__main__':
    run_main()


