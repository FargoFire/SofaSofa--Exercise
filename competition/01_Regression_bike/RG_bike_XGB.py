# -*- coding : utf-8 -*-
"""
    作者:     Fargo
    版本:     1.0
    日期:     2018/09/22

    公共自行车使用量预测:
    两个城市某街道上的几处公共自行车停车桩。根据时间、天气等信息，预测出该街区在一小时内的被借取的公共自行车的数量。
    任务类型：回归  训练集中共有10000条样本，预测集中有7000条样本

    变量名	解释
    id	行编号，没有实际意义
    y	一小时内自行车被借取的数量。在test.csv中，这是需要被预测的数值。
    city	表示该行记录所发生的城市，一共两个城市
    hour	当时的时间，精确到小时，24小时计时法
    is_workday	1表示工作日，0表示节假日或者周末
    temp_1	当时的气温，单位为摄氏度
    temp_2	当时的体感温度，单位为摄氏度
    weather	当时的天气状况，1为晴朗，2为多云、阴天，3为轻度降水天气，4为强降水天气
    wind	当时的风速，数值越大表示风速越大

    评价方法为RMSE(Root of Mean Squared Error),RMSE越小，说明模型预测得越准确。
"""

# xgboost回归模型(Python)
# 该模型预测结果的RMSE为：18.947

# -*- coding: utf-8 -*-

# 引入模块
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import math


# 读取数据


train_data = pd.read_csv("./data_bike/train.csv")
# test = pd.read_csv("./data_bike/test.csv")
# submit = pd.read_csv("./data_bike/sample_submit.csv")


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


def acc_rmse(y_pred,y_test):
    """
        评价方式 为RMSE(Root of Mean Squared Error),RMSE越小，说明模型预测得越准确
    """
    print('计算 决定系数...')

    error = []
    for i in range(len(y_pred)):
        error.append(y_pred[i] - y_test[i])  # 预测值 与 实际值误差

    squaredError = []
    for val in error:
        squaredError.append(val * val)  # target-prediction之差平方

    RMSE = np.sqrt(sum(squaredError) / len(squaredError))
    return RMSE


def get_best_model():
    pass


def run_main():

    # 1.处理数据
    print('处理数据...')
    # 1.1 分割训练集 测试集
    train_train, train_test = split_train_test(train_data)

    # 1.2 删除id
    train_train.drop('id', axis=1, inplace=True)
    train_test_real = train_test.drop('id', axis=1)

    train_train['time'] = train_train['hour'] // 4
    train_test_real['time'] = train_test_real['hour'] // 4

    # train_train['time_work'] = ( train_train['hour'] // 4 + 1 ) * train_train[ 'is_workday']
    # train_test_real['time_work'] = ( train_test_real['hour'] // 4 + 1 ) * train_test_real[ 'is_workday']

    train_test_pred = train_test_real.drop('y', axis=1)



    # # 1.3 特征范围归一化
    # scaler = StandardScaler()
    # train_train_scaler = scaler.fit_transform(train_train)  # Fit to data, then transform it.
    # train_test_scaler = scaler.transform(train_test_real)   # Perform standardization by centering and scaling


    # 取出训练集的y
    print('取出训练集的y...')
    y_train = train_train.pop('y')
    y_test = train_test_real.pop('y')

    # 建立一个默认的xgboost回归模型
    # reg = GradientBoostingRegressor(max_depth=5)
    print('训练回归模型  xgboost ')

    # class dask_ml.xgboost.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True,
    # objective='reg:linear', booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1,
    # max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1,
    # scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None, ** kwargs)

    max_depths = [3, 4, 5, 6, 7, 8, 9]
    for max_depth in max_depths:

        reg = XGBRegressor(max_depth=max_depth, learning_rate=0.1, min_child_weight=3, n_estimators=100, subsample=0.83)
        reg = reg.fit(train_train, y_train)  # fit之后，在predict预测
        y_pred = reg.predict(train_test_pred)

        print(' max_depths={}'.format(max_depth))
        print('决定系数 :', acc_rmse(y_pred, y_test))

    # 输出预测结果至my_XGB_prediction.csv
    # submit['y'] = y_pred
    # submit.to_csv('my_XGB_prediction.csv', index=False)
    # print( reg.score(y_pred ,y_test) )


if __name__ == '__main__':
    run_main()

