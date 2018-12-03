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
from xgboost import XGBRegressor
import pandas as pd

# 读取数据
train = pd.read_csv("./data_bike/train.csv")
test = pd.read_csv("./data_bike/test.csv")
submit = pd.read_csv("./data_bike/sample_submit.csv")


def run_main():
    # 删除id
    train.drop('id', axis=1, inplace=True)
    test.drop('id', axis=1, inplace=True)

    train['time'] = train['hour'] // 4
    test['time'] = test['hour'] // 4

    # 取出训练集的y
    y_train = train.pop('y')

    # 建立一个默认的xgboost回归模型
    reg = XGBRegressor(max_depth=6, min_child_weight=3,subsample=0.83)
    reg.fit(train, y_train)
    y_pred = reg.predict(test)

    # 输出预测结果至my_XGB_prediction.csv
    submit['y'] = y_pred
    submit.to_csv('Fargo_bike_prediction.csv', index=False)


if __name__ == '__main__':
    run_main()