# -*- coding : utf-8 -*-
"""
    作者:     Fargo
    版本:     1.0
    日期:     2018/10/01

    任务类型：二元分类、图像识别

    交通事故理赔审核:
    我们给出四千张图像作为训练集。每个图像中只有一个图形，要么是圆形，要么是正方形。
    根据这四千张图片训练出一个二元分类模型，并用它（不是用肉眼）在测试集上判断每个图像中的形状。
    训练集中共有4000个灰度图像，预测集中有3550个灰度图像。每个图像中都会含有大量的噪点。
    图像的分辨率为40x40，也就是40x40的矩阵，每个矩阵以行向量的形式被存放在train.csv和test.csv中。
    train.csv和test.csv中每行数据代表一个图像，也就是说每行都有1600个特征

    变量名	解释
    id	编号
    p_i_j	表示图像中第i行第j列上的像素点的灰度值，取值范围在0到255之间，i和j的取值都是0到39。
    y	表示该图像中的形状。0表示圆形，1表示方形。这是需要被预测的标签。

    评价方法为F1 score。F1 score的取值范围是0到1。越接近1，说明模型预测的结果越佳
"""

import math
import pandas as pd
import numpy as np
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def split_train_test(train_data,size=0.8):
    print('分割 训练集 测试集...')

    # 默认80%为训练集
    n_lines = train_data.shape[0]
    split_line_on = math.floor(n_lines * size)
    train_train = train_data.iloc[:split_line_on, : ]
    train_test = train_data.iloc[split_line_on:, : ]

    train_train = train_train.reset_index()
    train_test = train_test.reset_index()
    return train_train, train_test


def data_modify_suitable_train(data_set=None, type=True):
    pass


def load_train_test_data(train,test):
    #
    np.random.shuffle(train)  # 打乱顺序，多维矩阵中，只对第一维（行）做打乱顺序操作
    y_label = train[ : , -1]   # 'y'的数据

    data, data_test = data_modify_suitable_train(train,True), data_modify_suitable_train(test,False)
    train_x, test_x, train_y, test_y = train_test_split(data, y_label, test_size=0.7)
    return train_x, train_y, test_x, test_y, data_test


def run_main():
    """
        主程序
    """
    # 读取数据，分割训练集 测试集
    train_data = pd.read_csv('./data_Square_or_circle/train.csv')
    train , test = split_train_test(train_data)

    # 取出 y 去除id
    y_test = test.pop('y')
    train = np.array(train.drop('id', axis=1))
    test = np.array(test.drop('id', axis=1))

    print(train)

    # 将训练集 随机划分训练集和测试集
    train_x, train_y, test_x, test_y, data_set = load_train_test_data(train,test)









    # 计算F1 score
    F1 = f1_score()
    print('F1 score:' ,F1)


if __name__ == '__main__':
    run_main()

