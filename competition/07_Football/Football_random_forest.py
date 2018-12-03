# -*- coding : utf-8 -*-
"""
    作者：Fargo
    版本：1.0
    日期：2018.10.20

    任务类型： 回归

    足球运动员身价估计:
    每个足球运动员在转会市场都有各自的价码。本次数据练习的目的是根据球员的各项信息和能力值来预测该球员的市场价值。

    训练集中共有10441条样本，预测集中有7000条样本。每条样本代表一位球员，数据中每个球员有63项属性。数据中含有缺失值。
    变量名	解释
    id	行编号，没有实际意义
    club	该球员所属的俱乐部。该信息已经被编码。
    league	该球员所在的联赛。已被编码。
    birth_date	生日。格式为月/日/年。
    height_cm	身高（厘米）
    weight_kg	体重（公斤）
    nationality	国籍。已被编码。

    potential	球员的潜力。数值变量。

    pac	球员速度。数值变量。
    sho	射门（能力值）。数值变量。
    pas	传球（能力值）。数值变量。
    dri	带球（能力值）。数值变量。
    def	防守（能力值）。数值变量。
    phy	身体对抗（能力值）。数值变量。

    international_reputation	国际知名度。数值变量。

    skill_moves	技巧动作。数值变量。
    weak_foot	非惯用脚的能力值。数值变量。
    work_rate_att	球员进攻的倾向。分类变量，Low, Medium, High。
    work_rate_def	球员防守的倾向。分类变量，Low, Medium, High。
    preferred_foot	惯用脚。1表示右脚、2表示左脚。

    crossing	传中（能力值）。数值变量。
    finishing	完成射门（能力值）。数值变量。
    heading_accuracy	头球精度（能力值）。数值变量。
    short_passing	短传（能力值）。数值变量。
    volleys	凌空球（能力值）。数值变量。
    dribbling	盘带（能力值）。数值变量。
    curve	弧线（能力值）。数值变量。
    free_kick_accuracy	定位球精度（能力值）。数值变量。
    long_passing	长传（能力值）。数值变量。
    ball_control	控球（能力值）。数值变量。
    acceleration	加速度（能力值）。数值变量。
    sprint_speed	冲刺速度（能力值）。数值变量。
    agility	灵活性（能力值）。数值变量。
    reactions	反应（能力值）。数值变量。
    balance	身体协调（能力值）。数值变量。
    shot_power	射门力量（能力值）。数值变量。
    jumping	弹跳（能力值）。数值变量。
    stamina	体能（能力值）。数值变量。
    strength	力量（能力值）。数值变量。
    long_shots	远射（能力值）。数值变量。
    aggression	侵略性（能力值）。数值变量。
    interceptions	拦截（能力值）。数值变量。
    positioning	位置感（能力值）。数值变量。
    vision	视野（能力值）。数值变量。
    penalties	罚点球（能力值）。数值变量。
    marking	卡位（能力值）。数值变量。
    standing_tackle	断球（能力值）。数值变量。
    sliding_tackle	铲球（能力值）。数值变量。

    st	球员在射手位置的能力值。数值变量。 中锋
    cf	球员在锋线位置的能力值。数值变量。 中前峰
    rw	球员在右边锋位置的能力值。数值变量。
    lw	球员在左边锋位置的能力值。数值变量。

    cam	球员在前腰位置的能力值。数值变量。  攻击型中场
    cm	球员在中场位置的能力值。数值变量。
    cdm	球员在后腰位置的能力值。数值变量。

    rb	球员在右后卫位置的能力值。数值变量。
    cb	球员在中后卫的能力值。数值变量。
    lb	球员在左后卫置的能力值。数值变量。

    gk_diving	门将扑救（能力值）。数值变量。
    gk_handling	门将控球（能力值）。数值变量。
    gk_kicking	门将开球（能力值）。数值变量。
    gk_positioning	门将位置感（能力值）。数值变量。
    gk_reflexes	门将反应（能力值）。数值变量。

    gk	球员在守门员的能力值。数值变量。

    y	该球员的市场价值（单位为万欧元）。这是要被预测的数值。

    评价标准为MAE(Mean Absolute Error)。
    若真实值为y=(y1,y2,⋯,yn)，模型的预测值为y^=(y^1,y^2,⋯,y^n)，那么该模型的MAE计算公式为
    MAE=∑ni=1|yi−y^i| / n.
    MAE越小，说明模型预测得越准确。
"""
from datetime import date, datetime

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def data_train_test__split(train_data, test_data, cols_all):
    """
        分割训练集 和 测试集
    """
    # np.random.shuffle(train_data)
    labels_y = train_data[ : , -1]
    labels_x = train_data[ : , 0:train_data.shape[1] - 1]

    train_x, test_x, train_y, test_y = train_test_split(labels_x, labels_y, test_size=0.3)

    train_x = pd.DataFrame(train_x, columns=cols_all)
    test_x = pd.DataFrame(test_x, columns=cols_all)
    train_y = pd.DataFrame(train_y, columns=['y'])
    test_y = pd.DataFrame(test_y, columns=['y'])

    return train_x, test_x, train_y,  test_y


def RF_Regressor(train_x, test_x, cols_need, cols_position,TF=True):
    reg = RandomForestRegressor(random_state=150)
    reg.fit(train_x[train_x[cols_position] == TF][cols_need], train_x[train_x[cols_position] == TF]['y'])

    preds = reg.predict(test_x[test_x[cols_position] == TF][cols_need])
    return preds


def CAL_MAE(test_y, pred_y):
    sum = 0
    n_total = len(test_y)
    test_y = np.array(test_y)
    pred_y = np.array(pred_y)

    for i in range(1,n_total):
        t = abs( test_y[i] - pred_y[i] )
        sum += t

    MAE_s = sum / n_total
    return MAE_s

def run_main():
    """
        主函数
    """
    train_data = pd.read_csv('./data_football/train.csv')
    test_data = pd.read_csv('./data_football/test.csv')

    # 1 处理数据
    # 1.1 填补缺失值
    train_data = train_data.fillna(0)
    test_data = test_data.fillna(0)

    # 1.2 去除'id'
    train_data.drop('id', axis=1, inplace=True)
    test_data.drop('id', axis=1, inplace=True)

    # 1.3 特征工程
    # 1.3.1 获取年龄
    print('获取年龄...')
    today = datetime(2018, 1, 1)

    train_data['birth_date'] = pd.to_datetime(train_data['birth_date'])
    test_data['birth_date'] = pd.to_datetime(test_data['birth_date'])

    train_data['age'] = (today - train_data['birth_date']).apply(lambda x: x.days) / 365
    test_data['age'] = (today - test_data['birth_date']).apply(lambda x: x.days) / 365


    # 1.3.2 获取球员最擅长位置的评分
    print('获取球员最擅长位置的评分...')
    positions = ['rw', 'rb', 'st', 'lw', 'cf', 'cam', 'cm', 'cdm', 'cb', 'lb', 'gk']

    train_data['best_pos'] = train_data[positions].max(axis=1)
    test_data['best_pos'] = test_data[positions].max(axis=1)


    # 1.3.3 球员身体质量（BMI）
    print('球员身体质量BMI...')
    train_data['BMI'] =10000. * train_data['weight_kg'] / (train_data['height_cm']**2)
    test_data['BMI'] = 10000. * test_data['weight_kg'] / (test_data['height_cm'] ** 2)


    # 1.3.4 是否为门将
    print('门将...')
    train_data['is_gk'] = train_data['gk'] > 0
    test_data['is_gk'] = test_data['gk'] > 0

    gk_score = ['gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning', 'gk_reflexes']
    train_data['gk_score'] = train_data[gk_score].mean(axis=1)
    test_data['gk_score'] = test_data[gk_score].mean(axis=1)


    # 1.3.5 是否为 中后卫
    print('中后卫...')
    positions1 = ['rw', 'st', 'lw', 'cf', 'cam', 'cm']
    positions2 = [ 'rb', 'cb', 'lb']

    train_data['backfield'] =  train_data[positions2].mean(axis=1) - train_data[positions1].mean(axis=1)
    train_data['is_backfield'] = train_data[train_data['is_gk'] == False]['backfield'] > 2

    test_data['backfield'] = test_data[positions2].mean(axis=1) - test_data[positions1].mean(axis=1)
    test_data['is_backfield'] = test_data[test_data['is_gk'] == False]['backfield'] > 2

    backfield_skill = ['standing_tackle', 'sliding_tackle','marking', 'interceptions', 'reactions', 'long_passing', 'short_passing']

    train_data['backfield_skill'] = train_data[backfield_skill].sum(axis=1)
    test_data['backfield_skill'] = test_data[backfield_skill].sum(axis=1)


    # 1.3.6 是否为中场
    print('中场...')

    positions1 = ['rw', 'st', 'lw', 'cf' ]
    positions2 = [ 'cam', 'cm', 'cdm']

    train_data['midfielder'] =  train_data[positions2].mean(axis=1) - train_data[positions1].mean(axis=1)
    train_data['is_midfielder'] = train_data[train_data['is_backfield'] == False]['midfielder'] > 1

    test_data['midfielder'] = test_data[positions2].mean(axis=1) - test_data[positions1].mean(axis=1)
    test_data['is_midfielder'] = test_data[test_data['is_backfield'] == False]['midfielder'] > 1

    midfield_skill_1 = [ 'marking', 'vision', 'interceptions', 'long_shots', 'stamina', 'reactions', 'agility', ]
    midfield_skill_2 = ['ball_control', 'long_passing', 'curve', 'short_passing', 'dribbling']

    train_data['midfield_skill_1'] = train_data[midfield_skill_1].sum(axis=1)/100
    test_data['midfield_skill_1'] = test_data[midfield_skill_1].sum(axis=1)/100
    train_data['midfield_skill_2'] = train_data[midfield_skill_2].sum(axis=1)/100
    test_data['midfield_skill_2'] = test_data[midfield_skill_2].sum(axis=1)/100


    # 1.3.7 是否为前锋
    positions1 = [ 'cam', 'cm', 'cdm', 'rb', 'cb', 'lb']
    positions2 = ['rw', 'st', 'lw', 'cf']

    train_data['striker'] =  train_data[positions2].mean(axis=1) - train_data[positions1].mean(axis=1)
    train_data['is_striker'] = train_data[train_data['is_midfielder'] == False]['striker'] > 1

    test_data['striker'] = test_data[positions2].mean(axis=1) - test_data[positions1].mean(axis=1)
    test_data['is_striker'] = test_data[test_data['is_midfielder'] == False]['striker'] > 1

    striker_kill = ['crossing', 'finishing', 'volleys', 'dribbling', 'acceleration', 'sprint_speed', 'shot_power', 'agility']

    train_data['striker_kill'] = train_data[striker_kill].sum(axis=1) / 100
    test_data['striker_kill'] = test_data[striker_kill].sum(axis=1) / 100

    # 1.3.  其他人
    train_data['other'] = train_data['is_gk'] + train_data['is_backfield'] + train_data['is_midfielder'] + train_data['is_striker']
    test_data['other'] = test_data['is_gk'] + test_data['is_backfield'] + test_data['is_midfielder'] + test_data['is_striker']


    #  2 数据清洗
    #  2.1 将y移至最后一列，获取所有列名
    print('数据清洗...')

    train_data_y = train_data.pop('y')
    cols_all = train_data.columns.tolist()
    train_data['y'] = train_data_y


    #  2.2转为 numpy.ndarray
    train_data = np.array(train_data)
    test_data = np. array(test_data)


    # 2.2 分割训练集 测试集
    print('分割训练集 测试集...')
    train_x, test_x, train_y, test_y  = data_train_test__split(train_data, test_data, cols_all)
    train_x['y'] =  train_y

    # 3 随机森林训练
    print('随机森林训练...')
    test_x['pred'] = 0

    # 3.1 守门员数据
    cols_gk = ['age', 'best_pos', 'BMI', 'gk_score' ,'height_cm', 'weight_kg', 'potential', 'BMI', 'pac',
        'phy', 'international_reputation',]
    preds = RF_Regressor(train_x, test_x, cols_gk, 'is_gk',True)
    test_x.loc[test_x['is_gk'] == True, 'pred'] = preds


    # 3.2 中后卫数据
    cols_mf = ['backfield_skill', 'age', 'best_pos', 'BMI' ,'height_cm', 'weight_kg', 'potential', 'BMI', 'pac',
        'phy', 'international_reputation' ]
    preds = RF_Regressor(train_x, test_x, cols_mf, 'is_backfield',True)
    test_x.loc[test_x['is_backfield'] == True, 'pred'] = preds


    # 3.3 中场数据
    cols_bf = ['midfield_skill_1', 'midfield_skill_2', 'age', 'best_pos', 'BMI' ,'height_cm', 'weight_kg', 'potential', 'BMI', 'pac',
        'phy', 'international_reputation' ]
    preds = RF_Regressor(train_x, test_x, cols_bf, 'is_midfielder',True)
    test_x.loc[test_x['is_midfielder'] == True, 'pred'] = preds


    # 3.3 中场数据
    cols_st = ['striker_kill', 'age', 'best_pos', 'BMI' ,'height_cm', 'weight_kg', 'potential', 'BMI', 'pac',
        'phy', 'international_reputation' ]
    preds = RF_Regressor(train_x, test_x, cols_st, 'is_striker',True)
    test_x.loc[test_x['is_striker'] == True, 'pred'] = preds


    # 3. 其他人
    cols_other = ['age', 'best_pos', 'BMI' ,'height_cm', 'weight_kg', 'potential', 'BMI', 'pac',
                  'phy', 'international_reputation']
    preds = RF_Regressor(train_x, test_x, cols_other, 'other', False)
    test_x.loc[test_x['other'] == False, 'pred'] = preds


    # 4 计算MAE
    print('计算MAE')
    pred_y = test_x['pred']

    S = CAL_MAE(test_y, pred_y)
    print('MAE值：', S)


if __name__ == '__main__':
    run_main()





