# -*- coding : utf-8 -*-

import pandas as pd
import numpy as np
import math
from nltk import FreqDist
from sklearn.ensemble import GradientBoostingClassifier     # 0.83
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression         #  0.6
from sklearn.naive_bayes import GaussianNB                  #  0.8


# 读取训练集 测试集 验证集
data = pd.read_csv('./train.txt')

def split_train_test(name_df, size=0.7):
    """
        分割训练集和测试集
    """
    # 为保证每个类中的数据能在训练集中和测试集中的比例相同，所以需要依次对每个类进行处理
    name_df_train = pd.DataFrame()
    name_df_test = pd.DataFrame()

    genders =[0,1]
    for gender in genders:
        # 找出gender的记录
        name_df_w_gender = name_df[name_df['gender'] == gender]
        # 重新设置索引，保证每个类的记录是从0开始索引，方便之后的拆分
        name_df_w_gender = name_df_w_gender.reset_index()

        # 默认 80%训练集 20%测试集 分割
        n_lines = name_df_w_gender.shape[0]   # 数据集行数
        split_line_on = math.floor(n_lines * size)  # floor 向下取整
        name_df_w_gender_train = name_df_w_gender.iloc[ : split_line_on, : ]
        name_df_w_gender_test = name_df_w_gender.iloc[  split_line_on : , : ]

        name_df_train = name_df_train.append(name_df_w_gender_train)
        name_df_test = name_df_test.append(name_df_w_gender_test)

    name_df_train = name_df_train.reset_index()
    name_df_test = name_df_test.reset_index()
    return name_df_train,name_df_test


def get_word_TF(name_ci,row,boy_words_freqs,train_boy_num,girl_words_freqs,train_girl_num):
    """
        得到训练集  测试集中每个人的每个字的词频（Term Frequency，通常简称TF）
    """
    for j in range(len(name_ci)):
        # if  name_ci[j] in boy_words_freqs.index:
        try:
            boy_ci = boy_words_freqs.ix[name_ci[j]] * 100. / train_boy_num
            row[2 * j] = boy_ci['num']
        except:
            pass

        # if  name_ci[j] in girl_words_freqs.index:
        try:
            girl_ci = girl_words_freqs.ix[name_ci[j]] * 100. / train_girl_num
            row[2 * j + 1] = girl_ci['num']
        except:
            pass

def cal_acc(predicts,test_data):

    pred_gender = np.array(predicts)
    n_total = len(test_data)
    # 判断预测是否正确
    correct_list = [test_data['gender'][i] == pred_gender[i] for i in range(n_total)]
    print(sum(correct_list))
    acc = sum(correct_list) / n_total
    return acc


def run_main():

    # 1. 整理数据
    print('整理数据...')
    train_data,test_data = split_train_test(data)
    # 男生的所有名字
    train_boy = train_data[ train_data[ 'gender' ] == 1 ]
    train_boy_num = len(train_boy)
    train_boy_names = ''.join(train_boy['name'])

    # 女的所有名字
    train_girl = train_data[ train_data[ 'gender' ] == 0 ]
    train_girl_num = len(train_girl)
    train_girl_names = ''.join(train_girl['name'])

    # 计算词频
    print('计算词频...')
    # 男生每个字的次数
    fdist_boy = FreqDist(train_boy_names)
    boy_words_freqs = pd.DataFrame({  'num' : [fdist_boy[key] for key in fdist_boy.keys() ] },
                                     index = [key for key in fdist_boy.keys()])

    # 女生每个字的次数
    fdist_girl = FreqDist(train_girl_names)
    girl_words_freqs = pd.DataFrame({  'num' : [fdist_girl[key] for key in fdist_girl.keys() ] },
                                      index = [key for key in fdist_girl.keys()])


    # 得到训练集中每个人的每个字的词频（Term Frequency，通常简称TF）
    print('计算训练集 每个字的 词频...')
    train_TF = []
    for i in range(len(train_data)):
        name = train_data.at[i,'name']
        name_ci= []
        for n in range(len(name)):
            name_ci.append(name[n])
        row = [0., 0., 0., 0., train_data.at[i, 'gender']]
        a ,b,c,d =boy_words_freqs, train_boy_num, girl_words_freqs, train_girl_num
        get_word_TF(name_ci,row,a ,b,c,d)
        train_TF.append(row)
    # print(train_TF)


    # 得到测试集中每个人的每个字的词频（Term Frequency，通常简称TF）
    print('计算测试集 每个字的 词频...')
    test_TF = []
    for i in range(len(test_data)):
        name = test_data.at[i,'name']
        name_ci= []
        for n in range(len(name)):
            name_ci.append(name[n])
        row = [0., 0., 0., 0.]
        a ,b,c,d =boy_words_freqs, train_boy_num, girl_words_freqs, train_girl_num
        get_word_TF(name_ci,row,a ,b,c,d)
        test_TF.append(row)

    # 转换为pandas.DataFrame的形式
    # 1_f是指这个人的第一个字在训练集中所有女生的字中出现的频率
    # 2_f是指这个人的第二个字在训练集中所有女生的字中出现的频率
    # 1_m是指这个人的第一个字在训练集中所有男生的字中出现的频率
    # 2_m是指这个人的第二个字在训练集中所有男生的字中出现的频率
    print('转换为pandas.DataFrame的形式...')
    train_TF = pd.DataFrame(train_TF, columns=['1_f', '1_m', '2_f', '2_m', 'gender'])
    test_TF = pd.DataFrame(test_TF, columns=['1_f', '1_m', '2_f', '2_m'])


    # 训练GBDT模型
    print('训练GBDT模型...')
    clf = GradientBoostingClassifier(n_estimators=320,learning_rate=0.2)
    # clf = GaussianNB()
    # clf = LogisticRegression()
    # clf = RandomForestClassifier()

    # print(train_TF.drop(['gender'], axis=1))
    clf.fit(train_TF.drop(['gender'], axis=1), train_TF['gender'])
    predicts = clf.predict(test_TF)

    # 计算准确率
    print('计算准确率...')
    print('准确率：', cal_acc(predicts,test_data))

if __name__ == '__main__':
    run_main()
