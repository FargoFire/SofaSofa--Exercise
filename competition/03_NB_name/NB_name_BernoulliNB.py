# -*- coding : utf-8 -*-

import pandas as pd
import numpy as np
from nltk import FreqDist
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB


# 读取训练集 测试集 验证集
train_data = pd.read_csv('./train.txt')
test_data = pd.read_csv('./test.txt')
submit = pd.read_csv('./sample_submit.csv')

def get_word_TF(name_ci,row,boy_words_freqs,train_boy_num,girl_words_freqs,train_girl_num):
    """
        得到训练集  测试集中每个人的每个字的词频（Term Frequency，通常简称TF）
    """
    for j in range(len(name_ci)):
        try:
            boy_ci = boy_words_freqs.ix[name_ci[j]] * 100. / train_boy_num
            row[2 * j] = boy_ci['num']
        except:
            pass

        try:
            girl_ci = girl_words_freqs.ix[name_ci[j]] * 100. / train_girl_num
            row[2 * j + 1] = girl_ci['num']
        except:
            pass


def run_main():

    # 1. 整理数据

    # 男生的所有名字
    train_boy = train_data[ train_data[ 'gender' ] == 1 ]
    train_boy_num = len(train_boy)
    train_boy_names = ''.join(train_boy['name'])

    # 女的所有名字
    train_girl = train_data[ train_data[ 'gender' ] == 0 ]
    train_girl_num = len(train_girl)
    train_girl_names = ''.join(train_girl['name'])

    # 计算词频
    # 男生每个字的次数
    fdist_boy = FreqDist(train_boy_names)
    boy_words_freqs = pd.DataFrame({  'num' : [fdist_boy[key] for key in fdist_boy.keys() ] },
                                     index = [key for key in fdist_boy.keys()])

    # 女生每个字的次数
    fdist_girl = FreqDist(train_girl_names)
    girl_words_freqs = pd.DataFrame({  'num' : [fdist_girl[key] for key in fdist_girl.keys() ] },
                                      index = [key for key in fdist_girl.keys()])


    # 得到训练集中每个人的每个字的词频（Term Frequency，通常简称TF）
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


    # 得到测试集中每个人的每个字的词频（Term Frequency，通常简称TF）
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
    train_TF = pd.DataFrame(train_TF, columns=['1_f', '1_m', '2_f', '2_m', 'gender'])
    test_TF = pd.DataFrame(test_TF, columns=['1_f', '1_m', '2_f', '2_m'])


    # 训练GBDT模型
    clf = BernoulliNB()
    clf.fit(train_TF.drop(['gender'], axis=1) , train_TF['gender'])
    predicts = clf.predict(test_TF)

    # 输出预测结果至fargo_name_predict_BernoulliNB.csv
    submit['gender'] = np.array(predicts)
    submit.to_csv('./fargo_name_predict_BernoulliNB.csv', index=False)



if __name__ == '__main__':
    run_main()


# 准确率： 0.60162  2018.09。21