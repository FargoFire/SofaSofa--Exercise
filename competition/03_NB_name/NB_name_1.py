# -*- coding: utf-8 -*-

"""
   赛题： http://sofasofa.io/competition.php?id=3
   利用朴素贝叶斯对“机器读中文：根据名字判断性别”中的数据进行预测
   训练集中共有120000条样本，预测集中有32040条样本。
"""
import nltk
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import math
import numpy as np


def split_train_test(name_df, size=0.8):
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


def get_word_list_from_data(name_df):
    """
        将数据集中的词放入一个列表
    """
    word_list = []
    for _ , r_data in name_df.iterrows():
        word_list += r_data['name']
    return word_list


def extract_feat_from_data(name_df, name_collection, common_words_freqs):
    """
        特征提取
    """
    # 这里只选择TF-IDF特征作为例子
    # 可考虑使用词频或其他文本特征作为额外的特征

    n_sample = name_df.shape[0]
    n_feat = len(common_words_freqs)
    common_words = [word for word, _ in common_words_freqs]

    # 初始化
    X = np.zeros([n_sample, n_feat])
    y = np.zeros(n_sample)

    print('提取特征...')
    for i, r_data in name_df.iterrows():
        if (i + 1) % 5000 == 0:
            print('已完成{}个样本的特征提取'.format(i + 1))

        name = r_data['name']

        feat_vec = []
        for word in common_words:
            if word in name:
                # 如果在高频词中，计算TF-IDF值
                tf_idf_val = name_collection.tf_idf(word, name)
            else:
                tf_idf_val = 0

            feat_vec.append(tf_idf_val)

        # 赋值
        X[i, :] = np.array(feat_vec)
        y[i] = int(r_data['gender'])

    return X ,y


def cal_acc(true_labels, pred_labels):
    """
        计算准确率
    """
    n_total = len(true_labels)
    print(true_labels)
    correct_list = [true_labels[i] == pred_labels[i] for i in range(n_total)]

    acc = sum(correct_list) / n_total
    return acc


def run_main():
    """
        主函数
    """
    # 1. 读取训练集，验证集
    train_df_data = pd.read_csv('./train.txt')
    # test_df = pd.read_csv('./test.txt')

    print('加载数据...')
    train_df_data_g = train_df_data[ train_df_data['gender'] == 0 ]
    print('女',train_df_data_g.shape[0])
    train_df_data_b = train_df_data[train_df_data['gender'] == 1]
    print('男', train_df_data_b.shape[0])


    # 2. 分割训练集 测试集
    train_df_data = train_df_data.drop(['id'], axis=1)
    name_df_train , name_df_test = split_train_test(train_df_data)

    # 查看训练集测试集基本信息
    print('train: ',name_df_train.shape[0])
    # print('test:' , name_df_test.shape[0])
    # print('test:' , name_df_test)
    print('训练集中各类的数据个数：', name_df_train.groupby('gender').size())
    print('测试集中各类的数据个数：', name_df_test.groupby('gender').size())


    # 3. 特征提取
    # 计算词频
    n_common_words= 1500

    # 将训练集中的单词拿出来统计词频
    print('统计词频...')
    all_words_in_train = get_word_list_from_data(name_df_train)  # 将所有单词集合在一个list中
    fdist = nltk.FreqDist(all_words_in_train)                    #
    common_words_freqs = fdist.most_common(n_common_words)
    # print('出现最多的{}个词是：'.format(n_common_words))
    # for word, count in common_words_freqs:
    #     print('{}: {}次'.format(word, count))
    # print()

    # 在训练集上提取特征
    name_collection = nltk.TextCollection(name_df_train['name'].values.tolist())
    print('训练样本提取特征...', end=' \n', )
    train_X, train_y = extract_feat_from_data(name_df_train, name_collection, common_words_freqs)
    print('完成')
    print()

    print('测试样本提取特征...', end=' ')
    test_X, test_y = extract_feat_from_data(name_df_test, name_collection, common_words_freqs)
    print('完成')

    # 4. 训练模型Naive Bayes
    print('训练模型...', end=' ')
    gnb = GaussianNB()
    gnb.fit(train_X, train_y)
    print('完成')
    print()

    # 5. 预测
    print('测试模型...', end=' ')
    test_pred = gnb.predict(test_X)
    print('完成')

    # 输出准确率
    print('准确率：', cal_acc(test_y, test_pred))



if  __name__ == '__main__':
    run_main()










