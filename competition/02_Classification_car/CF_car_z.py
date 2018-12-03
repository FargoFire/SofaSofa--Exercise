# -*- coding : utf-8 -*-
"""
    作者:     Fargo
    版本:     1.0
    日期:     2018/09/25

    任务类型：二元分类

    交通事故理赔审核:
    在交通摩擦（事故）发生后，理赔员会前往现场勘察、采集信息，这些信息往往影响着车主是否能够得到保险公司的理赔。
    训练集数据包括理赔人员在现场对该事故方采集的36条信息，信息已经被编码，以及该事故方最终是否获得理赔。
    我们的任务是根据这36条信息预测该事故方没有被理赔的概率。

    变量名	解释
    CaseId	案例编号，没有实际意义
    Q1	理赔员现场勘察采集的信息，Q1代表第一个问题的信息。信息被编码成数字，数字的大小不代表真实的关系。
    Qk	同上，Qk代表第k个问题的信息。一共36个问题。
    Evaluation	表示最终审核结果。0表示授予理赔，1表示未通过理赔审核。在test.csv中，这是需要被预测的标签。

    评价方法为精度-召回曲线下面积(Precision-Recall AUC)，以下简称PR-AUC
    PR-AUC的取值范围是0到1。越接近1，说明模型预测的结果越接近真实结果
"""

# -*- coding: utf-8 -*-
import math

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# 读取数据
train = pd.read_csv("./data_car/train.csv")
test = pd.read_csv("./data_car/test.csv")
submit = pd.read_csv("./data_car/sample_submit.csv")




def run_main():
    """
        主函数
    """
    # 删除id
    train.drop('CaseId', axis=1, inplace=True)
    test.drop('CaseId', axis=1, inplace=True)

    # 取出训练集的y
    y_train = train.pop('Evaluation')

    # 建立随机森林模型
    clf = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=10)
    clf.fit(train, y_train)
    y_pred = clf.predict_proba(test)[:, 1]

    # 输出预测结果至my_RF_prediction.csv
    submit['Evaluation'] = y_pred
    submit.to_csv('Fargo_RF_CAR_prediction.csv', index=False)



if __name__ == '__main__':
    run_main()

