import numpy as np
import pandas as pd

# submit = pd.read_csv('./sample_submit.csv')
# # submit_gender = submit['gender']
# #
# #
# #
# # # 计算准确率
# # preds = pd.read_csv('./my3_TF_GBDT_prediction.csv')
# # pred_gender = preds['gender']
# #
# # cha = submit_gender - pred_gender
# # print(cha)
# # n_total = len(submit)
# # # correct_list = [submit_gender[i] == pred_gender[i] for i in range(n_total)]
# # correct_list = 0
# # for i in range(n_total):
# #     if cha[i] == 0 :
# #         correct_list += 1
# #
# # print(correct_list)
# # acc = correct_list / n_total
# # print('正确率: ', acc)

data = pd.read_csv('./sample_submit.csv')
n_total = 24000
# print(np.arange(100))
real_gender = pd.DataFrame({'gender': np.arange(24000)}, index=[np.arange(n_total)])
print(real_gender)
print(real_gender['gender'][10])