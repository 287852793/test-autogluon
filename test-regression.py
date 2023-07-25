#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/24 15:13
# @Author  : pyf
# @File    : test-regression.py
# @Description : 回归问题推理测试

from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd

# 构建训练数据集
train_data = TabularDataset('./dataset/amazonaws_inc/train.csv')

# 设置标签列
label = 'age'
# print("Summary of age variable: \n", train_data[label].describe())

# 训练模型
# save_path = 'models/aws-age'
# predictor = TabularPredictor(problem_type='regression', label=label, path=save_path).fit(train_data)

# 加载模型（加载一个已训练好的模型）
predictor = TabularPredictor.load('./models/aws-age')

# 模型可解释性分析（各字段权重分析）
# model_info = predictor.feature_importance(train_data)
# print(model_info)

# # 模型指标（模型在训练集上的表现）
# predictor_info = predictor.fit_summary(show_plot=True)
# print(predictor_info)
#
# # 构建测试数据集
# test_data = TabularDataset('./dataset/amazonaws_inc/test.csv')
# print(test_data.head())
#
# # 分离真值
# y_true = test_data[label]
# test_data_nolab = test_data.drop(columns=[label])
#
# # 模型推理
# y_pred = predictor.predict(test_data_nolab)
# print(y_pred)

# 构造一条数据
data = [{
    # 'age': 31,
    'workclass': 'Private',
    'fnlwgt': 169085,
    'education': '11th',
    'education-num': 12,
    'marital-status': 'Never-married',
    'occupation': 'Sales',
    'relationship': 'Not-in-family',
    'race': 'White',
    'sex': 'Female',
    'capital-gain': 0,
    'capital-loss': 0,
    'hours-per-week': 40,
    'native-country': 'United-States',
    'class': '<=50K'
}]

# 推理一条数据
df = pd.DataFrame(data)
res = predictor.predict(df)
print(res)
print(res.loc[0])
