#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/24 15:13
# @Author  : pyf
# @File    : test-regression.py
# @Description : 回归问题推理测试

from autogluon.tabular import TabularDataset, TabularPredictor

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

# 模型指标（模型在训练集上的表现）
predictor_info = predictor.fit_summary(show_plot=True)
print(predictor_info)

# 构建测试数据集
test_data = TabularDataset('./dataset/amazonaws_inc/test.csv')
print(test_data.head())

# 分离真值
y_true = test_data[label]
test_data_nolab = test_data.drop(columns=[label])

# 模型推理
y_pred = predictor.predict(test_data_nolab)
print(y_pred)