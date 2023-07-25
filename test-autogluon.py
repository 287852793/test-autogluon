#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/21 9:34
# @Author  : pyf
# @File    : test-autogluon.py
# @Description : 测试 autogluon 效果（以二分类问题为例）

from autogluon.tabular import TabularDataset, TabularPredictor

# data_url = 'https://raw.githubusercontent.com/mli/ag-docs/main/knot_theory/'
# train_data = TabularDataset(f'{data_url}train.csv')

# 构建训练数据集
train_data = TabularDataset('./dataset/amazonaws_inc/train.csv')

print(train_data.head())

# 设置标签列
label = 'class'
print(train_data[label].describe())

# 训练模型
# save_path = 'models/aws'
# predictor = TabularPredictor(label=label, path=save_path).fit(train_data)

# 加载模型（加载一个已训练好的模型）
predictor = TabularPredictor.load('./models/aws')

# 构建测试数据集
test_data = TabularDataset('./dataset/amazonaws_inc/test.csv')
print(test_data.head())

# 分离真值
y_true = test_data[label]
test_data_nolab = test_data.drop(columns=[label])

# 模型推理
y_pred = predictor.predict(test_data_nolab)
print(y_pred)

# 手动模型评估
# pref = predictor.evaluate_predictions(y_true=y_true, y_pred=y_pred)
# print(pref)

# 自动模型评估
# prefs = predictor.leaderboard(test_data, silent=True)
# # print(prefs)

# 多分类评估（计算各类型概率）
# pred_probs = predictor.predict_proba(test_data_nolab)
# print(pred_probs.head())

# 模型指标（模型在训练集上的表现）
# predictor_info = predictor.fit_summary(show_plot=True)
# print(predictor_info)

# 模型类型信息
# print("AutoGluon infers problem type is: ", predictor.problem_type)

# 模型可推导的信息（数据列）
# print("AutoGluon identified the following types of features:")
# print(predictor.feature_metadata)

# 手动指定推理模型
# predictor.predict(test_data, model='LightGBM')

