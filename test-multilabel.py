#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/7/25 9:41
# @Author  : pyf
# @File    : test-multilabel.py
# @Description : 测试多标签列模型训练及预测

from autogluon.tabular import TabularDataset, TabularPredictor
from MultilabelPredictor import MultilabelPredictor

# 构建训练集
train_data = TabularDataset('./dataset/amazonaws_inc/train.csv')

# 抽样
# subsample_size = 500  # subsample subset of data for faster demo, try setting this to much larger values
# train_data = train_data.sample(n=subsample_size, random_state=0)

# 训练信息
labels = ['education-num', 'education', 'class']  # which columns to predict based on the others
problem_types = ['regression', 'multiclass', 'binary']  # type of each prediction problem (optional)
eval_metrics = ['mean_absolute_error', 'accuracy',
                'accuracy']  # metrics used to evaluate predictions for each label (optional)
save_path = 'models/aws-multilabel'  # specifies folder to store trained models (optional)

# 模型训练
# multi_predictor = MultilabelPredictor(labels=labels, problem_types=problem_types, eval_metrics=eval_metrics,
#                                       path=save_path)
# multi_predictor.fit(train_data)

# 模型加载
multi_predictor = MultilabelPredictor.load(
    save_path)  # unnecessary, just demonstrates how to load previously-trained multilabel predictor from file

# 构造测试集
test_data = TabularDataset('./dataset/amazonaws_inc/test.csv')
test_data_nolab = test_data.drop(columns=labels)  # unnecessary, just to demonstrate we're not cheating here

# 模型推理
predictions = multi_predictor.predict(test_data_nolab)
print("Predictions:  \n", predictions)

# 获取模型中的某一个标签对应的子模型
predictor_class = multi_predictor.get_predictor('class')
print(predictor_class.leaderboard(silent=True))
