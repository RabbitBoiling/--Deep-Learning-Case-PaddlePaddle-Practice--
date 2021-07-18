# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 18:36:21 2021

@author: Biao
"""
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None) # 显示DataFrame的所有行
from sklearn import preprocessing
import tensorflow as tf
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def prepare_data(selected_data):
    # 为缺失Age记灵境和彷设置为平均僮
    age_mean_value = selected_data['Age'].mean()
    selected_data.loc[:,'Age'] = selected_data.loc[:,'Age'].fillna(age_mean_value)
    # 为缺失Fare记灵境究危
    fare_mean_value = selected_data['Fare'].mean()
    selected_data.loc[:,'Fare'] = selected_data.loc[:,'Fare'].fillna(fare_mean_value)
    # 为缺失Embarked记灵境芫僮
    selected_data.loc[:,'Embarked'] = selected_data.loc[:,'Embarked'].fillna('S')

    # sex字符串转痪为数字编码
    selected_data.loc[:,'Sex'] = selected_data.loc[:,'Sex'].map({'female': 0, 'male': 1}).astype(int)
    # 港口embarked团字母表示转唤为数字编码
    selected_data.loc[:,'Embarked'] = selected_data.loc[:,'Embarked'].map({'C': 0, 'Q': 1, 'S': 2}).astype(int)

    # drop不改变原有的df中的数据，而是返回另一个DataFrame来存放删除后的数据,axis = 1 表示删除列
    selected_df_data = selected_data.drop(['Name'], axis=1)

    # 转挽为ndarray数绢
    ndarray_data = selected_df_data.values
    # 后7叨悬特征列
    features = ndarray_data[:, 1:]
    # 第0列是标签列
    label = ndarray_data[:, 0]

    # 特彶詹标淮化
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    norm_features = minmax_scale.fit_transform(features)

    return norm_features, label

# 得到处理后的训练集
# 读取数据文件，结果为DataFrame格式
data_file_path = "./train.xlsx"
df_data = pd.read_excel(data_file_path)
# 筛选提取需要的待彶字段，去葆ticket, cabin等
selected_cols = ['Survived', 'Name', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
selected_df_data = df_data[selected_cols]

# shuffle, 打刮数据顺序，通过Pandas的拙样函数sample实现frac为百分比
shuffled_df_data = selected_df_data.sample(frac=1)
x_train_data, y_train_data = prepare_data(selected_data = shuffled_df_data)

# 得到处理后的测试集
data_file_path = "./train.xlsx"
df_data = pd.read_excel(data_file_path)
selected_cols = ['Survived', 'Name', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
selected_df_data = df_data[selected_cols]
shuffled_df_data = selected_df_data.sample(frac=1)
x_test_data, y_test_data = prepare_data(selected_data = shuffled_df_data)


# 建立模型结构
# 建立Keras序判槐型
model = tf.keras.models.Sequential()

# 加人第一厉，输入特徙数据是7见构可以厉i'nput_shape= (7, )
model.add(tf.keras.layers.Dense(units=64, input_dim=7, use_bias=True, kernel_initializer='uniform', bias_initializer='zeros',activation='relu'))
model.add(tf.keras.layers.Dense(units=32, activation='sigmoid'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
# 模型结构
model.summary()
# 模型设置
model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss='binary_crossentropy', metrics=['accuracy'])
# 模型训练
train_history = model.fit(x=x_train_data, y=y_train_data, validation_split=0.2, epochs=300, batch_size=32, verbose=2)
# 训练过程的历史数据：字典模式存储
train_history.history.keys()

# 模型训练过程可视化
def visu_train_history(train_history, train_metric, validation_metric):
    plt.plot(train_history.history[train_metric])
    plt.plot(train_history.history[validation_metric])
    plt.title('Train History')
    plt.ylabel(train_metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


visu_train_history(train_history, 'accuracy', 'val_accuracy')
visu_train_history(train_history, 'loss', 'val_loss')

# 模型评估
evaluate_result = model.evaluate(x=x_test_data, y=y_test_data)
model_evaluate_names = model.metrics_names
evaluate_result
print('模型的评估结果为：名字{}数据{}'.format(model_evaluate_names,evaluate_result))

# 模型应用
# Jack和Rose的旅客信息
Jack_info = [0, 'Jack', 3, 'male', 23, 1, 0, 5.0000, 'S']
Rose_info = [1, 'Rose', 1, 'female', 20, 1, 0, 100.0000, 'S']
# 测试信息
file_path = "./train.xlsx"
primary_data = pd.read_excel(file_path)
selected_primary_data = primary_data[selected_cols]
shuffled_primary_data = selected_primary_data.sample(frac=1)
new_test = pd.DataFrame([Jack_info, Rose_info],columns=selected_cols)
new_test_data = shuffled_primary_data.append(new_test)
x_verify_data, y_verify_data = prepare_data(selected_data = new_test_data)
Survival_probability = model.predict(x_verify_data)
print('Jack与Rose的生存概率分别为{},{}'.format(Survival_probability[891,0],Survival_probability[892,0]))

