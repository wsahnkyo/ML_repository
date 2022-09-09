#!/usr/bin/env python
# coding: utf-8
# python基础包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 特征处理与数据集划分
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from ML_Tensorflow.models.AutoInt import AutoInt
from ML_Tensorflow.utils import DenseFeat, SparseFeat, VarLenSparseFeat

# 导入模型


from tensorflow import keras

# 一些相关设置
import warnings

plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')


# ## 导入数据与初步处理

# In[2]:


# 简单处理特征， 包括填充缺失值，数值处理， 类别编码
def data_process(data_df, dense_features, sparse_features):
    # 连续型特征用0填充缺失, 并进行对数转换
    data_df[dense_features] = data_df[dense_features].fillna(0.0)
    for f in dense_features:
        data_df[f] = data_df[f].apply(lambda x: np.log(x + 1) if x > -1 else -1)

    # 离散型特征用-1填充，并进行LabelEncoder编码
    data_df[sparse_features] = data_df[sparse_features].fillna("-1")
    for f in sparse_features:
        lbe = LabelEncoder()
        data_df[f] = lbe.fit_transform(data_df[f])

    return data_df[dense_features + sparse_features]


# In[3]:


"""读取数据"""
data = pd.read_csv('data/criteo_sample.txt')

# In[4]:


"""划分dense和sparse特征"""
columns = data.columns.values
dense_features = [feat for feat in columns if 'I' in feat]
sparse_features = [feat for feat in columns if 'C' in feat]

# In[5]:


"""简单的数据预处理"""
train_data = data_process(data, dense_features, sparse_features)
train_data['label'] = data['label']

# In[6]:


"""划分数据集"""
trnx, valx, trny, valy = train_test_split(train_data[dense_features + sparse_features], train_data['label'],
                                          test_size=0.1)

# ## 模型建立

# In[7]:


"""特征分组， 分成linear部分和DNN部分(根据实际场景进行选择)，并将分组之后的特征用namedtuple封装起来(DenseFeat, SparseFeat)"""
linear_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4)
                          for i, feat in enumerate(sparse_features)] + \
                         [DenseFeat(feat, 1, ) for feat in dense_features]

dnn_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4)
                       for i, feat in enumerate(sparse_features)] + \
                      [DenseFeat(feat, 1, ) for feat in dense_features]

# In[8]:


"""构建AutoInt模型"""
model = AutoInt(linear_feature_columns, dnn_feature_columns)
model.summary()

# In[10]:


"""设置超参数"""
learning_rate = 0.001
batch_size = 64
epochs = 20

# In[11]:

"""模型编译"""
model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
              metrics=[keras.metrics.AUC()])

# In[6]:


"""模型训练"""
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),  # 早停
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.01, verbose=1)  # 调整学习率
]

# 将输入数据转化成字典的形式输入
trnx_model_input = {name: trnx[name] for name in dense_features + sparse_features}
valx_model_input = {name: valx[name] for name in dense_features + sparse_features}

history = model.fit(
    trnx_model_input,
    trny.values,
    epochs=epochs,
    # callbacks=callbacks,  # checkpoint
    batch_size=batch_size,
    validation_split=0.2
)

# In[13]:


"""可视化下看看训练情况"""
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# In[14]:


"""模型评估"""
print('val AUC: %f' % model.evaluate(valx_model_input, valy, batch_size=batch_size)[1])

# ## 模型架构

# In[16]:


from tensorflow import keras

keras.utils.plot_model(model, to_file='./AutoInt_arc.png', show_shapes=True)
