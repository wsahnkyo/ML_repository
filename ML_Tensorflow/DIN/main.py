#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import pickle
import random

from time import time

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
import tensorflow
from tensorflow import keras
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.metrics import AUC
# from tensorflow.keras.losses import binary_crossentropy

from DIN import DIN
from data_crecate import create_amazon_electronic_dataset

import warnings

warnings.filterwarnings('ignore')

# In[2]:


"""数据生成"""
file_name = './dataset/remap.pkl'
feature_columns, behavior_list, (train_X, train_y), (val_X, val_y), (test_X, test_y) = create_amazon_electronic_dataset(
    file_name)

# In[3]:


"""超参数设置"""
maxlen = 40
embed_dim = 8
att_hidden_units = [80, 40]
ffn_hidden_units = [256, 128, 64]
dnn_dropout = 0.5
att_activation = 'sigmoid'
ffn_activation = 'prelu'

learning_rate = 0.001
batch_size = 64
epochs = 50

# In[4]:


"""模型建立"""
model = DIN(feature_columns, behavior_list, att_hidden_units, ffn_hidden_units, att_activation, ffn_activation, maxlen,
            dnn_dropout)
model.summary()

# In[5]:


"""模型编译"""
model.compile(loss=tensorflow.keras.losses.binary_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adam(learning_rate=learning_rate), metrics=[keras.metrics.AUC()])

# In[6]:


"""模型训练"""
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),  # 早停
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.01, verbose=1)  # 调整学习率
]
history = model.fit(train_X,
                    train_y,
                    epochs=epochs,
                    validation_data=(val_X, val_y),
                    batch_size=batch_size,
                    callbacks=callbacks
                    )

# In[9]:


"""可视化下看看训练情况"""
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# 这里发现1个epoch的时候，后面就开始过拟合了。我这次用的数据量太小了。

# In[10]:


"""模型评估"""
print('test AUC: %f' % model.evaluate(test_X, test_y)[1])
