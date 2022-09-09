#!/usr/bin/env python
# coding: utf-8

# # Description:
# 这是实验的数据预处理模块，此次实验使用的亚马逊产品数据集里面的Electronics子集， 具体详情描述可以参考：[http://jmcauley.ucsd.edu/data/amazon/](http://jmcauley.ucsd.edu/data/amazon/)。 这里用的2014年的那两个per-category dataset。大体思路分为两个部分：
# 1. 把原始的json数据转成pd的形式， 从meta数据集中只保留在reviews文件中出现过的商品
# 2. 把pd数据转成pkl数据， 后面用这个生成数据

# In[1]:


import numpy as np
import pandas as pd
import pickle
import gc
import random
from tqdm import tqdm

random.seed(2020)


# # Convert_pd

# In[3]:


def to_df(file_path):
    """
        转换为DataFrame结构
        file_path: 文件路径
        return: DtaFrame
    """
    with open(file_path, 'r') as fin:
        df = {}
        i = 0
        for line in tqdm(fin):
            #print(line)
            df[i] = eval(line)   #   直接针对字符串运行
            i += 1
            
            if i > 1000000:   # 笔记本内存不够了， 先提取少量一部分, 如果电脑允许，这里可以去掉
                break
        df = pd.DataFrame.from_dict(df, orient='index')
        return df            


# In[4]:


# 处理review
reviews_df = to_df('./raw_data/reviews_Electronics.json')


# In[5]:


reviews_df.head()


# In[6]:


with open('./raw_data/reviews.pkl', 'wb') as f:
    pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)


# In[7]:


unique_asin = reviews_df['asin'].unique()


# In[8]:


del reviews_df
gc.collect()


# In[9]:


# 处理meta_Electroics  从meta数据集中只保留在reviews文件中出现过的商品
meta_df = to_df('./raw_data/meta_Electronics.json')
meta_df = meta_df[meta_df['asin'].isin(unique_asin)]
meta_df = meta_df.reset_index(drop=True)


# In[15]:


meta_df.head()


# In[10]:


pickle.dump(meta_df, open('./raw_data/meta.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)


# # remap_id
# 这里再次进行处理， 基于上面的pkl文件， 处理如下：
# 1. reviews_df保留'reviewerID'【用户ID】, 'asin'【产品ID】, 'unixReviewTime'【浏览时间】三列
# 2. meta_df保留'asin'【产品ID】, 'categories'【种类】两列

# In[12]:


reviews = pd.read_pickle('./raw_data/reviews.pkl')
reviews_df = reviews[['reviewerID', 'asin', 'unixReviewTime']]

meta = pd.read_pickle('./raw_data/meta.pkl')
meta_df = meta[['asin', 'categories']]

del reviews, meta
gc.collect()


# In[13]:


# meta_df只保留最后一个
meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])


# In[14]:


meta_df.head()


# In[15]:


reviews_df.head()


# In[16]:


print(meta_df.shape, reviews_df.shape)


# In[17]:


# 上面的这个数太大了还是， 所以这里在进行采样一波， 按照用户的reviewerID采样， 采样出10万的用户数据来
select_user_id = np.random.choice(reviews_df['reviewerID'].unique(), size=100000, replace=False)
reviews_df = reviews_df[reviews_df['reviewerID'].isin(select_user_id)]
meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]


# In[18]:


print(meta_df.shape, reviews_df.shape)


# In[19]:


def build_map(df, col_name):
    """
    制作一个映射， 键为列名， 值为序列数字
    df: review_df / meta_df
    col_name: 列名
    return: 字典， 键
    """
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(len(key))))          # 这个是建立字典的常用操作， 得记住这个写法 [值， 索引]
    df[col_name] = df[col_name].map(lambda x: m[x])        # 这地方是把原来的值变为索引了？
    return m, key


# In[20]:


# 给物品ID， 物品种类， 用户ID，建立值 -> 索引的映射
asin_map, asin_key = build_map(meta_df, 'asin')
cate_map, cate_key = build_map(meta_df, 'categories')
revi_map, revi_key = build_map(reviews_df, 'reviewerID')


# In[21]:


user_count, item_count, cate_count, example_count = len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]
print(user_count, item_count, cate_count, example_count)


# In[22]:


# 按物品id排序， 并重置索引
meta_df = meta_df.sort_values('asin').reset_index(drop=True)


# In[23]:


# reviews_df文件物品id进行映射， 并按照用户id，浏览时间进行排序重置索引
reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])
reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime']).reset_index(drop=True)
reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]


# In[24]:


# 各个物品对应的类别
cate_list = np.array(meta_df['categories'], dtype='int32')


# In[25]:


# 保存所需数据为pkl文件
with open('./dataset/remap.pkl', 'wb') as f:
    pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count, example_count), f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((asin_key, cate_key, revi_key), f, pickle.HIGHEST_PROTOCOL)

