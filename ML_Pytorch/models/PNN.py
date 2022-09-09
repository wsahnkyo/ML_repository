#!/usr/bin/env python
# coding: utf-8

import torch

import torch.nn as nn
import torch.nn.functional as F


import warnings
warnings.filterwarnings('ignore')







# ## 建立模型
# 建立模型有三种方式：
# 1. 继承nn.Module基类构建自定义模型
# 2. nn.Sequential按层顺序构建模型
# 3. 继承nn.Module基类构建模型， 并辅助应用模型容器进行封装
# 
# 这里我们依然会使用第三种方式， 因为embedding依然是很多层。 模型的结构如下：
# 
# ![](img/pnn.png)
# 
# 这里简单的分析一下这个模型， 说几个比较重要的细节：
# 1. 这里的输入， 由于都进行了embedding， 所以这里应该是类别型的特征， 关于数值型的特征， 在把类别都交叉完了之后， 才把数值型的特征加入进去
# 2. 交叉层这里， 左边和右边其实用的同样的一层， 有着同样的神经单元个数， 只不过这里进行计算的时候， 得分开算，左边的是单个特征的线性组合lz， 而右边是两两特征进行交叉后特征的线性组合lp。 得到这两个之后， 两者进行相加得到最终的组合， 然后再relu激活才是交叉层的输出。
# 3. 交叉层这里图上给出的是**一个神经元**内部的计算情况， 注意这里是一个神经元内部的计算， 这些圈不是表示多个神经元。
# 
# 下面说一下代码的逻辑：
# 1. 首先， 我们定义一个DNN神经网络， 这个也就是上面图片里面的交叉层上面的那一部分结构， 也就是很多个全连接层的一个网络， 之所以单独定义这样的一个网络， 是因为更加的灵活， 加多少层， 每层神经元个数是多少我们就可以自己指定了， 这里会涉及到一个小操作技巧。
# 2. 然后就是定义整个PNN网络， 核心部分就是在前向传播。




# 定义一个全连接层的神经网络
class DNN(nn.Module):
    
    def __init__(self, hidden_units, dropout=0.):
        """
        hidden_units:列表， 每个元素表示每一层的神经单元个数，比如[256, 128, 64]，两层网络， 第一层神经单元128个，第二层64，注意第一个是输入维度
        dropout: 失活率
        """
        super(DNN, self).__init__()
        
        # 下面创建深层网络的代码 由于Pytorch的nn.Linear需要的输入是(输入特征数量， 输出特征数量)格式， 所以我们传入hidden_units， 
        # 必须产生元素对的形式才能定义具体的线性层， 且Pytorch中线性层只有线性层， 不带激活函数。 这个要与tf里面的Dense区分开。
        self.dnn_network = nn.ModuleList([nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_units[:-1], hidden_units[1:]))])
        self.dropout = nn.Dropout(p=dropout)
    
    # 前向传播中， 需要遍历dnn_network， 不要忘了加激活函数
    def forward(self, x):
        
        for linear in self.dnn_network:
            x = linear(x)
            x = F.relu(x)
        
        x = self.dropout(x)
        
        return x


class ProductLayer(nn.Module):
    
    def __init__(self, mode, embed_dim, field_num, hidden_units):
        
        super(ProductLayer, self).__init__()
        self.mode = mode
        # product层， 由于交叉这里分为两部分， 一部分是单独的特征运算， 也就是上面结构的z部分， 一个是两两交叉， p部分， 而p部分还分为了内积交叉和外积交叉
        # 所以， 这里需要自己定义参数张量进行计算
        # z部分的w， 这里的神经单元个数是hidden_units[0], 上面我们说过， 全连接层的第一层神经单元个数是hidden_units[1]， 而0层是输入层的神经
        # 单元个数， 正好是product层的输出层  关于维度， 这个可以看在博客中的分析
        self.w_z = nn.Parameter(torch.rand([field_num, embed_dim, hidden_units[0]]))
        
        # p部分, 分内积和外积两种操作
        if mode == 'in':
            self.w_p = nn.Parameter(torch.rand([field_num, field_num, hidden_units[0]]))
        else:
            self.w_p = nn.Parameter(torch.rand([embed_dim, embed_dim, hidden_units[0]]))
        
        self.l_b = torch.rand([hidden_units[0], ], requires_grad=True)
    
    def forward(self, z, sparse_embeds):
        # lz部分
        l_z = torch.mm(z.reshape(z.shape[0], -1), self.w_z.permute((2, 0, 1)).reshape(self.w_z.shape[2], -1).T)# (None, hidden_units[0])
        
        # lp 部分
        if self.mode == 'in':  # in模式  内积操作  p就是两两embedding先内积得到的[field_dim, field_dim]的矩阵
            p = torch.matmul(sparse_embeds, sparse_embeds.permute((0, 2, 1)))  # [None, field_num, field_num]
        else:  # 外积模式  这里的p矩阵是两两embedding先外积得到n*n个[embed_dim, embed_dim]的矩阵， 然后对应位置求和得到最终的1个[embed_dim, embed_dim]的矩阵
            # 所以这里实现的时候， 可以先把sparse_embeds矩阵在field_num方向上先求和， 然后再外积
            f_sum = torch.unsqueeze(torch.sum(sparse_embeds, dim=1), dim=1)  # [None, 1, embed_dim]
            p = torch.matmul(f_sum.permute((0, 2,1)), f_sum)     # [None, embed_dim, embed_dim]
        
        l_p = torch.mm(p.reshape(p.shape[0], -1), self.w_p.permute((2, 0, 1)).reshape(self.w_p.shape[2], -1).T)  # [None, hidden_units[0]]
        
        output = l_p + l_z + self.l_b
        return output


# In[33]:


# 下面我们定义真正的PNN网络
# 这里的逻辑是底层输入（类别型特征) -> embedding层 -> product 层 -> DNN -> 输出
class PNN(nn.Module):
    
    def __init__(self, feature_columns, hidden_units, mode='in', dropout=0., embed_dim=10, outdim=1):
        """
        DeepCrossing：
            feature_info: 特征信息（数值特征， 类别特征， 类别特征embedding映射)
            hidden_units: 列表， 全连接层的每一层神经单元个数， 这里注意一下， 第一层神经单元个数实际上是hidden_units[1]， 因为hidden_units[0]是输入层
            dropout: Dropout层的失活比例
            embed_dim: embedding的维度m
            outdim: 网络的输出维度
        """
        super(PNN, self).__init__()
        self.dense_feature_cols, self.sparse_feature_cols = feature_columns
        # self.dense_feas, self.sparse_feas, self.sparse_feas_map = feature_info
        self.field_num = len(self.sparse_feature_cols)
        self.dense_num = len(self.dense_feature_cols)
        self.mode = mode
        self.embed_dim = embed_dim
        
         # embedding层， 这里需要一个列表的形式， 因为每个类别特征都需要embedding
        # self.embed_layers = nn.ModuleDict({
        #     'embed_' + str(key): nn.Embedding(num_embeddings=val, embedding_dim=self.embed_dim)
        #     for key, val in self.sparse_feas_map.items()
        # })

        self.embed_layers = nn.ModuleDict({
            'embed_' + str(i): nn.Embedding(num_embeddings=feat['feat_num'], embedding_dim=feat['embed_dim'])
            for i, feat in enumerate(self.sparse_feature_cols)
        })
        
        # Product层
        self.product = ProductLayer(mode, embed_dim, self.field_num, hidden_units)
        
        # dnn 层
        hidden_units[0] += self.dense_num
        self.dnn_network = DNN(hidden_units, dropout)
        self.dense_final = nn.Linear(hidden_units[-1], 1)
    
    def forward(self, x):

        dense_inputs, sparse_inputs = x[:, :len(self.dense_feature_cols)], x[:, len(self.dense_feature_cols):]
        sparse_inputs = sparse_inputs.long()       # 转成long类型才能作为nn.embedding的输入
        sparse_embeds = [self.embed_layers['embed_'+str(i)](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        # 上面这个sparse_embeds的维度是 [field_num, None, embed_dim]
        sparse_embeds = torch.stack(sparse_embeds)
        sparse_embeds = sparse_embeds.permute((1, 0, 2))   # [None, field_num, embed_dim]  注意此时空间不连续， 下面改变形状不能用view，用reshape
        z = sparse_embeds
        # product layer
        sparse_inputs = self.product(z, sparse_embeds)


        # 把上面的连起来， 注意此时要加上数值特征
        l1 = F.relu(torch.cat([sparse_inputs, dense_inputs], axis=-1))
        # dnn_network
        dnn_x = self.dnn_network(l1)
        
        outputs = F.sigmoid(self.dense_final(dnn_x))
        outputs = outputs.view(outputs.shape[0], )
        return outputs 

