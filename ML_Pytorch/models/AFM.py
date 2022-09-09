#!/usr/bin/env python
# coding: utf-8


"""导入包"""
import itertools

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F


import warnings
warnings.filterwarnings('ignore')


class Dnn(nn.Module):
    def __init__(self, hidden_units, dropout=0.):
        """
        hidden_units: 列表， 每个元素表示每一层的神经单元个数， 比如[256, 128, 64], 两层网络， 第一层神经单元128， 第二层64， 第一个维度是输入维度
        dropout = 0.
        """
        super(Dnn, self).__init__()
        
        self.dnn_network = nn.ModuleList([nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_units[:-1], hidden_units[1:]))])
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):  
        for linear in self.dnn_network:
            x = linear(x)
            x = F.relu(x)    
        x = self.dropout(x) 
        return x

class Attention_layer(nn.Module):
    def __init__(self, att_units):
        """
        :param att_units: [embed_dim, att_vector]
        """
        super(Attention_layer, self).__init__()
        
        self.att_w = nn.Linear(att_units[0], att_units[1])
        self.att_dense = nn.Linear(att_units[1], 1)
    
    def forward(self, bi_interaction):     # bi_interaction (None, (field_num*(field_num-1)_/2, embed_dim)
        a = self.att_w(bi_interaction)    # (None, (field_num*(field_num-1)_/2, t)
        a = F.relu(a)             # (None, (field_num*(field_num-1)_/2, t)
        att_scores = self.att_dense(a)  # (None, (field_num*(field_num-1)_/2, 1)
        att_weight = F.softmax(att_scores, dim=1)  #  (None, (field_num*(field_num-1)_/2, 1)

        att_out = torch.sum(att_weight * bi_interaction, dim=1)   # (None, embed_dim)
        return att_out     

class AFM(nn.Module):
    def __init__(self, feature_columns, mode, hidden_units, att_vector=8, dropout=0.5, useDNN=False):
        """
        AFM:
        :param feature_columns: 特征信息， 这个传入的是fea_cols array[0] dense_info  array[1] sparse_info
        :param mode: A string, 三种模式, 'max': max pooling, 'avg': average pooling 'att', Attention
        :param att_vector: 注意力网络的隐藏层单元个数
        :param hidden_units: DNN网络的隐藏单元个数， 一个列表的形式， 列表的长度代表层数， 每个元素代表每一层神经元个数， lambda文里面没加
        :param dropout: Dropout比率
        :param useDNN: 默认不使用DNN网络
        """
        super(AFM, self).__init__()
        self.dense_feature_cols, self.sparse_feature_cols = feature_columns
        self.mode = mode
        self.useDNN = useDNN
        
        # embedding
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(i): nn.Embedding(num_embeddings=feat['feat_num'], embedding_dim=feat['embed_dim'])
            for i, feat in enumerate(self.sparse_feature_cols)
        })
        
        # 如果是注意机制的话，这里需要加一个注意力网络
        if self.mode == 'att':   
            self.attention = Attention_layer([self.sparse_feature_cols[0]['embed_dim'], att_vector])
            
        # 如果使用DNN的话， 这里需要初始化DNN网络
        if self.useDNN:
            # 这里要注意Pytorch的linear和tf的dense的不同之处， 前者的linear需要输入特征和输出特征维度， 而传入的hidden_units的第一个是第一层隐藏的神经单元个数，这里需要加个输入维度
            self.fea_num = len(self.dense_feature_cols) + self.sparse_feature_cols[0]['embed_dim']
            hidden_units.insert(0, self.fea_num)

            self.bn = nn.BatchNorm1d(self.fea_num)     
            self.dnn_network = Dnn(hidden_units, dropout)
            self.nn_final_linear = nn.Linear(hidden_units[-1], 1)
        else:
            self.fea_num = len(self.dense_feature_cols) + self.sparse_feature_cols[0]['embed_dim']
            self.nn_final_linear = nn.Linear(self.fea_num, 1)
    
    def forward(self, x):
        dense_inputs, sparse_inputs = x[:, :len(self.dense_feature_cols)], x[:, len(self.dense_feature_cols):]
        sparse_inputs = sparse_inputs.long()       # 转成long类型才能作为nn.embedding的输入
        sparse_embeds = [self.embed_layers['embed_'+str(i)](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        sparse_embeds = torch.stack(sparse_embeds)     # embedding堆起来， (field_dim, None, embed_dim)
        sparse_embeds = sparse_embeds.permute((1, 0, 2))
        # 这里得到embedding向量之后 sparse_embeds(None, field_num, embed_dim)
        # 下面进行两两交叉， 注意这时候不能加和了，也就是NFM的那个计算公式不能用， 这里两两交叉的结果要进入Attention
        # 两两交叉enbedding之后的结果是一个(None, (field_num*field_num-1)/2, embed_dim) 
        # 这里实现的时候采用一个技巧就是组合
        #比如fild_num有4个的话，那么组合embeding就是[0,1] [0,2],[0,3],[1,2],[1,3],[2,3]位置的embedding乘积操作
        first = []
        second = []
        for f, s in itertools.combinations(range(sparse_embeds.shape[1]), 2):
            first.append(f)
            second.append(s)
        # 取出first位置的embedding  假设field是3的话，就是[0, 0, 0, 1, 1, 2]位置的embedding
        p = sparse_embeds[:, first, :]     # (None, (field_num*(field_num-1)_/2, embed_dim)
        q = sparse_embeds[:, second, :]   # (None, (field_num*(field_num-1)_/2, embed_dim)
        bi_interaction = p * q    # (None, (field_num*(field_num-1)_/2, embed_dim)
        
        if self.mode == 'max':
            att_out = torch.sum(bi_interaction, dim=1)  #  (None, embed_dim)
        elif self.mode == 'avg':
            att_out = torch.mean(bi_interaction, dim=1)  # (None, embed_dim)
        else:
            # 注意力网络
            att_out = self.attention(bi_interaction)  # (None, embed_dim)
        
        # 把离散特征和连续特征进行拼接
        x = torch.cat([att_out, dense_inputs], dim=-1)
        
        if not self.useDNN:
            outputs = F.sigmoid(self.nn_final_linear(x))
        else:
            # BatchNormalization
            x = self.bn(x)
            # deep
            dnn_outputs = self.nn_final_linear(self.dnn_network(x))
            outputs = F.sigmoid(dnn_outputs)

        outputs = outputs.view(outputs.shape[0], )
        return outputs


