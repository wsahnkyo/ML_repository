#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F


import warnings

warnings.filterwarnings('ignore')


class CrossNetwork(nn.Module):
    """
    Cross Network
    """

    def __init__(self, layer_num, input_dim):
        super(CrossNetwork, self).__init__()
        self.layer_num = layer_num

        # 定义网络层的参数
        self.cross_weights = nn.ParameterList([
            nn.Parameter(torch.rand(input_dim, 1))
            for i in range(self.layer_num)
        ])
        self.cross_bias = nn.ParameterList([
            nn.Parameter(torch.rand(input_dim, 1))
            for i in range(self.layer_num)
        ])

    def forward(self, x):
        # x是(None, dim)的形状， 先扩展一个维度到(None, dim, 1)
        x_0 = torch.unsqueeze(x, dim=2)
        x = x_0.clone()
        xT = x_0.clone().permute((0, 2, 1))  # （None, 1, dim)
        for i in range(self.layer_num):
            x = torch.matmul(torch.bmm(x_0, xT), self.cross_weights[i]) + self.cross_bias[i] + x  # (None, dim, 1)
            xT = x.clone().permute((0, 2, 1))  # (None, 1, dim)

        x = torch.squeeze(x)  # (None, dim)
        return x


class Dnn(nn.Module):
    """
    Dnn part
    """

    def __init__(self, hidden_units, dropout=0.):
        """
        hidden_units: 列表， 每个元素表示每一层的神经单元个数， 比如[256, 128, 64], 两层网络， 第一层神经单元128， 第二层64， 第一个维度是输入维度
        dropout: 失活率
        """
        super(Dnn, self).__init__()

        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_units[:-1], hidden_units[1:]))])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        for linear in self.dnn_network:
            x = linear(x)
            x = F.relu(x)

        x = self.dropout(x)
        return x


class DCN(nn.Module):
    def __init__(self, feature_columns, hidden_units, layer_num, dnn_dropout=0.):
        super(DCN, self).__init__()
        self.dense_feature_cols, self.sparse_feature_cols = feature_columns

        # embedding 
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(i): nn.Embedding(num_embeddings=feat['feat_num'], embedding_dim=feat['embed_dim'])
            for i, feat in enumerate(self.sparse_feature_cols)
        })

        hidden_units.insert(0,
                            len(self.dense_feature_cols) + len(self.sparse_feature_cols) * self.sparse_feature_cols[0][
                                'embed_dim'])
        self.dnn_network = Dnn(hidden_units)
        self.cross_network = CrossNetwork(layer_num, hidden_units[0])  # layer_num是交叉网络的层数， hidden_units[0]表示输入的整体维度大小
        self.final_linear = nn.Linear(hidden_units[-1] + hidden_units[0], 1)

    def forward(self, x):
        dense_input, sparse_inputs = x[:, :len(self.dense_feature_cols)], x[:, len(self.dense_feature_cols):]
        sparse_inputs = sparse_inputs.long()
        sparse_embeds = [self.embed_layers['embed_' + str(i)](sparse_inputs[:, i]) for i in
                         range(sparse_inputs.shape[1])]
        sparse_embeds = torch.cat(sparse_embeds, axis=-1)

        x = torch.cat([sparse_embeds, dense_input], axis=-1)

        # cross Network
        cross_out = self.cross_network(x)

        # Deep Network
        deep_out = self.dnn_network(x)

        #  Concatenate
        total_x = torch.cat([cross_out, deep_out], axis=-1)

        # out
        outputs = F.sigmoid(self.final_linear(total_x))
        outputs = outputs.view(outputs.shape[0], )
        return outputs
