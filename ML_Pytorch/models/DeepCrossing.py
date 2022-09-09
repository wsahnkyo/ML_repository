#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings('ignore')


# 首先， 自定义一个残差块
class Residual_block(nn.Module):
    """
    Define Residual_block
    """

    def __init__(self, hidden_unit, dim_stack):
        super(Residual_block, self).__init__()
        self.linear1 = nn.Linear(dim_stack, hidden_unit)
        self.linear2 = nn.Linear(hidden_unit, dim_stack)
        self.relu = nn.ReLU()

    def forward(self, x):
        orig_x = x.clone()
        x = self.linear1(x)
        x = self.linear2(x)
        outputs = self.relu(x + orig_x)
        return outputs


# 定义deep Crossing 网络
class DeepCrossing(nn.Module):

    def __init__(self, feature_columns, hidden_units, dropout=0., embedding_dim=10, output_dim=1):
        super(DeepCrossing, self).__init__()
        self.dense_feature_cols, self.sparse_feature_cols = feature_columns

        # embedding
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(i): nn.Embedding(num_embeddings=feat['feat_num'], embedding_dim=feat['embed_dim'])
            for i, feat in enumerate(self.sparse_feature_cols)
        })

        # 统计embedding_dim的总维度
        embed_dim_sum = sum([embedding_dim] * len(self.sparse_feature_cols))

        # stack layers的总维度
        dim_stack = len(self.dense_feature_cols) + embed_dim_sum

        # 残差层
        self.res_layers = nn.ModuleList([
            Residual_block(unit, dim_stack) for unit in hidden_units
        ])

        # dropout层
        self.res_dropout = nn.Dropout(dropout)

        # 线性层
        self.linear = nn.Linear(dim_stack, output_dim)

    def forward(self, x):
        dense_inputs, sparse_inputs = x[:, :13], x[:, 13:]
        sparse_inputs = sparse_inputs.long()  # 需要转成长张量， 这个是embedding的输入要求格式
        sparse_embeds = [self.embed_layers['embed_' + str(i)](sparse_inputs[:, i]) for i in
                         range(sparse_inputs.shape[1])]

        sparse_embed = torch.cat(sparse_embeds, axis=-1)
        stack = torch.cat([sparse_embed, dense_inputs], axis=-1)
        r = stack
        for res in self.res_layers:
            r = res(r)

        r = self.res_dropout(r)
        outputs = F.sigmoid(self.linear(r))
        outputs = outputs.view(outputs.shape[0], )
        return outputs
