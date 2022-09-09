#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

class Dnn(nn.Module):

    def __init__(self, hidden_units, dropout=0.):
        super(Dnn, self).__init__()
        print(list(zip(hidden_units[:-1], hidden_units[1:])))
        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_units[:-1], hidden_units[1:]))])
        self.dropout = nn.Dropout(p=dropout)
        self.final_linear = nn.Linear(hidden_units[-1], 1)

    def forward(self, x):
        for linear in self.dnn_network:
            x = linear(x)
            x = F.relu(x)

        x = self.dropout(x)
        x = self.final_linear(x)
        return x


class WideDeepAttention(nn.Module):
    def __init__(self, feature_columns, hidden_units, embedding_dim=40, dropout=0.):
        super(WideDeepAttention, self).__init__()
        self.dense_feature_cols, self.sparse_feature_cols = feature_columns
        # embedding
        self.sparse_embed_layers = nn.ModuleDict({
            'embed_' + str(i): nn.Embedding(num_embeddings=feat['feat_num'], embedding_dim=feat['embed_dim'])
            for i, feat in enumerate(self.sparse_feature_cols)
        })
        self.dense_embed_layers = nn.Embedding(num_embeddings=10000000, embedding_dim=embedding_dim)
        hidden_units.insert(0, (len(self.dense_feature_cols) + len(self.sparse_feature_cols)) *
                            self.sparse_feature_cols[0]['embed_dim'])
        self.dnn_network = Dnn(hidden_units, dropout=dropout)
        self.multiheadAttention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1)
        self.linear = nn.Linear(in_features=len(self.dense_feature_cols), out_features=1)

    def forward(self, x):
        dense_input, sparse_inputs = x[:, :len(self.dense_feature_cols)], x[:, len(self.dense_feature_cols):]
        dense_input = dense_input.long()
        sparse_inputs = sparse_inputs.long()
        dense_embeds = self.dense_embed_layers(dense_input)
        sparse_embeds = [self.sparse_embed_layers['embed_' + str(i)](sparse_inputs[:, i]) for i in
                         range(sparse_inputs.shape[1])]
        sparse_embeds = torch.stack(sparse_embeds, dim=1)
        input_embeds = torch.hstack((sparse_embeds, dense_embeds))
        multihead_attention_output = self.multiheadAttention(input_embeds, input_embeds, input_embeds)
        input_embeds = input_embeds + multihead_attention_output[0]
        dnn_input = torch.flatten(input_embeds, start_dim=1)
        wide_out = self.linear(dense_input.float())
        deep_out = self.dnn_network(dnn_input)
        outputs = F.sigmoid(0.5 * (wide_out + deep_out))
        outputs = outputs.view(outputs.shape[0], )
        return outputs