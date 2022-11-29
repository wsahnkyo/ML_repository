# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')


class CIN(torch.nn.Module):
    def __init__(self, input_dim, num_layers=2):
        super(CIN, self).__init__()
        # CIN 网络有几层，也就是要几阶
        self.num_layers = num_layers
        # 一维卷积层
        self.conv_layers = torch.nn.ModuleList()
        fc_input_dim = 0
        for i in range(self.num_layers):
            ''' in_channels: 输入信号的通道 向量的维度 ,input_dim的长度指的是特征的总数
                out_channels:卷积产生的通道。有多少个out_channels，就需要多少个1维卷积 
                kerner_size :卷积核的尺寸，卷积核的大小为(k,)，第二个维度是由in_channels来决定的，所以实际上卷积大小为kerner_size*in_channels
                stride : 卷积步长 
                dilation :卷积核元素之间的间距'''
            self.conv_layers.append(

                torch.nn.Conv1d(in_channels=input_dim * input_dim, out_channels=input_dim, kernel_size=1,
                                stride=1, dilation=1, bias=True))
        self.fc = torch.nn.Linear(input_dim * num_layers, 1)

    def forward(self, x):
        xs = list()
        '''举例  x.shape = [1,22,16] 1表示batch_size,表示有几维数据，22表示特征的维数，16是embedding层的向量大小
        经过 x.unsqueeze(2)后 x.shape = [1,22,1,16]
        经过 x.unsqueeze(1)后 x.shape = [1,1,22,16]  
        x.unsqueeze(2) * x.unsqueeze(1) 后   x.shape =[1,22,22,16]
        进过卷积层后变为 x.shape =[1,16,16]
        经过 sum pooling  变为 1维
         '''
        x0, h = x.unsqueeze(2), x
        for i in range(self.num_layers):
            h1 = h.unsqueeze(1)
            x = x0 * h1
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)
            x = self.conv_layers[i](x)
            x = F.relu(x)
            h = x
            xs.append(x)
        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))


class Linear(nn.Module):
    """
    Linear part
    """

    def __init__(self, input_dim):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=1)

    def forward(self, x):
        return self.linear(x)


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


class XDeepFM(nn.Module):
    def __init__(self, feature_columns, hidden_units, layer_num, dropout=0.):
        super(XDeepFM, self).__init__()
        self.dense_feature_cols, self.sparse_feature_cols = feature_columns

        # embedding
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(i): nn.Embedding(num_embeddings=feat['feat_num'], embedding_dim=feat['embed_dim'])
            for i, feat in enumerate(self.sparse_feature_cols)
        })

        hidden_units.insert(0,
                            len(self.dense_feature_cols) + len(self.sparse_feature_cols) * self.sparse_feature_cols[0][
                                'embed_dim'])
        self.linear = Linear(len(self.dense_feature_cols))
        self.dnn_network = Dnn(hidden_units)
        self.cin_network = CIN(len(self.sparse_feature_cols), layer_num)
        self.final_linear = nn.Linear(hidden_units[-1], 1)

    def forward(self, x):
        dense_input, sparse_inputs = x[:, :len(self.dense_feature_cols)], x[:, len(self.dense_feature_cols):]
        sparse_inputs = sparse_inputs.long()
        sparse_embeds = [self.embed_layers['embed_' + str(i)](sparse_inputs[:, i]) for i in
                         range(sparse_inputs.shape[1])]
        cin_input = torch.stack(sparse_embeds, dim=1)
        sparse_embeds = torch.cat(sparse_embeds, axis=-1)

        x = torch.cat([sparse_embeds, dense_input], axis=-1)

        # linear NetWork
        linear_out = self.linear(dense_input)
        # cin Network
        cin_out = self.cin_network(cin_input)

        # Deep Network
        deep_out = self.dnn_network(x)
        deep_out = self.final_linear(deep_out)
        #  Concatenate
        # total_x = torch.cat([linear_out, cin_out, deep_out], axis=-1)

        # out
        outputs = F.sigmoid(linear_out + cin_out + deep_out)
        outputs = outputs.view(outputs.shape[0], )
        return outputs
