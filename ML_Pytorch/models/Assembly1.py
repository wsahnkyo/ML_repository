# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')


class Linear(nn.Module):
    """
    Linear part
    """

    def __init__(self, input_dim):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=1)

    def forward(self, x):
        return self.linear(x)


class SENetAttention(nn.Module):
    """
    Squeeze-and-Excitation Attention
    输入shape: [batch_size, num_fields, d_embed]   #num_fields即num_features
    输出shape: [batch_size, num_fields, d_embed]
    """

    def __init__(self, num_fields, reduction_ratio=3):
        super().__init__()
        reduced_size = max(1, int(num_fields / reduction_ratio))
        self.excitation = nn.Sequential(nn.Linear(num_fields, reduced_size, bias=False),
                                        nn.ReLU(),
                                        nn.Linear(reduced_size, num_fields, bias=False),
                                        nn.ReLU())

    def forward(self, x):
        Z = torch.mean(x, dim=-1, out=None)  # 1,Sequeeze
        A = self.excitation(Z)  # 2,Excitation
        V = x * A.unsqueeze(-1)  # 3,Re-Weight
        return V


class BilinearInteraction(nn.Module):
    """
    双线性FFM
    输入shape: [batch_size, num_fields, d_embed] #num_fields即num_features
    输出shape: [batch_size, num_fields*(num_fields-1)/2, d_embed]
    """

    def __init__(self, num_fields, d_embed, bilinear_type="field_interaction"):
        super().__init__()
        self.bilinear_type = bilinear_type
        if self.bilinear_type == "field_all":
            self.bilinear_layer = nn.Linear(d_embed, d_embed, bias=False)
        elif self.bilinear_type == "field_each":
            self.bilinear_layer = nn.ModuleList([nn.Linear(d_embed, d_embed, bias=False)
                                                 for i in range(num_fields)])
        elif self.bilinear_type == "field_interaction":
            self.bilinear_layer = nn.ModuleList([nn.Linear(d_embed, d_embed, bias=False)
                                                 for i, j in combinations(range(num_fields), 2)])
        else:
            raise NotImplementedError()

    def forward(self, feature_emb):
        feature_emb_list = torch.split(feature_emb, 1, dim=1)
        if self.bilinear_type == "field_all":
            bilinear_list = [self.bilinear_layer(v_i) * v_j
                             for v_i, v_j in combinations(feature_emb_list, 2)]
        elif self.bilinear_type == "field_each":
            bilinear_list = [self.bilinear_layer[i](feature_emb_list[i]) * feature_emb_list[j]
                             for i, j in combinations(range(len(feature_emb_list)), 2)]
        elif self.bilinear_type == "field_interaction":
            bilinear_list = [self.bilinear_layer[i](v[0]) * v[1]
                             for i, v in enumerate(combinations(feature_emb_list, 2))]
        return torch.cat(bilinear_list, dim=1)


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


class MultiLayerPerceptron(nn.Module):
    def __init__(self, d_in, d_layers, dropout,
                 d_out=1):
        super().__init__()
        layers = []
        for d in d_layers:
            layers.append(nn.Linear(d_in, d))
            layers.append(nn.BatchNorm1d(d))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            d_in = d
        layers.append(nn.Linear(d_layers[-1], d_out))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        float tensor of size ``(batch_size, d_in)``
        """
        return self.mlp(x)


class Assembly1(torch.nn.Module):
    def __init__(self, feature_columns, hidden_units, layer_num, dropout=0.):
        super(Assembly1, self).__init__()
        self.dense_feature_cols, self.sparse_feature_cols = feature_columns

        # embedding
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(i): nn.Embedding(num_embeddings=feat['feat_num'], embedding_dim=feat['embed_dim'])
            for i, feat in enumerate(self.sparse_feature_cols)
        })

        # hidden_units.insert(0, len(self.sparse_feature_cols) * self.sparse_feature_cols[0][
        #     'embed_dim'])
        # self.senet = SENET()
        self.cin = CIN(len(self.sparse_feature_cols), layer_num)  # layer_num是交叉网络的层数， hidden_units[0]表示输入的整体维度大小
        self.linear = Linear(len(self.dense_feature_cols))
        # self.final_linear = nn.Linear(hidden_units[-1] + hidden_units[0], 1)
        self.se_attention = SENetAttention(len(self.sparse_feature_cols))
        self.bilinear = BilinearInteraction(len(self.sparse_feature_cols), self.sparse_feature_cols[0]['embed_dim'],
                                            "field_interaction")
        mlp_in = len(self.sparse_feature_cols) * (len(self.sparse_feature_cols) - 1) * self.sparse_feature_cols[0][
            'embed_dim']
        self.mlp = MultiLayerPerceptron(
            d_in=mlp_in,
            d_layers=hidden_units,
            dropout=dropout,
            d_out=1
        )

    def forward(self, x):
        dense_input, sparse_inputs = x[:, :len(self.dense_feature_cols)], x[:, len(self.dense_feature_cols):]
        sparse_inputs = sparse_inputs.long()
        sparse_embeds = [self.embed_layers['embed_' + str(i)](sparse_inputs[:, i]) for i in
                         range(sparse_inputs.shape[1])]
        # x = torch.cat(sparse_embeds, axis=-1)
        x = torch.stack(sparse_embeds, dim=1)
        # x = torch.cat([sparse_embeds, dense_input], axis=-1)
        # Wide
        wide_out = self.linear(dense_input)

        # cin Network
        cin_out = self.cin(x)

        # senet Network
        # senet_out = self.senet(x)
        # 2，interaction
        se_embedding = self.se_attention(x)
        ffm_out = self.bilinear(x)
        se_ffm_out = self.bilinear(se_embedding)
        x_interaction = torch.flatten(torch.cat([ffm_out, se_ffm_out], dim=1), start_dim=1)

        # 3，mlp
        x_deep = self.mlp(x_interaction)
        #  Concatenate
        # total_x = torch.cat([cin_out, senet_out, wide_out], axis=-1)
        total_x = cin_out + wide_out + x_deep

        # out
        outputs = F.sigmoid(total_x)
        outputs = outputs.view(outputs.shape[0], )
        return outputs
