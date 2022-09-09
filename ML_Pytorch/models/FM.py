import torch

import torch.nn as nn
import torch.nn.functional as F

import warnings

warnings.filterwarnings('ignore')


class FM(nn.Module):
    """FM part"""

    def __init__(self, feature_columns):
        """
        latent_dim: 各个离散特征隐向量的维度
        input_shape: 这个最后离散特征embedding之后的拼接和dense拼接的总特征个数
        """
        super(FM, self).__init__()
        self.dense_feature_cols, self.sparse_feature_cols = feature_columns
        # embedding
        self.embed_layers = nn.ModuleDict({
            'embed_' + str(i): nn.Embedding(num_embeddings=feat['feat_num'], embedding_dim=feat['embed_dim'])
            for i, feat in enumerate(self.sparse_feature_cols)
        })
        self.fea_num = len(self.dense_feature_cols) + len(self.sparse_feature_cols) * self.sparse_feature_cols[0][
            'embed_dim']
        # 定义三个矩阵， 一个是全局偏置，一个是一阶权重矩阵， 一个是二阶交叉矩阵，注意这里的参数由于是可学习参数，需要用nn.Parameter进行定义
        self.w0 = nn.Parameter(torch.zeros([1, ]))
        self.w1 = nn.Parameter(torch.rand([self.fea_num, 1]))
        self.w2 = nn.Parameter(torch.rand([self.fea_num, self.sparse_feature_cols[0]['embed_dim']]))

    def forward(self, x):
        dense_inputs, sparse_inputs = x[:, :len(self.dense_feature_cols)], x[:, len(self.dense_feature_cols):]
        sparse_inputs = sparse_inputs.long()  # 转成long类型才能作为nn.embedding的输入
        sparse_embeds = [self.embed_layers['embed_' + str(i)](sparse_inputs[:, i]) for i in
                         range(sparse_inputs.shape[1])]
        sparse_embeds = torch.cat(sparse_embeds, dim=-1)
        # 把离散特征和连续特征进行拼接作为FM和DNN的输入
        x = torch.cat([sparse_embeds, dense_inputs], dim=-1)
        # 一阶交叉
        first_order = self.w0 + torch.mm(x, self.w1)  # (samples_num, 1)
        # 二阶交叉  这个用FM的最终化简公式
        second_order = 1 / 2 * torch.sum(
            torch.pow(torch.mm(x, self.w2), 2) - torch.mm(torch.pow(x, 2), torch.pow(self.w2, 2)),
            dim=1,
            keepdim=True
        )  # (samples_num, 1)
        # 模型的最后输出
        outputs = F.sigmoid(first_order + second_order)
        outputs = outputs.view(outputs.shape[0], )
        return outputs
