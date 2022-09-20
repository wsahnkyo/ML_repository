import torch
from torch import nn
from itertools import combinations


class NumEmbedding(nn.Module):
    """
    连续特征用linear层编码
    输入shape: [batch_size,num_features, d_in], # d_in 通常是1
    输出shape: [batch_size,num_features, d_out]
    """

    def __init__(self, n: int, d_in: int, d_out: int, bias: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(n, d_in, d_out))
        self.bias = nn.Parameter(torch.Tensor(n, d_out)) if bias else None
        with torch.no_grad():
            for i in range(n):
                layer = nn.Linear(d_in, d_out)
                self.weight[i] = layer.weight.T
                if self.bias is not None:
                    self.bias[i] = layer.bias

    def forward(self, x_num):
        assert x_num.ndim == 3
        # x = x_num[..., None] * self.weight[None]
        # x = x.sum(-2)
        x = torch.einsum("bfi,fij->bfj", x_num, self.weight)
        if self.bias is not None:
            x = x + self.bias[None]
        return x


class CatEmbedding(nn.Module):
    """
    离散特征用Embedding层编码
    输入shape: [batch_size, num_features],
    输出shape: [batch_size, num_features, d_embed]
    """

    def __init__(self, categories, d_embed):
        super().__init__()
        self.embedding = nn.Embedding(sum(categories), d_embed)
        self.offsets = nn.Parameter(
            torch.tensor([0] + categories[:-1]).cumsum(0), requires_grad=False)

        nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x_cat):
        """
        x_cat: Long tensor of size ``(batch_size, features_num)``
        """
        x = x_cat + self.offsets[None]
        return self.embedding(x)


class CatLinear(nn.Module):
    """
    离散特征用Embedding实现线性层（等价于先F.onehot再nn.Linear()）
    输入shape: [batch_size, num_features ],
    输出shape: [batch_size, d_out]
    """

    def __init__(self, categories, d_out=1):
        super().__init__()
        self.fc = nn.Embedding(sum(categories), d_out)
        self.bias = nn.Parameter(torch.zeros((d_out,)))
        self.offsets = nn.Parameter(
            torch.tensor([0] + categories[:-1]).cumsum(0), requires_grad=False)
        nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x_cat):
        """
        Long tensor of size ``(batch_size, num_features)``
        """
        x = x_cat + self.offsets[None]
        return torch.sum(self.fc(x), dim=1) + self.bias


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


# mlp
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


# fibinet
class FiBiNET(nn.Module):

    def __init__(self,
                 d_numerical,
                 categories,
                 d_embed,
                 mlp_layers,
                 mlp_dropout,
                 reduction_ratio=3,
                 bilinear_type="field_interaction",
                 n_classes=1):

        super().__init__()

        if d_numerical is None:
            d_numerical = 0
        if categories is None:
            categories = []

        self.categories = categories
        self.n_classes = n_classes

        self.num_linear = nn.Linear(d_numerical, n_classes) if d_numerical else None
        self.cat_linear = CatLinear(categories, n_classes) if categories else None

        self.num_embedding = NumEmbedding(d_numerical, 1, d_embed) if d_numerical else None
        self.cat_embedding = CatEmbedding(categories, d_embed) if categories else None

        num_fields = d_numerical + len(categories)

        self.se_attention = SENetAttention(num_fields, reduction_ratio)
        self.bilinear = BilinearInteraction(num_fields, d_embed, bilinear_type)

        mlp_in = num_fields * (num_fields - 1) * d_embed
        self.mlp = MultiLayerPerceptron(
            d_in=mlp_in,
            d_layers=mlp_layers,
            dropout=mlp_dropout,
            d_out=n_classes
        )

    def forward(self, x):
        """
        x_num: numerical features
        x_cat: category features
        """
        x_num, x_cat = x

        # 一，wide部分
        x_linear = 0.0
        if self.num_linear:
            x_linear = x_linear + self.num_linear(x_num)
        if self.cat_linear:
            x_linear = x_linear + self.cat_linear(x_cat)

        # 二，deep部分

        # 1，embedding
        x_embedding = []
        if self.num_embedding:
            x_embedding.append(self.num_embedding(x_num[..., None]))
        if self.cat_embedding:
            x_embedding.append(self.cat_embedding(x_cat))
        x_embedding = torch.cat(x_embedding, dim=1)

        # 2，interaction
        se_embedding = self.se_attention(x_embedding)
        ffm_out = self.bilinear(x_embedding)
        se_ffm_out = self.bilinear(se_embedding)
        x_interaction = torch.flatten(torch.cat([ffm_out, se_ffm_out], dim=1), start_dim=1)

        # 3，mlp
        x_deep = self.mlp(x_interaction)

        # 三，高低融合
        x_out = x_linear + x_deep
        if self.n_classes == 1:
            x_out = x_out.squeeze(-1)
        return x_out


##测试 FiBiNET

model = FiBiNET(d_numerical=3, categories=[4, 3, 2],
                d_embed=4, mlp_layers=[20, 20], mlp_dropout=0.25,
                reduction_ratio=3,
                bilinear_type="field_interaction",
                n_classes=1)

x_num = torch.randn(2, 3)
x_cat = torch.randint(0, 2, (2, 3))
print(model((x_num, x_cat)))
