# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F


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
            fc_input_dim += input_dim
        self.fc = torch.nn.Linear(fc_input_dim, 1)

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
            x = F.relu(self.conv_layers[i](x))
            h = x
            xs.append(x)
        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))
