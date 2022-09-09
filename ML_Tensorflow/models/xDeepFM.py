# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.regularizers import l2

from ML_Tensorflow.utils import DenseFeat, SparseFeat, VarLenSparseFeat, concat_embedding_list, get_dnn_output,build_input_layers, get_linear_logits, build_embedding_layers


class CIN(keras.layers.Layer):
    def __init__(self, cin_size, l2_reg=1e-4):
        """
        :param: cin_size: A list. [H_1, H_2, ....H_T], a list of number of layers
        """
        super(CIN, self).__init__()
        self.cin_size = cin_size
        self.l2_reg = l2_reg

    def build(self, input_shape):
        # input_shape  [None, field_nums, embedding_dim]
        self.field_nums = input_shape[1]

        # CIN 的每一层大小，这里加入第0层，也就是输入层H_0
        self.field_nums = [self.field_nums] + self.cin_size

        # 过滤器
        self.cin_W = {
            'CIN_W_' + str(i): self.add_weight(
                name='CIN_W_' + str(i),
                shape=(1, self.field_nums[0] * self.field_nums[i], self.field_nums[i + 1]),  # 这个大小要理解
                initializer='random_uniform',
                regularizer=l2(self.l2_reg),
                trainable=True
            )
            for i in range(len(self.field_nums) - 1)
        }

        super(CIN, self).build(input_shape)

    def call(self, inputs):
        # inputs [None, field_num, embed_dim]
        embed_dim = inputs.shape[-1]
        hidden_layers_results = [inputs]

        # 从embedding的维度把张量一个个的切开,这个为了后面逐通道进行卷积，算起来好算
        # 这个结果是个list， list长度是embed_dim, 每个元素维度是[None, field_nums[0], 1]  field_nums[0]即输入的特征个数
        # 即把输入的[None, field_num, embed_dim]，切成了embed_dim个[None, field_nums[0], 1]的张量
        split_X_0 = tf.split(hidden_layers_results[0], embed_dim, 2)

        for idx, size in enumerate(self.cin_size):
            # 这个操作和上面是同理的，也是为了逐通道卷积的时候更加方便，分割的是当一层的输入Xk-1
            split_X_K = tf.split(hidden_layers_results[-1], embed_dim,
                                 2)  # embed_dim个[None, field_nums[i], 1] feild_nums[i] 当前隐藏层单元数量

            # 外积的运算
            out_product_res_m = tf.matmul(split_X_0, split_X_K,
                                          transpose_b=True)  # [embed_dim, None, field_nums[0], field_nums[i]]
            out_product_res_o = tf.reshape(out_product_res_m,
                                           shape=[embed_dim, -1, self.field_nums[0] * self.field_nums[idx]])  # 后两维合并起来
            out_product_res = tf.transpose(out_product_res_o,
                                           perm=[1, 0, 2])  # [None, dim, field_nums[0]*field_nums[i]]

            # 卷积运算
            # 这个理解的时候每个样本相当于1张通道为1的照片 dim为宽度， field_nums[0]*field_nums[i]为长度
            # 这时候的卷积核大小是field_nums[0]*field_nums[i]的, 这样一个卷积核的卷积操作相当于在dim上进行滑动，每一次滑动会得到一个数
            # 这样一个卷积核之后，会得到dim个数，即得到了[None, dim, 1]的张量， 这个即当前层某个神经元的输出
            # 当前层一共有field_nums[i+1]个神经元， 也就是field_nums[i+1]个卷积核，最终的这个输出维度[None, dim, field_nums[i+1]]
            cur_layer_out = tf.nn.conv1d(input=out_product_res, filters=self.cin_W['CIN_W_' + str(idx)], stride=1,
                                         padding='VALID')

            cur_layer_out = tf.transpose(cur_layer_out, perm=[0, 2, 1])  # [None, field_num[i+1], dim]

            hidden_layers_results.append(cur_layer_out)

        # 最后CIN的结果，要取每个中间层的输出，这里不要第0层的了
        final_result = hidden_layers_results[1:]  # 这个的维度T个[None, field_num[i], dim]  T 是CIN的网络层数

        # 接下来在第一维度上拼起来  
        result = tf.concat(final_result, axis=1)  # [None, H1+H2+...HT, dim]
        # 接下来， dim维度上加和，并把第三个维度1干掉
        result = tf.reduce_sum(result, axis=-1, keepdims=False)  # [None, H1+H2+..HT]

        return result


def xDeepFM(linear_feature_columns, dnn_feature_columns, cin_size=[128, 128]):
    # 构建输入层，即所有特征对应的Input()层，这里使用字典的形式返回，方便后续构建模型
    dense_input_dict, sparse_input_dict = build_input_layers(linear_feature_columns + dnn_feature_columns)

    # 构建模型的输入层，模型的输入层不能是字典的形式，应该将字典的形式转换成列表的形式
    # 注意：这里实际的输入预Input层对应，是通过模型输入时候的字典数据的key与对应name的Input层
    input_layers = list(dense_input_dict.values()) + list(sparse_input_dict.values())

    # 线性部分的计算逻辑 -- linear
    linear_logits = get_linear_logits(dense_input_dict, sparse_input_dict, linear_feature_columns)

    # 构建维度为k的embedding层，这里使用字典的形式返回，方便后面搭建模型
    # 线性层和dnn层统一的embedding层
    embedding_layer_dict = build_embedding_layers(linear_feature_columns + dnn_feature_columns, sparse_input_dict,
                                                  is_linear=False)

    # DNN侧的计算逻辑 -- Deep
    # 将dnn_feature_columns里面的连续特征筛选出来，并把相应的Input层拼接到一块
    dnn_dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    dnn_dense_feature_columns = [fc.name for fc in dnn_dense_feature_columns]
    dnn_concat_dense_inputs = keras.layers.Concatenate(axis=1)(
        [dense_input_dict[col] for col in dnn_dense_feature_columns])

    # 将dnn_feature_columns里面的离散特征筛选出来，相应的embedding层拼接到一块
    dnn_sparse_kd_embed = concat_embedding_list(dnn_feature_columns, sparse_input_dict, embedding_layer_dict,
                                                flatten=True)
    dnn_concat_sparse_kd_embed = keras.layers.Concatenate(axis=1)(dnn_sparse_kd_embed)

    # DNN层的输入和输出
    dnn_input = keras.layers.Concatenate(axis=1)([dnn_concat_dense_inputs, dnn_concat_sparse_kd_embed])
    dnn_out = get_dnn_output(dnn_input)
    dnn_logits = keras.layers.Dense(1)(dnn_out)

    # CIN侧的计算逻辑， 这里使用的DNN feature里面的sparse部分,这里不要flatten
    exFM_sparse_kd_embed = concat_embedding_list(dnn_feature_columns, sparse_input_dict, embedding_layer_dict,
                                                 flatten=False)
    exFM_input = keras.layers.Concatenate(axis=1)(exFM_sparse_kd_embed)
    exFM_out = CIN(cin_size=cin_size)(exFM_input)
    exFM_logits = keras.layers.Dense(1)(exFM_out)

    # 三边的结果stack
    stack_output = keras.layers.Add()([linear_logits, dnn_logits, exFM_logits])

    # 输出层
    output_layer = keras.layers.Dense(1, activation='sigmoid')(stack_output)

    model = keras.Model(input_layers, output_layer)

    return model
