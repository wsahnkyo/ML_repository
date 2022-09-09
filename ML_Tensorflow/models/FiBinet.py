# -*- coding: utf-8 -*-
import itertools

import tensorflow as tf
from tensorflow import keras
from ML_Tensorflow.utils import DenseFeat, SparseFeat, VarLenSparseFeat, concat_embedding_list, get_dnn_output,build_input_layers, get_linear_logits, build_embedding_layers


class SENETLayer(keras.layers.Layer):
    def __init__(self, reduction_ratio, seed=2021):
        super(SENETLayer, self).__init__()
        self.reduction_ratio = reduction_ratio
        self.seed = seed

    def build(self, input_shape):
        # input_shape  [None, field_nums, embedding_dim]
        self.field_size = input_shape[1]
        self.embedding_size = input_shape[-1]

        # 中间层的神经单元个数 f/r
        reduction_size = max(1, self.field_size // self.reduction_ratio)

        # FC layer1和layer2的参数
        self.W_1 = self.add_weight(shape=(
            self.field_size, reduction_size), initializer=keras.initializers.glorot_normal(seed=self.seed), name="W_1")
        self.W_2 = self.add_weight(shape=(
            reduction_size, self.field_size), initializer=keras.initializers.glorot_normal(seed=self.seed), name="W_2")

        self.tensordot = tf.keras.layers.Lambda(
            lambda x: tf.tensordot(x[0], x[1], axes=(-1, 0)))

        # Be sure to call this somewhere!
        super(SENETLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs [None, field_num, embed_dim]

        # Squeeze -> [None, field_num]
        Z = tf.reduce_mean(inputs, axis=-1)

        # Excitation
        A_1 = tf.nn.relu(self.tensordot([Z, self.W_1]))  # [None, reduction_size]
        A_2 = tf.nn.relu(self.tensordot([A_1, self.W_2]))  # [None, field_num]

        # Re-Weight
        V = tf.multiply(inputs, tf.expand_dims(A_2, axis=2))  # [None, field_num, embedding_dim]

        return V


class BilinearInteraction(keras.layers.Layer):
    """BilinearInteraction Layer used in FiBiNET.
      Input shape
        - 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
        - 3D tensor with shape: ``(batch_size,filed_size*(filed_size-1)/2,embedding_size)``.
    """

    def __init__(self, bilinear_type="interaction", seed=2021, **kwargs):
        super(BilinearInteraction, self).__init__(**kwargs)
        self.bilinear_type = bilinear_type
        self.seed = seed

    def build(self, input_shape):
        # input_shape: [None, field_num, embed_num]
        self.field_size = input_shape[1]
        self.embedding_size = input_shape[-1]

        if self.bilinear_type == "all":  # 所有embedding矩阵共用一个矩阵W
            self.W = self.add_weight(shape=(self.embedding_size, self.embedding_size),
                                     initializer=keras.initializers.glorot_normal(
                                         seed=self.seed), name="bilinear_weight")
        elif self.bilinear_type == "each":  # 每个field共用一个矩阵W
            self.W_list = [self.add_weight(shape=(self.embedding_size, self.embedding_size),
                                           initializer=keras.initializers.glorot_normal(
                                               seed=self.seed), name="bilinear_weight" + str(i)) for i in
                           range(self.field_size - 1)]
        elif self.bilinear_type == "interaction":  # 每个交互用一个矩阵W
            self.W_list = [self.add_weight(shape=(self.embedding_size, self.embedding_size),
                                           initializer=keras.initializers.glorot_normal(
                                               seed=self.seed), name="bilinear_weight" + str(i) + '_' + str(j)) for i, j
                           in
                           itertools.combinations(range(self.field_size), 2)]
        else:
            raise NotImplementedError

        super(BilinearInteraction, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        # inputs: [None, field_nums, embed_dims]
        # 这里把inputs从field_nums处split, 划分成field_nums个embed_dims长向量的列表
        inputs = tf.split(inputs, self.field_size, axis=1)  # [(None, embed_dims), (None, embed_dims), ..] 
        n = len(inputs)  # field_nums个

        if self.bilinear_type == "all":
            # inputs[i] (none, embed_dims)    self.W (embed_dims, embed_dims) -> (None, embed_dims)
            vidots = [tf.tensordot(inputs[i], self.W, axes=(-1, 0)) for i in range(n)]  # 点积
            p = [tf.multiply(vidots[i], inputs[j]) for i, j in itertools.combinations(range(n), 2)]  # 哈达玛积
        elif self.bilinear_type == "each":
            vidots = [tf.tensordot(inputs[i], self.W_list[i], axes=(-1, 0)) for i in range(n - 1)]
            # 假设3个域， 则两两组合[(0,1), (0,2), (1,2)]  这里的vidots是第一个维度， inputs是第二个维度 哈达玛积运算
            p = [tf.multiply(vidots[i], inputs[j]) for i, j in itertools.combinations(range(n), 2)]
        elif self.bilinear_type == "interaction":
            # combinations(inputs, 2)  这个得到的是两两向量交互的结果列表
            # 比如 combinations([[1,2], [3,4], [5,6]], 2)
            # 得到 [([1, 2], [3, 4]), ([1, 2], [5, 6]), ([3, 4], [5, 6])]  (v[0], v[1]) 先v[0]与W点积，然后再和v[1]哈达玛积
            p = [tf.multiply(tf.tensordot(v[0], w, axes=(-1, 0)), v[1])
                 for v, w in zip(itertools.combinations(inputs, 2), self.W_list)]
        else:
            raise NotImplementedError

        output = keras.layers.Concatenate(axis=1)(p)
        return output


def FiBiNet(linear_feature_columns, dnn_feature_columns, bilinear_type='interaction', reduction_ratio=3,
            hidden_units=[128, 128]):
    """
    :param linear_feature_columns, dnn_feature_columns: 封装好的wide端和deep端的特征
    :param bilinear_type: 双线性交互类型， 有'all', 'each', 'interaction'三种
    :param reduction_ratio: senet里面reduction ratio
    :param hidden_units: DNN隐藏单元个数
    """

    # 构建输出层, 即所有特征对应的Input()层， 用字典的形式返回，方便后续构建模型
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

    # 将dnn_feature_columns里面的离散特征筛选出来，相应的embedding层拼接到一块,然后过SENet_layer
    dnn_sparse_kd_embed = concat_embedding_list(dnn_feature_columns, sparse_input_dict, embedding_layer_dict,
                                                flatten=False)
    sparse_embedding_list = keras.layers.Concatenate(axis=1)(dnn_sparse_kd_embed)

    # SENet layer
    senet_embedding_list = SENETLayer(reduction_ratio)(sparse_embedding_list)

    # 双线性交互层
    senet_bilinear_out = BilinearInteraction(bilinear_type=bilinear_type)(senet_embedding_list)
    senet_bilinear_out = keras.layers.Flatten()(senet_bilinear_out)
    raw_bilinear_out = BilinearInteraction(bilinear_type=bilinear_type)(sparse_embedding_list)
    raw_bilinear_out = keras.layers.Flatten()(raw_bilinear_out)

    bilinear_out = keras.layers.Concatenate(axis=1)([senet_bilinear_out, raw_bilinear_out])

    # DNN层的输入和输出
    dnn_input = keras.layers.Concatenate(axis=1)([bilinear_out, dnn_concat_dense_inputs])
    dnn_out = get_dnn_output(dnn_input, hidden_units=hidden_units)
    dnn_logits = keras.layers.Dense(1)(dnn_out)

    # 最后的输出
    final_logits = keras.layers.Add()([linear_logits, dnn_logits])

    # 输出层
    output_layer = keras.layers.Dense(1, activation='sigmoid')(final_logits)

    model = keras.Model(input_layers, output_layer)

    return model
