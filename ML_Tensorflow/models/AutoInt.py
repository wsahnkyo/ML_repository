# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import  keras

from ML_Tensorflow.utils import DenseFeat, SparseFeat, VarLenSparseFeat, concat_embedding_list, get_dnn_output,build_input_layers, get_linear_logits, build_embedding_layers

class InteractingLayer(keras.layers.Layer):
    """A layer user in AutoInt that model the correction between different feature fields by multi-head self-att mechanism
        input: 3维张量, (none, field_num, embedding_size)
        output: 3维张量, (none, field_num, att_embedding_size * head_num)
    """
    def __init__(self, att_embedding_size=8, head_num=2, use_res=True, seed=2021):
        super(InteractingLayer, self).__init__()
        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.use_res = use_res
        self.seed = seed
        
    
    def build(self, input_shape):
        embedding_size = int(input_shape[-1])
        
        # 定义三个矩阵Wq, Wk, Wv
        self.W_query = self.add_weight(name="query", shape=[embedding_size, self.att_embedding_size * self.head_num],
                                        dtype=tf.float32, initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed))
        self.W_key = self.add_weight(name="key", shape=[embedding_size, self.att_embedding_size * self.head_num], 
                                     dtype=tf.float32, initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed+1))
        self.W_value = self.add_weight(name="value", shape=[embedding_size, self.att_embedding_size * self.head_num],
                                      dtype=tf.float32, initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed+2))
        
        if self.use_res:
            self.W_res = self.add_weight(name="res", shape=[embedding_size, self.att_embedding_size * self.head_num],
                                        dtype=tf.float32, initializer=tf.keras.initializers.TruncatedNormal(seed=self.seed+3))
        
        super(InteractingLayer, self).build(input_shape)
    
    def call(self, inputs):
        # inputs (none, field_nums, embed_num)
        
        querys = tf.tensordot(inputs, self.W_query, axes=(-1, 0))   # (None, field_nums, att_emb_size*head_num)
        keys = tf.tensordot(inputs, self.W_key, axes=(-1, 0))
        values = tf.tensordot(inputs, self.W_value, axes=(-1, 0))
        
        # 多头注意力计算 按照头分开  (head_num, None, field_nums, att_embed_size)
        querys = tf.stack(tf.split(querys, self.head_num, axis=2))
        keys = tf.stack(tf.split(keys, self.head_num, axis=2))
        values = tf.stack(tf.split(values, self.head_num, axis=2))
        
        # Q * K, key的后两维转置，然后再矩阵乘法 
        inner_product = tf.matmul(querys, keys, transpose_b=True)    # (head_num, None, field_nums, field_nums)
        normal_att_scores = tf.nn.softmax(inner_product, axis=-1)
        
        result = tf.matmul(normal_att_scores, values)   # (head_num, None, field_nums, att_embed_size)
        result = tf.concat(tf.split(result, self.head_num, ), axis=-1)  # (1, None, field_nums, att_emb_size*head_num)
        result = tf.squeeze(result, axis=0)  # (None, field_num, att_emb_size*head_num)
        
        if self.use_res:
            result += tf.tensordot(inputs, self.W_res, axes=(-1, 0))
        
        result = tf.nn.relu(result)
        
        return result


def AutoInt(linear_feature_columns, dnn_feature_columns, att_layer_num=3, att_embedding_size=8, att_head_num=2, att_res=True):
    """
    :param att_layer_num: transformer块的数量，一个transformer块里面是自注意力计算 + 残差计算
    :param att_embedding_size:  文章里面的d', 自注意力时候的att的维度
    :param att_head_num: 头的数量或者自注意力子空间的数量
    :param att_res: 是否使用残差网络
    """
    # 构建输入层，即所有特征对应的Input()层，这里使用字典的形式返回，方便后续构建模型
    dense_input_dict, sparse_input_dict = build_input_layers(linear_feature_columns+dnn_feature_columns)
    
    # 构建模型的输入层，模型的输入层不能是字典的形式，应该将字典的形式转换成列表的形式
    # 注意：这里实际的输入预Input层对应，是通过模型输入时候的字典数据的key与对应name的Input层
    input_layers = list(dense_input_dict.values()) + list(sparse_input_dict.values())
    
    # 线性部分的计算逻辑 -- linear
    linear_logits = get_linear_logits(dense_input_dict, sparse_input_dict, linear_feature_columns)
    
    # 构建维度为k的embedding层，这里使用字典的形式返回，方便后面搭建模型
    # 线性层和dnn层统一的embedding层
    embedding_layer_dict = build_embedding_layers(linear_feature_columns+dnn_feature_columns, sparse_input_dict, is_linear=False)
    
    # 构造self-att的输入
    att_sparse_kd_embed = concat_embedding_list(dnn_feature_columns, sparse_input_dict, embedding_layer_dict, flatten=False)
    att_input = keras.layers.Concatenate(axis=1)(att_sparse_kd_embed)   # (None, field_num, embed_num)
    
    # 下面的循环，就是transformer的前向传播，多个transformer块的计算逻辑
    for _ in range(att_layer_num):
        att_input = InteractingLayer(att_embedding_size, att_head_num, att_res)(att_input)
    att_output = keras.layers.Flatten()(att_input)
    #att_logits = Dense(1)(att_output)
    
    # DNN侧的计算逻辑 -- Deep
    # 将dnn_feature_columns里面的连续特征筛选出来，并把相应的Input层拼接到一块
    dnn_dense_feature_columns = list(filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if dnn_feature_columns else []
    dnn_dense_feature_columns = [fc.name for fc in dnn_dense_feature_columns]
    dnn_concat_dense_inputs = keras.layers.Concatenate(axis=1)([dense_input_dict[col] for col in dnn_dense_feature_columns])
    
    # 将dnn_feature_columns里面的离散特征筛选出来，相应的embedding层拼接到一块
    dnn_sparse_kd_embed = concat_embedding_list(dnn_feature_columns, sparse_input_dict, embedding_layer_dict, flatten=True)
    dnn_concat_sparse_kd_embed = keras.layers.Concatenate(axis=1)(dnn_sparse_kd_embed)
    
    # DNN层的输入和输出
    dnn_input = keras.layers.Concatenate(axis=1)([dnn_concat_dense_inputs, dnn_concat_sparse_kd_embed, att_output])
    dnn_out = get_dnn_output(dnn_input)
    dnn_logits = keras.layers.Dense(1)(dnn_out)
     
    # 三边的结果stack
    stack_output = keras.layers.Add()([linear_logits, dnn_logits])
    
    # 输出层
    output_layer = keras.layers.Dense(1, activation='sigmoid')(stack_output)
    
    model = keras.Model(input_layers, output_layer)
    
    return model