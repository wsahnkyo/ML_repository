from collections import namedtuple
from tensorflow import keras

DEFAULT_GROUP_NAME = "default_group"


# 构建输入层
# 将输入的数据转换成字典的形式，定义输入层的时候让输入层的name和字典中特征的key一致，就可以使得输入的数据和对应的Input层对应
def build_input_layers(feature_columns):
    """构建Input层字典，并以dense和sparse两类字典的形式返回"""
    dense_input_dict, sparse_input_dict = {}, {}
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            sparse_input_dict[fc.name] = keras.layers.Input(shape=(1,), name=fc.name, dtype=fc.dtype)
        elif isinstance(fc, DenseFeat):
            dense_input_dict[fc.name] = keras.layers.Input(shape=(fc.dimension,), name=fc.name, dtype=fc.dtype)
    return dense_input_dict, sparse_input_dict


# 构建embedding层
def build_embedding_layers(feature_columns, input_layer_dict, is_linear):
    # 定义一个embedding层对应的字典
    embedding_layers_dict = dict()

    # 将特征中的sparse特征筛选出来
    sparse_features_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if feature_columns else []

    # 如果是用于线性部分的embedding层，其维度是1，否则维度是自己定义的embedding维度
    if is_linear:
        for fc in sparse_features_columns:
            embedding_layers_dict[fc.name] = keras.layers.Embedding(fc.vocabulary_size, 1, name='1d_emb_' + fc.name)
    else:
        for fc in sparse_features_columns:
            embedding_layers_dict[fc.name] = keras.layers.Embedding(fc.vocabulary_size, fc.embedding_dim,
                                                                    name='kd_emb_' + fc.name)

    return embedding_layers_dict


# 将所有的sparse特征embedding拼接
def concat_embedding_list(feature_columns, input_layer_dict, embedding_layer_dict, flatten=False):
    # 将sparse特征筛选出来
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))

    embedding_list = []
    for fc in sparse_feature_columns:
        _input = input_layer_dict[fc.name]  # 获取输入层
        _embed = embedding_layer_dict[fc.name]  # B x 1 x dim  获取对应的embedding层
        embed = _embed(_input)  # B x dim  将input层输入到embedding层中

        # 是否需要flatten, 如果embedding列表最终是直接输入到Dense层中，需要进行Flatten，否则不需要
        if flatten:
            embed = keras.layers.Flatten()(embed)

        embedding_list.append(embed)

    return embedding_list


def get_dnn_output(dnn_input, hidden_units=[1024, 512, 256], dropout=0.3, activation='relu'):
    # 建立dnn_network
    dnn_network = [keras.layers.Dense(units=unit, activation=activation) for unit in hidden_units]
    dropout = keras.layers.Dropout(dropout)

    # 前向传播
    x = dnn_input
    for dnn in dnn_network:
        x = dropout(dnn(x))

    return x


# 得到线性部分的计算结果, 即线性部分计算的前向传播逻辑
def get_linear_logits(dense_input_dict, sparse_input_dict, linear_feature_columns):
    """
    线性部分的计算，所有特征的Input层，然后经过一个全连接层线性计算结果logits
    即FM线性部分的那块计算w1x1+w2x2+...wnxn + b,只不过，连续特征和离散特征这里的线性计算还不太一样
        连续特征由于是数值，可以直接过全连接，得到线性这边的输出。
        离散特征需要先embedding得到1维embedding，然后直接把这个1维的embedding相加就得到离散这边的线性输出。
    :param dense_input_dict: A dict. 连续特征构建的输入层字典 形式{'dense_name': Input(shape, name, dtype)}
    :param sparse_input_dict: A dict. 离散特征构建的输入层字典 形式{'sparse_name': Input(shape, name, dtype)}
    :param linear_feature_columns: A list. 里面的每个元素是namedtuple(元组的一种扩展类型，同时支持序号和属性名访问组件)类型，表示的是linear数据的特征封装版
    """
    # 把所有的dense特征合并起来,经过一个神经元的全连接，做的计算  w1x1 + w2x2 + w3x3....wnxn
    concat_dense_inputs = keras.layers.Concatenate(axis=1)(list(dense_input_dict.values()))
    dense_logits_output = keras.layers.Dense(1)(concat_dense_inputs)

    # 获取linear部分sparse特征的embedding层，这里使用embedding的原因：
    # 对于linear部分直接将特征进行OneHot然后通过一个全连接层，当维度特别大的时候，计算比较慢
    # 使用embedding层的好处就是可以通过查表的方式获取到非零元素对应的权重，然后将这些权重相加，提升效率
    linear_embedding_layers = build_embedding_layers(linear_feature_columns, sparse_input_dict, is_linear=True)

    # 将一维的embedding拼接，注意这里需要一个Flatten层， 使维度对应
    sparse_1d_embed = []
    for fc in linear_feature_columns:
        # 离散特征要进行embedding
        if isinstance(fc, SparseFeat):
            # 找到对应Input层，然后后面接上embedding层
            feat_input = sparse_input_dict[fc.name]
            embed = keras.layers.Flatten()(linear_embedding_layers[fc.name](feat_input))
            sparse_1d_embed.append(embed)

    # embedding中查询得到的权重就是对应onehot向量中一个位置的权重，所以后面不用再接一个全连接了，本身一维的embedding就相当于全连接
    # 只不过是这里的输入特征只有0和1，所以直接向非零元素对应的权重相加就等同于进行了全连接操作(非零元素部分乘的是1)
    sparse_logits_output = keras.layers.Add()(sparse_1d_embed)

    # 最终将dense特征和sparse特征对应的logits相加，得到最终linear的logits
    linear_part = keras.layers.Add()([dense_logits_output, sparse_logits_output])

    return linear_part

# 统一输入
# SparseFeat继承了namedtuple, 通过__new__方法中设置的参数，实现对namedtuple中某些字段的初始化
class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'dtype', 'embedding_name',
                             'group_name'])):
    # 它的作用是阻止在实例化类时为实例分配dict，默认情况下每个类都会有一个dict, 通过__dict__访问，这个dict维护了这个实例的所有属性
    # 当需要创建大量的实例时，创建大量的__dict__会浪费大量的内存，所以这里使用__slots__()进行限制，当然如果需要某些属性被访问到，需要
    # 在__slots__()中将对应的属性填写进去
    __slots__ = ()

    # new方法是在__init__方法之前运行的，new方法的返回值是类的实例，也就是类中的self
    # new方法中传入的参数是cls,而init的方法传入的参数是self
    # __new__ 负责对象的创建，__init__ 负责对象的初始化
    # 这里使用__new__的原因是，这里最终是想创建一个namedtuple对象，并且避免namedtuple初始化时需要填写所有参数的的情况，使用了一个类来包装
    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, dtype="int32", embedding_name=None,
                group_name=DEFAULT_GROUP_NAME):
        if embedding_name is None:
            embedding_name = name
        if embedding_dim == "auto":
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))  # 如果没有指定embedding_dim的一个默认值，这个默认值是怎么来的？
        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, dtype,
                                              embedding_name, group_name)

    # 要想使用自定义的类作为字典的键，就需要重写类的哈希函数，否则无法将其作为字典的键来使用
    # 由于这个类不需要比较大小所以不必重写__eq__()方法
    def __hash__(self):
        return self.name.__hash__()


# 数值特征，这里需要注意，数值特征不一定只是一维的，也可以是一个多维的
class DenseFeat(namedtuple('DenseFeat', ['name', 'dimension', 'dtype'])):
    __slots__ = ()

    def __new__(cls, name, dimension=1, dtype="float32"):
        return super(DenseFeat, cls).__new__(cls, name, dimension, dtype)

    def __hash__(self):
        return self.name.__hash__()


# 长度变化的稀疏特征，其实就是id序列特征
class VarLenSparseFeat(namedtuple('VarLenSparseFeat',
                                  ['sparsefeat', 'maxlen', 'combiner', 'length_name', 'weight_name', 'weight_norm'])):
    __slots__ = ()

    def __new__(cls, sparsefeat, maxlen, combiner="mean", length_name=None, weight_name=None, weight_norm=True):
        return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, maxlen, combiner, length_name, weight_name,
                                                    weight_norm)

    # 由于这里传进来的sparsefeat, 本身就是一个自定义的类型，也有很多有用的信息，例如name, embedding_dim等等
    # 对于VarLenSparseFeat类来说，只不过是一个sparsefeat的序列，需要获取到sparsefeat的相关属性

    # 使用@property装饰器，将一个函数的返回值作为类的属性来使用
    @property
    def name(self):
        return self.sparsefeat.name

    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def use_hash(self):
        return self.sparsefeat.use_hash

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name

    @property
    def group_name(self):
        return self.sparsefeat.group_name

    def __hash__(self):
        return self.name.__hash__()