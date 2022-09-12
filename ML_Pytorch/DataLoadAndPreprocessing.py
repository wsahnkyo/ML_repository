import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def sparsFeature(feat, feat_num, embed_dim=10):
    """
    create dictionary for sparse feature
    :param feat: feature_name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def denseFeature(feat):
    """
    create dictionary for dense feature
    :param feat: dense feature name
    : return
    """
    return {'feat': feat}


# 读入数据集，并进行预处理
def create_cretio_data(embed_dim=10, test_size=0.2, val_size=0.2, path="./data", data_size=0.06):
    try:
        # import data
        data_df = pd.read_csv(path + '/data.csv')
        true_data = data_df[(data_df['Column1'] == 1)]
        false_data = data_df[(data_df['Column1'] == 0)]

        # print(int(true_data['Column1'].count()/100))
        #
        # print(int(false_data['Column1'].count()/100))

        # return
        # print( data_df[(data_df['Column1'] == 1)])


        data_df = pd.concat([false_data.sample(int(false_data['Column1'].count() * data_size)),
                             true_data.sample(int(true_data['Column1'].count() * data_size))])


        # test_df = pd.read_csv('./data/validation_df.csv')

        # 进行数据合并
        # label = train_df['Label']
        # testlabel = test_df['Label']
        # del train_df['Label']
        #
        # data_df = pd.concat((train_df, test_df))
        # del data_df['Id']

        sparse_feas = []
        dense_feas = []
        # 特征分开类别
        for index, col in enumerate(data_df):
            if (index >= 1 and index <= 13):
                dense_feas.append(col)
            if (index >= 14 and index <= 39):
                sparse_feas.append(col)

        # data_df[dense_feas] = data_df[dense_feas] + 2

        # # 填充缺失值
        data_df[sparse_feas] = data_df[sparse_feas].fillna('-1')
        data_df[dense_feas] = data_df[dense_feas].fillna(0)

        # # 把特征列保存成字典, 方便类别特征的处理工作
        feature_columns = [[denseFeature(feat) for feat in dense_feas]] + [
            [sparsFeature(feat, len(data_df[feat].unique()), embed_dim=embed_dim) for feat in sparse_feas]]
        np.save(path + '/preprocessed_data/fea_col.npy', feature_columns)

        #
        # 数据预处理
        # 进行编码  类别特征编码
        for feat in sparse_feas:
            le = LabelEncoder()
            data_df[feat] = le.fit_transform(data_df[feat])
        # 数值特征归一化
        for feat in dense_feas:
            mms = MinMaxScaler()
            data_df[feat] = mms.fit_transform(data_df[feat].values.reshape(-1, 1))
        # 数值特征归一化
        # mms = MinMaxScaler()
        # data_df[dense_feas] = mms.fit_transform(data_df[dense_feas])

        # # 分开测试集和训练集
        # train = data_df[:train_df.shape[0]]
        # test = data_df[train_df.shape[0]:]
        #
        # train['Label'] = label
        #

        val_true_data = data_df[(data_df['Column1'] == 1)]
        val_false_data = data_df[(data_df['Column1'] == 0)]

        # print(int(true_data['Column1'].count()/100))
        #
        # print(int(false_data['Column1'].count()/100))

        # return
        # print( data_df[(data_df['Column1'] == 1)])

        val_df = pd.concat([val_false_data.sample(int(val_false_data['Column1'].count() * val_size)),
                             val_true_data.sample(int(val_true_data['Column1'].count() * val_size))])

        data_df.drop(val_df.index, inplace=True)

        #
        # test_df = data_df.concat([false_data.sample(int(false_data['Column1'].count() * test_size)),
        #                           true_data.sample(int(true_data['Column1'].count() * test_size))])

        # # 划分测试集
        # train_set, test_set = train_test_split(data_df, test_size=test_size, random_state=2020)
        # # 划分验证集
        # train_set, val_set = train_test_split(train_set, test_size=val_size, random_state=2020)

        data_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)
        val_df.reset_index(drop=True, inplace=True)

        data_df.to_csv(path + '/preprocessed_data/train.csv', index=0)
        val_df.to_csv(path + '/preprocessed_data/test.csv', index=0)
        val_df.to_csv(path + '/preprocessed_data/val.csv', index=0)
    except Exception as e:
        print(e)


create_cretio_data(path="./data", data_size=1)
# fea_col = np.load('preprocessed_data/' + 'fea_col.npy', allow_pickle=True)
# print(fea_col)
