import numpy as np
import pandas as pd
import datetime
import torch.torch_version
import torch.nn as nn
import warnings

from sklearn.metrics import roc_auc_score
from sklearn import metrics
from torch.utils.data import DataLoader, TensorDataset
from ML_Pytorch.models.AFM import AFM
from ML_Pytorch.models.DCN import DCN
from ML_Pytorch.models.DeepCrossing import DeepCrossing
from ML_Pytorch.models.DeepFM import DeepFM
from ML_Pytorch.models.FM import FM
from ML_Pytorch.models.NFM import NFM
from ML_Pytorch.models.PNN import PNN
from ML_Pytorch.models.WideDeep2 import WideDeep as WideDeep2
from ML_Pytorch.models.WideDeep import WideDeep
from ML_Pytorch.models.WideDeepAttention import WideDeepAttention
import logging.config

logging.config.fileConfig('../logging.conf')
logger = logging.getLogger('train')
val_logger = logging.getLogger('val')

warnings.filterwarnings('ignore')


class ML_General():

    def __init__(self, hidden_units=[256, 128], dropout=0., embedding_dim=40, epochs=50, batch_size=64,
                 dataset_path=None, model_name='WideDeep'):
        val_logger.info(
            "hidden_units:{}, dropout:{}, embedding_dim:{}, epochs:{}, batch_size:{}, model_name:{} ".format(
                hidden_units,
                dropout,
                embedding_dim, epochs,
                batch_size,
                model_name))

        self.batch_size = batch_size
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.epochs = epochs
        self.dataset_path = dataset_path
        self.embedding_dim = embedding_dim
        # 读入训练集，验证集和测试集
        train = pd.read_csv(dataset_path + "/train.csv")
        val = pd.read_csv(dataset_path + "/val.csv")
        test = pd.read_csv(dataset_path + "/test.csv")
        logger.info("train count {}".format(train['Column1'].count()))
        logger.info("val count {}".format(val['Column1'].count()))
        logger.info("test count {}".format(test['Column1'].count()))
        val_logger.info("train count {}".format(train['Column1'].count()))
        val_logger.info("val count {}".format(val['Column1'].count()))
        trn_x, trn_y = train.drop(columns='Column1').values, train['Column1'].values
        val_x, val_y = val.drop(columns='Column1').values, val['Column1'].values
        test_x, test_y = test.drop(columns='Column1').values, test['Column1'].values
        fea_col = np.load(dataset_path + '/fea_col.npy', allow_pickle=True)
        dl_train_dataset = TensorDataset(torch.tensor(trn_x).float(), torch.tensor(trn_y).float())
        dl_val_dataset = TensorDataset(torch.tensor(val_x).float(), torch.tensor(val_y).float())
        dl_train = DataLoader(dl_train_dataset, shuffle=True, batch_size=self.batch_size)
        dl_val = DataLoader(dl_val_dataset, shuffle=True, batch_size=self.batch_size)
        self.dl_train = dl_train
        self.dl_val = dl_val
        self.fea_col = fea_col
        self.model = self.model(model_name)
        self.loss_func = nn.BCELoss()
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.001, weight_decay=0.001)
        self.test_x = torch.tensor(test_x).float()
        self.test_y = torch.tensor(test_y).float()

    def auc(self, y_pred, y_true):
        pred = y_pred.data
        y = y_true.data
        return roc_auc_score(y, pred)

    def model(self, model_name):

        if model_name == 'WideDeep':
            return WideDeep(feature_columns=self.fea_col, hidden_units=self.hidden_units, dropout=self.dropout)
        if model_name == 'WideDeep2':
            return WideDeep2(feature_columns=self.fea_col, hidden_units=self.hidden_units, dropout=self.dropout)
        if model_name == 'WideDeepAttention':
            return WideDeepAttention(feature_columns=self.fea_col, hidden_units=self.hidden_units,
                                     embedding_dim=self.embedding_dim,
                                     dropout=self.dropout)
        if model_name == 'NFM':
            return NFM(feature_columns=self.fea_col, hidden_units=self.hidden_units, dropout=self.dropout)
        if model_name == 'DCN':
            return DCN(feature_columns=self.fea_col, hidden_units=self.hidden_units, dropout=self.dropout,
                       layer_num=3)
        if model_name == 'PNN':
            return PNN(feature_columns=self.fea_col, hidden_units=self.hidden_units, dropout=self.dropout)
        if model_name == 'DeepFM':
            return DeepFM(feature_columns=self.fea_col, hidden_units=self.hidden_units, dropout=self.dropout)
        if model_name == 'DeepCrossing':
            return DeepCrossing(feature_columns=self.fea_col, hidden_units=self.hidden_units,
                                dropout=self.dropout, embedding_dim=self.embedding_dim)
        if model_name == 'FM':
            return FM(feature_columns=self.fea_col)
        if model_name == 'AFM':
            return AFM(feature_columns=self.fea_col, mode="avg", hidden_units=self.hidden_units, dropout=self.dropout)

    def train(self):

        # 模型的相关设置
        logger.info('Model {} is start_training.........'.format(self.model.__class__.__name__))
        val_logger.info('Model  is {}'.format(self.model.__class__.__name__))

        for epoch in range(1, self.epochs + 1):
            logger.info('========' * 8 + '%s' % datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            # 训练阶段
            self.model.train()
            loss_sum = 0.0
            metric_sum = 0.0
            acc_sum = 0.0
            step = 1
            for step, (features, labels) in enumerate(self.dl_train, 1):
                # 梯度清零
                self.optimizer.zero_grad()
                # 正向传播
                predictions = self.model(features);
                loss = self.loss_func(predictions, labels)
                try:
                    metric = self.auc(predictions, labels)
                    y_pred = torch.where(predictions > 0.5, torch.ones_like(predictions), torch.zeros_like(predictions))
                    acc = metrics.accuracy_score(labels.data, y_pred.data)
                except ValueError as err:
                    logger.error(err)
                    pass
                # 反向传播
                loss.backward()
                self.optimizer.step()

                # 打印batch级别日志
                loss_sum += loss.item()
                metric_sum += metric.item()
                acc_sum += acc.item()
            if (epoch % 10 == 0):
                self.validation()
                val_logger.info(
                    "epoch:{}, loss:{}, auc:{}, acc:{}".format(epoch, (loss_sum / step), (metric_sum / step),
                                                               (acc_sum / step)))
            logger.info("epoch:{}, loss:{}, auc:{}, acc:{}".format(epoch, (loss_sum / step), (metric_sum / step),
                                                                   (acc_sum / step)))
        logger.info('Finished Training')

    def validation(self):
        self.model.eval()
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_acc_sum = 0.0
        val_step = 1

        for val_step, (features, labels) in enumerate(self.dl_val, 1):
            with torch.no_grad():
                predictions = self.model(features)
                val_loss = self.loss_func(predictions, labels)
                try:
                    val_metric = self.auc(predictions, labels)
                    y_pred = torch.where(predictions > 0.5, torch.ones_like(predictions),
                                         torch.zeros_like(predictions))
                    val_acc = metrics.accuracy_score(labels.data, y_pred.data)
                except ValueError:
                    pass

            val_loss_sum += val_loss.item()
            val_metric_sum += val_metric.item()
            val_acc_sum += val_acc.item()

        val_logger.info("val loss:{}, val auc:{}, val acc:{}".format(val_loss_sum / val_step, val_metric_sum / val_step,
                                                                     val_acc_sum / val_step))

    def test(self):
        y_pred_probs = self.model(self.test_x)
        y_pred = torch.where(y_pred_probs > 0.5, torch.ones_like(y_pred_probs), torch.zeros_like(y_pred_probs))
        val_logger.info("test loss:{}, test auc:{}, test acc:{}".format(self.loss_func(y_pred_probs, y_pred),
                                                                        self.auc(y_pred, self.test_y),
                                                                        metrics.accuracy_score(self.test_y.data,
                                                                                               y_pred.data)))

    def save(self):
        torch.save(self.model.state_dict(), self.dataset_path + "/model_parameter.pkl")


if __name__ == '__main__':

    # models = ['AFM','DCN','DeepCrossing','DeepFM','FFM','FM','NFM','PNN','WideDeep', 'WideDeep2', 'WideDeepAttention']
    models = ['AFM', 'DCN', 'DeepCrossing', 'DeepFM', 'FM', 'NFM', 'PNN', 'WideDeep', 'WideDeep2', 'WideDeepAttention']
    for model in models:
        ml = ML_General(dataset_path="./data/preprocessed_data", batch_size=64, dropout=0.9, embedding_dim=10,
                        epochs=1, model_name=model)
        ml.train()
    # ml.save()

    # ml.model.load_state_dict(torch.load("D:/DataSet/model_parameter.pkl"))

    # ml.test()
    # ml.save()
    # ml.test()

    # ml.model.load_state_dict(torch.load("D:/DataSet/model_parameter.pkl"))
    # ml.model.forward(input)  # 进行使用
    # DataLoadAndPreprocessing.create_cretio_data(embed_dim=10, test_size=0.2, val_size=0.2, path="./data")
