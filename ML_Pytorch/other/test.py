import numpy as np
import pandas as pd
import datetime

from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from models.FiBiNET import FiBiNET


def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "==========" * 8 + "%s" % nowtime)
    print(info + '...\n\n')


from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

dfdata = pd.read_csv("D:\\train_1m.txt\\train_1m.txt", sep="\t", header=None)
dfdata.columns = ["label"] + ["I" + str(x) for x in range(1, 14)] + [
    "C" + str(x) for x in range(14, 40)]

cat_cols = [x for x in dfdata.columns if x.startswith('C')]
num_cols = [x for x in dfdata.columns if x.startswith('I')]
num_pipe = Pipeline(steps=[('impute', SimpleImputer()), ('quantile', QuantileTransformer())])

for col in cat_cols:
    dfdata[col] = LabelEncoder().fit_transform(dfdata[col])

dfdata[num_cols] = num_pipe.fit_transform(dfdata[num_cols])

categories = [dfdata[col].max() + 1 for col in cat_cols]

print("aaa")
print("aaa")
import torchkeras
import torch
from torch.utils.data import Dataset, DataLoader


# DataFrame转换成torch数据集Dataset, 特征分割成X_num,X_cat方式
class DfDataset(Dataset):
    def __init__(self, df,
                 label_col,
                 num_features,
                 cat_features,
                 categories,
                 is_training=True):

        self.X_num = torch.tensor(df[num_features].values).float() if num_features else None
        self.X_cat = torch.tensor(df[cat_features].values).long() if cat_features else None
        self.Y = torch.tensor(df[label_col].values).float()
        self.categories = categories
        self.is_training = is_training

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        if self.is_training:
            return ((self.X_num[index], self.X_cat[index]), self.Y[index])
        else:
            return (self.X_num[index], self.X_cat[index])

    def get_categories(self):
        return self.categories


dftrain_val, dftest = train_test_split(dfdata, test_size=0.2)
dftrain, dfval = train_test_split(dftrain_val, test_size=0.2)

ds_train = DfDataset(dftrain, label_col="label", num_features=num_cols, cat_features=cat_cols,
                     categories=categories, is_training=True)

ds_val = DfDataset(dfval, label_col="label", num_features=num_cols, cat_features=cat_cols,
                   categories=categories, is_training=True)

ds_test = DfDataset(dftest, label_col="label", num_features=num_cols, cat_features=cat_cols,
                    categories=categories, is_training=True)

dl_train = DataLoader(ds_train, batch_size=2048, shuffle=True)
dl_val = DataLoader(ds_val, batch_size=2048, shuffle=False)
dl_test = DataLoader(ds_test, batch_size=2048, shuffle=False)

for features, labels in dl_train:
    break


def create_net():
    net = FiBiNET(
        d_numerical=ds_train.X_num.shape[1],
        categories=ds_train.get_categories(),
        d_embed=8, mlp_layers=[128, 64, 32], mlp_dropout=0.25,
        reduction_ratio=3,
        bilinear_type="field_all",
        n_classes=1

    )
    return net


from torchkeras import summary

net = create_net()
import os, sys, time
import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm

import torch
from torch import nn
from accelerate import Accelerator
from copy import deepcopy


def printlog(info):
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "==========" * 8 + "%s" % nowtime)
    print(str(info) + "\n")


class StepRunner:
    def __init__(self, net, loss_fn, stage="train", metrics_dict=None,
                 optimizer=None, lr_scheduler=None,
                 accelerator=None
                 ):
        self.net, self.loss_fn, self.metrics_dict, self.stage = net, loss_fn, metrics_dict, stage
        self.optimizer, self.lr_scheduler = optimizer, lr_scheduler
        self.accelerator = accelerator

    def __call__(self, features, labels):
        # loss
        preds = self.net(features)
        loss = self.loss_fn(preds, labels)

        # backward()
        if self.optimizer is not None and self.stage == "train":
            if self.accelerator is None:
                loss.backward()
            else:
                self.accelerator.backward(loss)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

        # metrics
        step_metrics = {self.stage + "_" + name: metric_fn(preds, labels).item()
                        for name, metric_fn in self.metrics_dict.items()}
        return loss.item(), step_metrics


class EpochRunner:
    def __init__(self, steprunner):
        self.steprunner = steprunner
        self.stage = steprunner.stage
        self.steprunner.net.train() if self.stage == "train" else self.steprunner.net.eval()

    def __call__(self, dataloader):
        total_loss, step = 0, 0
        loop = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, batch in loop:
            features, labels = batch
            if self.stage == "train":
                loss, step_metrics = self.steprunner(features, labels)
            else:
                with torch.no_grad():
                    loss, step_metrics = self.steprunner(features, labels)

            step_log = dict({self.stage + "_loss": loss}, **step_metrics)

            total_loss += loss
            step += 1
            if i != len(dataloader) - 1:
                loop.set_postfix(**step_log)
            else:
                epoch_loss = total_loss / step
                epoch_metrics = {self.stage + "_" + name: metric_fn.compute().item()
                                 for name, metric_fn in self.steprunner.metrics_dict.items()}
                epoch_log = dict({self.stage + "_loss": epoch_loss}, **epoch_metrics)
                loop.set_postfix(**epoch_log)

                for name, metric_fn in self.steprunner.metrics_dict.items():
                    metric_fn.reset()
        return epoch_log


class KerasModel(torch.nn.Module):
    def __init__(self, net, loss_fn, metrics_dict=None, optimizer=None, lr_scheduler=None):
        super().__init__()
        self.accelerator = Accelerator()
        self.history = {}

        self.net = net
        self.loss_fn = loss_fn
        self.metrics_dict = nn.ModuleDict(metrics_dict)

        self.optimizer = optimizer if optimizer is not None else torch.optim.Adam(
            self.parameters(), lr=1e-2)
        self.lr_scheduler = lr_scheduler

        self.net, self.loss_fn, self.metrics_dict, self.optimizer = self.accelerator.prepare(
            self.net, self.loss_fn, self.metrics_dict, self.optimizer)

    def forward(self, x):
        if self.net:
            return self.net.forward(x)
        else:
            raise NotImplementedError

    def fit(self, train_data, val_data=None, epochs=10, ckpt_path='checkpoint.pt',
            patience=5, monitor="val_loss", mode="min"):

        train_data = self.accelerator.prepare(train_data)
        val_data = self.accelerator.prepare(val_data) if val_data else []

        for epoch in range(1, epochs + 1):
            printlog("Epoch {0} / {1}".format(epoch, epochs))

            # 1，train -------------------------------------------------
            train_step_runner = StepRunner(net=self.net, stage="train",
                                           loss_fn=self.loss_fn, metrics_dict=deepcopy(self.metrics_dict),
                                           optimizer=self.optimizer, lr_scheduler=self.lr_scheduler,
                                           accelerator=self.accelerator)
            train_epoch_runner = EpochRunner(train_step_runner)
            train_metrics = train_epoch_runner(train_data)

            for name, metric in train_metrics.items():
                self.history[name] = self.history.get(name, []) + [metric]

            # 2，validate -------------------------------------------------
            if val_data:
                val_step_runner = StepRunner(net=self.net, stage="val",
                                             loss_fn=self.loss_fn, metrics_dict=deepcopy(self.metrics_dict),
                                             accelerator=self.accelerator)
                val_epoch_runner = EpochRunner(val_step_runner)
                with torch.no_grad():
                    val_metrics = val_epoch_runner(val_data)
                val_metrics["epoch"] = epoch
                for name, metric in val_metrics.items():
                    self.history[name] = self.history.get(name, []) + [metric]

            # 3，early-stopping -------------------------------------------------
            arr_scores = self.history[monitor]
            best_score_idx = np.argmax(arr_scores) if mode == "max" else np.argmin(arr_scores)
            if best_score_idx == len(arr_scores) - 1:
                torch.save(self.net.state_dict(), ckpt_path)
                print("<<<<<< reach best {0} : {1} >>>>>>".format(monitor,
                                                                  arr_scores[best_score_idx]), file=sys.stderr)
            if len(arr_scores) - best_score_idx > patience:
                print("<<<<<< {} without improvement in {} epoch, early stopping >>>>>>".format(
                    monitor, patience), file=sys.stderr)
                self.net.load_state_dict(torch.load(ckpt_path))
                break

        return pd.DataFrame(self.history)

    @torch.no_grad()
    def evaluate(self, val_data):
        val_data = self.accelerator.prepare(val_data)
        val_step_runner = StepRunner(net=self.net, stage="val",
                                     loss_fn=self.loss_fn, metrics_dict=deepcopy(self.metrics_dict),
                                     accelerator=self.accelerator)
        val_epoch_runner = EpochRunner(val_step_runner)
        val_metrics = val_epoch_runner(val_data)
        return val_metrics

    @torch.no_grad()
    def predict(self, dataloader):
        dataloader = self.accelerator.prepare(dataloader)
        result = torch.cat([self.forward(t[0]) for t in dataloader])
        return result.data


from torchkeras.metrics import AUC

loss_fn = nn.BCEWithLogitsLoss()

metrics_dict = {"auc": AUC()}

optimizer = torch.optim.Adam(net.parameters(), lr=0.002, weight_decay=0.001)

model = KerasModel(net,
                   loss_fn=loss_fn,
                   metrics_dict=metrics_dict,
                   optimizer=optimizer
                   )
dfhistory = model.fit(train_data=dl_train, val_data=dl_val, epochs=100, patience=5,
                      monitor="val_auc", mode="max", ckpt_path='checkpoint.pt')

# % matplotlib
# inline
# % config
# InlineBackend.figure_format = 'svg'

import matplotlib.pyplot as plt


def plot_metric(dfhistory, metric):
    train_metrics = dfhistory["train_" + metric]
    val_metrics = dfhistory['val_' + metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics, 'bo--')
    plt.plot(epochs, val_metrics, 'ro-')
    plt.title('Training and validation ' + metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_" + metric, 'val_' + metric])
    plt.show()


plot_metric(dfhistory, "loss")
plot_metric(dfhistory, "auc")

from sklearn.metrics import roc_auc_score

preds = torch.sigmoid(model.predict(dl_val))
labels = torch.cat([x[-1] for x in dl_val])

val_auc = roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy())
print(val_auc)

torch.save(model.net.state_dict(), "best_fibinet.pt")
net_clone = create_net()
net_clone.load_state_dict(torch.load("best_fibinet.pt"))

from sklearn.metrics import roc_auc_score

net_clone.eval()
preds = torch.cat([torch.sigmoid(net_clone(x[0])).data for x in dl_val])
labels = torch.cat([x[-1] for x in dl_val])

val_auc = roc_auc_score(labels.cpu().numpy(), preds.cpu().numpy())
print(val_auc)
