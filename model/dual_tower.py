from __future__ import print_function

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from sklearn.metrics import *
from torch.utils.data import DataLoader
from tqdm import tqdm

from preprocessing.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, create_embedding_matrix, \
    get_varlen_pooling_list, build_input_features
from layers.core import PredictionLayer
from preprocessing.utils import slice_arrays
from tensorflow.python.keras.callbacks import CallbackList
from preprocessing.callbacks import History
from preprocessing.utils import contrast_loss
from preprocessing.utils import dual_augmented_loss


class DualTower(nn.Module):
    def __init__(self, user_dnn_feature_columns, item_dnn_feature_columns, l2_reg_embedding=1e-5,
                 init_std=0.0001, seed=1024, task='binary', device='cpu', gpus=None):
        super(DualTower, self).__init__()
        torch.manual_seed(seed)

        self.reg_loss = torch.zeros((1,), device=device)
        self.aux_loss = torch.zeros((1,), device=device)
        self.device = device
        self.gpus = gpus
        if self.gpus and str(self.gpus[0]) not in self.device:
            raise ValueError("`gpus[0]` should be the same gpu with `device`")

        self.feature_index = build_input_features(user_dnn_feature_columns + item_dnn_feature_columns)

        self.user_dnn_feature_columns = user_dnn_feature_columns
        self.user_embedding_dict = create_embedding_matrix(self.user_dnn_feature_columns, init_std,
                                                           sparse=False, device=device)

        self.item_dnn_feature_columns = item_dnn_feature_columns
        self.item_embedding_dict = create_embedding_matrix(self.item_dnn_feature_columns, init_std,
                                                           sparse=False, device=device)

        self.regularization_weight = []
        self.add_regularization_weight(self.user_embedding_dict.parameters(), l2=l2_reg_embedding)
        self.add_regularization_weight(self.item_embedding_dict.parameters(), l2=l2_reg_embedding)

        self.out = PredictionLayer(task,)
        self.to(device)

        # parameters of callbacks
        self._is_graph_network = True  # used for ModelCheckpoint
        self.stop_training = False  # used for EarlyStopping
        self.history = History()



    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, shuffle=True, callbacks=None):
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)

            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]

        elif validation_split and 0 < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))

            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))

        else:
            val_x = []
            val_y = []

        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        train_tensor_data = Data.TensorDataset(torch.from_numpy(
            np.concatenate(x, axis=-1)), torch.from_numpy(y))
        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func

        optim = self.optim

        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

        train_loader = DataLoader(dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        sample_num = len(train_tensor_data)
        steps_per_epoch = (sample_num - 1) // batch_size + 1


        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.set_model(self)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):  # for tf1.4
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False

        # train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        for epoch in range(initial_epoch, epochs):
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            train_result = {}

            with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                for _, (x_train, y_train) in t:
                    x = x_train.to(self.device).float()
                    y = y_train.to(self.device).float()

                    y_au = y_train.to(self.device)

                    out = model(x)
                    # print(out)
                    y_pred,user_embedding,item_embedding,user_augment_vector,item_augment_vector\
                        = out[0],out[1],out[2],out[3],out[4]


                    loss_v,loss_u = dual_augmented_loss(y_au,user_embedding,
                                                        item_embedding,user_augment_vector,item_augment_vector)


                    y_pred = y_pred.squeeze()

                    optim.zero_grad()
                    # print("y",y_pred.shape,y.shape)
                    loss = loss_func(y_pred, y.squeeze(), reduction='sum')
                    reg_loss = self.get_regularization_loss()

                    # total_loss = loss + reg_loss + self.aux_loss + contras
                    total_loss = loss + reg_loss + self.aux_loss + loss_v + loss_u
                    # print(total_loss, contras, loss)

                    loss_epoch += loss.item()
                    total_loss_epoch += total_loss.item()
                    total_loss.backward()
                    optim.step()

                    if verbose > 0:
                        for name, metric_fun in self.metrics.items():
                            if name not in train_result:
                                train_result[name] = []
                            train_result[name].append(metric_fun(
                                y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype('float64')
                            ))

            # add epoch_logs
            epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result = self.evaluate(val_x, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result

            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))

                eval_str = "{0}s - loss: {1: .4f}".format(epoch_time, epoch_logs["loss"])

                for name in self.metrics:
                    eval_str += " - " + name + ": {0: .4f} ".format(epoch_logs[name]) + " - " + \
                                "val_" + name + ": {0: .4f}".format(epoch_logs["val_" + name])
                print(eval_str)
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()
        return self.history

    def evaluate(self, x, y, batch_size=256):
        pred_ans = self.predict(x, batch_size)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            eval_result[name] = metric_fun(y, pred_ans)
        return eval_result

    def predict(self, x, batch_size=256):
        model = self.eval()
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)

        tensor_data = Data.TensorDataset(
            torch.from_numpy(np.concatenate(x, axis=-1))
        )
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size
        )

        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()

                y_pred = model(x)[0].cpu().data.numpy()
                # y_pred = model(x).cpu().data.numpy()
                pred_ans.append(y_pred)

        return np.concatenate(pred_ans).astype("float64")

    def input_from_feature_columns(self, X, feature_columns, embedding_dict, support_dense=True):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []

        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []

        if not support_dense and len(dense_feature_columns) > 0:
            raise ValueError(
                "DenseFeat is not supported in dnn_feature_columns")

        sparse_embedding_list = [embedding_dict[feat.embedding_name](
            X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]].long()) for
            feat in sparse_feature_columns]

        varlen_sparse_embedding_list = get_varlen_pooling_list(embedding_dict, X, self.feature_index,
                                                               varlen_sparse_feature_columns, self.device)

        dense_value_list = [X[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            dense_feature_columns]

        return sparse_embedding_list + varlen_sparse_embedding_list, dense_value_list

    def compute_input_dim(self, feature_columns, include_sparse=True, include_dense=True, feature_group=False):
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        dense_feature_columns = list(
            filter(lambda x: isinstance(x, DenseFeat), feature_columns)) if len(feature_columns) else []

        dense_input_dim = sum(
            map(lambda x: x.dimension, dense_feature_columns))
        if feature_group:
            sparse_input_dim = len(sparse_feature_columns)
        else:
            sparse_input_dim = sum(feat.embedding_dim for feat in sparse_feature_columns)
        input_dim = 0
        if include_sparse:
            input_dim += sparse_input_dim
        if include_dense:
            input_dim += dense_input_dim
        return input_dim

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self):
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss

    def add_auxiliary_loss(self, aux_loss, alpha):
        self.aux_loss = aux_loss * alpha




    def compile(self, optimizer, loss=None, metrics=None,lr = 0.01):
        self.metrics_names = ["loss"]
        self.optim = self._get_optim(optimizer,lr)
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)

    def _get_optim(self, optimizer,lr):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr= lr)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters(),lr= lr)  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters())  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters())
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim

    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            if loss == "binary_crossentropy":
                loss_func = F.binary_cross_entropy
            elif loss == "mse":
                loss_func = F.mse_loss
            elif loss == "mae":
                loss_func = F.l1_loss
            else:
                raise NotImplementedError
        else:
            loss_func = loss
        return loss_func

    def _log_loss(self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
        # change eps to improve calculation accuracy
        return log_loss(y_true,
                        y_pred,
                        eps,
                        normalize,
                        sample_weight,
                        labels)

    def _get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    # metrics_[metric] = lambda y_true, y_pred: accuracy_score(
                    #     y_true, np.where(y_pred > 0.5, 1, 0))
                    metrics_[metric] = lambda y_true, y_pred: accuracy_score(
                        y_true, np.where(y_pred > 0.5, 1, 0))
                self.metrics_names.append(metric)
        return metrics_

    @property
    def embedding_size(self):
        feature_columns = self.dnn_feature_columns
        sparse_feature_columns = list(
            filter(lambda x: isinstance(x, (SparseFeat, VarLenSparseFeat)), feature_columns)) if len(
            feature_columns) else []
        embedding_size_set = set([feat.embedding_dim for feat in sparse_feature_columns])
        if len(embedding_size_set) > 1:
            raise ValueError("embedding_dim of SparseFeat and VarlenSparseFeat must be same in this model!")
        return list(embedding_size_set)[0]