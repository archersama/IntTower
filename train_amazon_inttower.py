import numpy as np
import pandas as pd
import torch
import time
import torchvision
import random
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from preprocessing.inputs import SparseFeat, DenseFeat, VarLenSparseFeat
from model.dssm import DSSM
from model.col_dssm import Col_DSSM
from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.tensorboard import SummaryWriter
from utils import create_amazon_electronic_dataset
from model.IntTower import IntTower


def data_process(data_path):
    data = pd.read_csv(data_path)
    # data = data.drop(data[data['overall'] == 3].index)
    data['overall'] = data['overall'].apply(lambda x: 1 if x >= 4 else 0)
    data['price'] = data['price'].fillna(data['price'].mean())
    data = data.sort_values(by='unixReviewTime', ascending=True)
    # train = data.iloc[:int(len(data)*0.8)].copy()
    # test = data.iloc[int(len(data)*0.8):].copy()
    # train, test = train_test_split(data, test_size=0.2)
    # return train, test, data
    return data


def get_user_feature(data):
    data_group = data[data['overall'] == 1]
    data_group = data_group[['reviewerID', 'asin']].groupby('reviewerID').agg(list).reset_index()
    data_group['user_hist'] = data_group['asin'].apply(lambda x: '|'.join([str(i) for i in x]))
    data = pd.merge(data_group.drop('asin', axis=1), data, on='reviewerID')
    data_group = data[['reviewerID', 'overall']].groupby('reviewerID').agg('mean').reset_index()
    data_group.rename(columns={'overall': 'user_mean_rating'}, inplace=True)
    data = pd.merge(data_group, data, on='reviewerID')
    return data


def get_item_feature(data):
    data_group = data[['asin', 'overall']].groupby('asin').agg('mean').reset_index()
    data_group.rename(columns={'overall': 'item_mean_rating'}, inplace=True)
    data = pd.merge(data_group, data, on='asin')
    return data


def get_var_feature(data, col):
    key2index = {}

    def split(x):
        key_ans = x.split('|')
        for key in key_ans:
            if key not in key2index:
                # Notice : input value 0 is a special "padding",\
                # so we do not use 0 to encode valid feature for sequence input
                key2index[key] = len(key2index) + 1
        return list(map(lambda x: key2index[x], key_ans))

    var_feature = list(map(split, data[col].values))
    var_feature_length = np.array(list(map(len, var_feature)))
    max_len = max(var_feature_length)
    var_feature = pad_sequences(var_feature, maxlen=max_len, padding='post', )
    return key2index, var_feature, max_len


def get_test_var_feature(data, col, key2index, max_len):
    print("user_hist_list: \n")

    def split(x):
        key_ans = x.split('|')
        for key in key_ans:
            if key not in key2index:
                # Notice : input value 0 is a special "padding",
                # so we do not use 0 to encode valid feature for sequence input
                key2index[key] = len(key2index) + 1
        return list(map(lambda x: key2index[x], key_ans))

    test_hist = list(map(split, data[col].values))
    test_hist = pad_sequences(test_hist, maxlen=max_len, padding='post')
    return test_hist


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # %%

    embedding_dim = 32
    epoch = 10
    batch_size = 2048
    dropout = 0.3
    seed = 1023
    lr = 0.001


    print("1")




    setup_seed(seed)
    data_path = './data/amazon_eletronics.csv'

    data = data_process(data_path)
    data = get_user_feature(data)
    data = get_item_feature(data)

    sparse_features = ['reviewerID', 'asin', 'categories']
    dense_features = ['user_mean_rating', 'item_mean_rating','price']
    target = ['overall']

    user_sparse_features, user_dense_features = ['reviewerID'], ['user_mean_rating']
    item_sparse_features, item_dense_features = ['asin', 'categories'], ['item_mean_rating','price']



    # 1.Label Encoding for sparse features,and process sequence features
    for feat in sparse_features:
        lbe = LabelEncoder()
        lbe.fit(data[feat])
        data[feat] = lbe.transform(data[feat])
        # data[feat] = lbe.transform(test[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    mms.fit(data[dense_features])
    data[dense_features] = mms.transform(data[dense_features])

    train,test = train_test_split(data,test_size=0.2)

    # 2.preprocess the sequence feature
    # genres_key2index, train_genres_list, genres_maxlen = get_var_feature(train, 'genres')
    user_key2index, train_user_hist, user_maxlen = get_var_feature(train, 'user_hist')

    user_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=embedding_dim)
                            for i, feat in enumerate(user_sparse_features)] + [DenseFeat(feat, 1, ) for feat in
                                                                               user_dense_features]
    item_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=embedding_dim)
                            for i, feat in enumerate(item_sparse_features)] + [DenseFeat(feat, 1, ) for feat in
                                                                               item_dense_features]

    train_model_input = {name: train[name] for name in sparse_features + dense_features}

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    es = EarlyStopping(monitor='val_auc', min_delta=0, verbose=1,
                       patience=3, mode='max', baseline=None)
    mdckpt = ModelCheckpoint(filepath='amazon_fetower.ckpt', monitor='val_auc',
                             mode='max', verbose=1, save_best_only=True,save_weights_only=True)
    model = IntTower(user_feature_columns, item_feature_columns, field_dim= 64, task='binary', dnn_dropout=dropout,
                     device=device, user_head=2,item_head=2, user_filed_size=1, item_filed_size=2)

    model.compile("adam", "binary_crossentropy", metrics=['auc', 'accuracy', 'logloss']
                  ,lr = lr )

    params = list(model.parameters())
    num_params = 0
    for param in params:
        curr_num_params = 1
        for size_count in param.size():
            curr_num_params *= size_count
        num_params += curr_num_params
    print("total number of parameters: " + str(num_params))

    model.fit(train_model_input, train[target].values, batch_size=batch_size,
              epochs=epoch, verbose=2, validation_split=0.2,callbacks=[es,mdckpt])

    model.load_state_dict(torch.load('amazon_fetower.ckpt'))
    # 测试时不启用 BatchNormalization 和 Dropout
    model.eval()

    test_model_input = {name: test[name] for name in sparse_features + dense_features}

    pred_ts = model.predict(test_model_input, batch_size=2048)

    print("test LogLoss", round(log_loss(test[target].values, pred_ts), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ts), 4))

