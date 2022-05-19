import pandas as pd
import torch
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint
from preprocessing.inputs import SparseFeat, DenseFeat, VarLenSparseFeat
from preprocessing.inputs import SparseFeat, DenseFeat, get_feature_names
from model.deepfm import DeepFM
from model.dcn import DCN
from model.cold import Cold
from model.autoint import AutoInt
from model.wdm import WideDeep
from torchsummary import summary

def data_process(data_path, samp_rows=100000):
    data = pd.read_csv(data_path)
    data = data.drop(data[data['rating'] == 3].index)
    data['rating'] = data['rating'].apply(lambda x: 1 if x > 3 else 0)
    data = data.sort_values(by='timestamp', ascending=True)
    train,test = train_test_split(data,test_size= 0.2 )
    return train, test, data


def get_user_feature(data):
    data_group = data[data['rating'] == 1]
    data_group = data_group[['user_id', 'movie_id']].groupby('user_id').agg(list).reset_index()
    data_group['user_hist'] = data_group['movie_id'].apply(lambda x: '|'.join([str(i) for i in x]))
    data = pd.merge(data_group.drop('movie_id', axis=1), data, on='user_id')
    data_group = data[['user_id', 'rating']].groupby('user_id').agg('mean').reset_index()
    data_group.rename(columns={'rating': 'user_mean_rating'}, inplace=True)
    data = pd.merge(data_group, data, on='user_id')
    return data


def get_item_feature(data):
    data_group = data[['movie_id', 'rating']].groupby('movie_id').agg('mean').reset_index()
    data_group.rename(columns={'rating': 'item_mean_rating'}, inplace=True)
    data = pd.merge(data_group, data, on='movie_id')
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

def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    # 1.load data
    print("1")
    embedding_dim = 32
    epoch = 10
    batch_size = 2048
    seed = 1023
    setup_seed(1023)
    lr = 0.001
    dropout = 0.1

    data_path = './data/movielens.txt'
    train,test,data = data_process(data_path)

    train = get_user_feature(train)
    train = get_item_feature(train)

    test = get_user_feature(test)
    test = get_item_feature(test)

    sparse_features = ['user_id', 'movie_id', 'gender', 'age', 'occupation']
    dense_features = ['user_mean_rating', 'item_mean_rating']
    target = ['rating']

    user_sparse_features, user_dense_features = ['user_id', 'gender', 'age', 'occupation'], ['user_mean_rating']
    item_sparse_features, item_dense_features = ['movie_id', ], ['item_mean_rating']

    # 1.Label Encoding for sparse features,and process sequence features
    for feat in sparse_features:
        lbe = LabelEncoder()
        lbe.fit(data[feat])
        train[feat] = lbe.transform(train[feat])
        test[feat] = lbe.transform(test[feat])

    mms = MinMaxScaler(feature_range=(0, 1))
    mms.fit(train[dense_features])
    mms.fit(test[dense_features])
    train[dense_features] = mms.transform(train[dense_features])
    test[dense_features] = mms.transform(test[dense_features])



    # 2.preprocess the sequence feature
    genres_key2index, train_genres_list, genres_maxlen = get_var_feature(train, 'genres')
    user_key2index, train_user_hist, user_maxlen = get_var_feature(train, 'user_hist')

    sparse_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=embedding_dim)
                            for i, feat in enumerate(sparse_features)]

    dense_feature_columns = [DenseFeat(feat, 1, ) for feat in dense_features]

    item_varlen_feature_columns = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=1000, embedding_dim=32),
                                                    maxlen=genres_maxlen, combiner='mean', length_name=None)]

    user_varlen_feature_columns = [VarLenSparseFeat(SparseFeat('user_hist', vocabulary_size=4000, embedding_dim=32),
                                                    maxlen=user_maxlen, combiner='mean', length_name=None)]

    # 3.generate input data for model
    # sparse_feature_columns += user_varlen_feature_columns
    sparse_feature_columns += item_varlen_feature_columns

    linear_feature_columns = sparse_feature_columns + dense_feature_columns
    dnn_feature_columns = sparse_feature_columns + dense_feature_columns


    # add user history as user_varlen_feature_columns
    train_model_input = {name: train[name] for name in sparse_features + dense_features}
    train_model_input["genres"] = train_genres_list
    # train_model_input["user_hist"] = train_user_hist
    # print(train_model_input)
    # %%
    # 4.Define Model,train,predict and evaluate
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    es = EarlyStopping(monitor='val_auc', min_delta=0, verbose=1,
                       patience=3, mode='max', baseline=None)
    mdckpt = ModelCheckpoint(filepath='wide_model.ckpt', monitor='val_auc',
                             mode='max', verbose=1, save_best_only=True, save_weights_only=True)

    # If you want to try another model such as "autoint", just replace the model name directly
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary', dnn_dropout=dropout,
                    device=device)
    model.compile("adam", "binary_crossentropy", metrics=['auc', 'accuracy', 'logloss']
                  , lr=lr)





    model.fit(train_model_input, train[target].values, batch_size= batch_size, epochs=epoch, verbose=2, validation_split=0.2,
              callbacks=[es, mdckpt])





    model.load_state_dict(torch.load('wide_model.ckpt'))
    # 测试时不启用 BatchNormalization 和 Dropout
    model.eval()




    test_genres_list = get_test_var_feature(test, 'genres', genres_key2index, genres_maxlen)
    # test_user_hist = get_test_var_feature(test, 'user_hist', user_key2index, user_maxlen)

    test_model_input = {name: test[name] for name in sparse_features + dense_features}
    # train_model_input["genres"] = train_genres_list

    test_model_input["genres"] = test_genres_list
    # test_model_input["user_hist"] = test_user_hist

    # %%
    pred_ts = model.predict(test_model_input, batch_size=2048)

    # %%

    print("test LogLoss", round(log_loss(test[target].values, pred_ts), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ts), 4))


