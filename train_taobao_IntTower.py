import numpy as np
import pandas as pd
import torch
import torchvision
import random
import time
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from preprocessing.inputs import SparseFeat, DenseFeat, VarLenSparseFeat
from model.dssm import DSSM
from model.wdm import WideDeep
from model.dssm import DSSM
from model.col_dssm import Col_DSSM
from model.autoint import AutoInt
from model.autoint import AutoInt
from model.dcn import DCN
from model.IntTower import IntTower
from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint
from utils import create_amazon_electronic_dataset

def optimiz_memory(raw_data):
    optimized_g2 = raw_data.copy()

    g2_int = raw_data.select_dtypes(include=['int'])
    converted_int = g2_int.apply(pd.to_numeric,downcast='unsigned')
    optimized_g2[converted_int.columns] = converted_int

    g2_float = raw_data.select_dtypes(include=['float'])
    converted_float = g2_float.apply(pd.to_numeric,downcast='float')
    optimized_g2[converted_float.columns] = converted_float
    return optimized_g2


def optimiz_memory_profile(raw_data):
    optimized_gl = raw_data.copy()

    gl_int = raw_data.select_dtypes(include=['int'])
    converted_int = gl_int.apply(pd.to_numeric,downcast='unsigned')
    optimized_gl[converted_int.columns] = converted_int


    gl_obj = raw_data.select_dtypes(include=['object']).copy()
    converted_obj = pd.DataFrame()
    for col in gl_obj.columns:
        num_unique_values = len(gl_obj[col].unique())
        num_total_values = len(gl_obj[col])
        if num_unique_values / num_total_values < 0.5:
            converted_obj.loc[:,col] = gl_obj[col].astype('category')
        else:
            converted_obj.loc[:,col] = gl_obj[col]
    optimized_gl[converted_obj.columns] = converted_obj
    return optimized_gl



def data_process(profile_path,ad_path,user_path):
    profile_data = pd.read_csv(profile_path)
    ad_data = pd.read_csv(ad_path)
    user_data = pd.read_csv(user_path)
    profile_data = optimiz_memory_profile(profile_data)
    ad_data = optimiz_memory(ad_data)
    user_data = optimiz_memory(user_data)
    profile_data.rename(columns={'user':'userid'}, inplace = True)
    user_data.rename(columns={'new_user_class_level ':'new_user_class_level'}, inplace=True)
    df1 = profile_data.merge(user_data, on="userid")
    data = df1.merge(ad_data, on="adgroup_id")
    data['brand'] = data['brand'].fillna('-1', ).astype('int32')
    # data['age_level'] = data['age_level'].fillna('-1', )
    # data['cms_segid'] = data['cms_segid'].fillna('-1', )
    # data['cms_group_id'] = data['cms_group_id'].fillna('-1', )
    # data['final_gender_code'] = data['final_gender_code'].fillna('-1', )
    data['pvalue_level'] = data['pvalue_level'].fillna('-1', ).astype('int32')
    # data['shopping_level'] = data['shopping_level'].fillna('-1', )
    # data['occupation'] = data['occupation'].fillna('-1', )
    data['new_user_class_level'] = data['new_user_class_level'].fillna('-1', ).astype('int32')
    data = data.sort_values(by='time_stamp', ascending=True)
    return data


def get_user_feature(data):
    data_group = data[data['clk'] == 1]
    data_group = data_group[['userid', 'adgroup_id']].groupby('userid').agg(list).reset_index()
    data_group['user_hist'] = data_group['adgroup_id'].apply(lambda x: '|'.join([str(i) for i in x]))
    data = pd.merge(data_group.drop('adgroup_id', axis=1), data, on='userid')
    data_group = data[['userid', 'clk']].groupby('userid').agg('mean').reset_index()
#     data_group.rename(columns={'overall': 'user_mean_rating'}, inplace=True)
    data = pd.merge(data_group, data, on='userid')
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
    epoch = 15
    batch_size = 2048
    dropout = 0.5
    seed = 1023
    lr = 0.0001

    print("1")

    setup_seed(seed)

    profile_path = './data/raw_sample.csv'
    ad_path = './data/ad_feature.csv'
    user_path = './data/user_profile.csv'


    data = data_process(profile_path,ad_path,user_path)

    data = get_user_feature(data)



    sparse_features = ['userid', 'adgroup_id', 'pid', 'cms_segid', 'cms_group_id',
                       'final_gender_code','shopping_level', 'occupation', 'cate_id', 'campaign_id',
                       'customer','age_level', 'brand','pvalue_level','new_user_class_level']

    dense_features = ['price']

    user_sparse_features, user_dense_features = ['userid','cms_segid', 'cms_group_id','final_gender_code',
                                                 'age_level','pvalue_level','shopping_level','occupation',
                                                 'new_user_class_level',], []
    item_sparse_features, item_dense_features = ['adgroup_id', 'cate_id','campaign_id','customer',
                                                 'brand','pid'], ['price']
    target = ['clk_y']



    # user_sparse_features, user_dense_features = ['reviewerID'], ['user_mean_rating']
    # item_sparse_features, item_dense_features = ['asin', 'categories'], ['item_mean_rating','price']


    # 1.Label Encoding for sparse features,and process sequence features
    for feat in sparse_features:
        lbe = LabelEncoder()
        lbe.fit(data[feat])
        data[feat] = lbe.transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    mms.fit(data[dense_features])
    data[dense_features] = mms.transform(data[dense_features])

    train, test = train_test_split(data, test_size=0.2)



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

    # %%
    # 4.Define Model,train,predict and evaluate
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:2'

    es = EarlyStopping(monitor='val_auc', min_delta=0, verbose=1,
                       patience=3, mode='max', baseline=None)
    mdckpt = ModelCheckpoint(filepath='fe_model_2.ckpt', monitor='val_auc',
                             mode='max', verbose=1, save_best_only=True, save_weights_only=True)
    model = IntTower(user_feature_columns, item_feature_columns, field_dim= 16, task='binary', dnn_dropout=dropout,
           device=device, user_head=4,item_head=4,user_filed_size=9,item_filed_size=6)

    model.compile("adam", "binary_crossentropy", metrics=['auc', 'accuracy', 'logloss']
                  , lr=lr)

    model.fit(train_model_input, train[target].values, batch_size=batch_size,
              epochs=epoch, verbose=2, validation_split=0.2, callbacks=[es, mdckpt])

    # 5.preprocess the test data
    model.load_state_dict(torch.load('fe_model_2.ckpt'))
    # 测试时不启用 BatchNormalization 和 Dropout
    model.eval()

    test_model_input = {name: test[name] for name in sparse_features + dense_features}






    pred_ts = model.predict(test_model_input, batch_size=500)

    print("test LogLoss", round(log_loss(test[target].values, pred_ts), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ts), 4))


