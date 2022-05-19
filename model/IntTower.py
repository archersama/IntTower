"""

"""

from model.base_tower import BaseTower
from preprocessing.inputs import combined_dnn_input, compute_input_dim

from layers.core import DNN
from layers.core import User_Fe_DNN,Item_Fe_DNN
import torch
from preprocessing.utils import Cosine_Similarity
from preprocessing.utils import col_score
from preprocessing.utils import fe_score
from preprocessing.utils import single_score
from layers.interaction import SENETLayer
from layers.interaction import LightSE

class IntTower(BaseTower):

    def __init__(self, user_dnn_feature_columns, item_dnn_feature_columns, gamma=1, dnn_use_bn=True,
                 dnn_hidden_units=(300, 300, 128), field_dim = 32, user_head=1,item_head=1, dnn_activation='relu', l2_reg_dnn=0, l2_reg_embedding=1e-5,
                 dnn_dropout = 0, init_std=0.0001, seed=124, task='binary', device='cpu', gpus=None,user_filed_size = 1,
                 item_filed_size = 1):
        super(IntTower, self).__init__(user_dnn_feature_columns, item_dnn_feature_columns,
                                    l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                    device=device, gpus=gpus)

        if len(user_dnn_feature_columns) > 0:
            self.user_fe_dnn= User_Fe_DNN(compute_input_dim(user_dnn_feature_columns),field_dim, dnn_hidden_units,
                                activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                use_bn=dnn_use_bn, user_head = user_head, init_std=init_std, device=device)
            self.user_dnn_embedding = None
            # self.user_col_rep = []

        if len(item_dnn_feature_columns) > 0:
            self.item_fe_dnn = Item_Fe_DNN(compute_input_dim(item_dnn_feature_columns), field_dim,dnn_hidden_units,
                                activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                                use_bn=dnn_use_bn, item_head = item_head,init_std=init_std, device=device)
            self.item_dnn_embedding = None
            # self.item_col_rep = []

        self.gamma = gamma
        self.l2_reg_embedding = l2_reg_embedding
        self.seed = seed
        self.task = task
        self.device = device
        self.gpus = gpus
        self.user_head = user_head
        self.item_head = item_head

        self.user_filed_size = user_filed_size
        self.item_filed_size = item_filed_size
        self.User_sim_non_local = LightSE(self.user_filed_size, seed, device)
        self.Item_sim_non_local = LightSE(self.item_filed_size,  seed, device)
        self.User_SE = SENETLayer(self.user_filed_size, 3, seed, device)
        self.Item_SE = SENETLayer(self.item_filed_size, 3, seed, device)
        self.field_dim = field_dim


    def forward(self, inputs):
        # print('inputs shape:', inputs.shape)
        if len(self.user_dnn_feature_columns) > 0:
            user_sparse_embedding_list, user_dense_value_list = \
                self.input_from_feature_columns(inputs, self.user_dnn_feature_columns, self.user_embedding_dict)

            user_sparse_embedding = torch.cat(user_sparse_embedding_list, dim=1)
            User_sim_embedding = self.User_sim_non_local(user_sparse_embedding)
            sparse_dnn_input = torch.flatten(User_sim_embedding, start_dim=1)
            if(len(user_dense_value_list)>0):
                dense_dnn_input = torch.flatten(torch.cat(user_dense_value_list, dim=-1), start_dim=1)
                user_dnn_input = torch.cat([sparse_dnn_input, dense_dnn_input],axis=-1)
            else:
                user_dnn_input = sparse_dnn_input

            self.user_fe_rep = self.user_fe_dnn(user_dnn_input)
            self.user_dnn_embedding = self.user_fe_rep[-1]


        if len(self.item_dnn_feature_columns) > 0:
            item_sparse_embedding_list, item_dense_value_list = \
                self.input_from_feature_columns(inputs, self.item_dnn_feature_columns, self.item_embedding_dict)


            item_sparse_embedding = torch.cat(item_sparse_embedding_list, dim=1)
            Item_sim_embedding = self.Item_sim_non_local(item_sparse_embedding)
            sparse_dnn_input = torch.flatten(Item_sim_embedding, start_dim=1)
            dense_dnn_input = torch.flatten(torch.cat(item_dense_value_list, dim=-1), start_dim=1)

            item_dnn_input = torch.cat([sparse_dnn_input, dense_dnn_input], axis=-1)


            self.item_fe_rep = self.item_fe_dnn(item_dnn_input)
            self.item_dnn_embedding = self.item_fe_rep[-1]


        if len(self.user_dnn_feature_columns) > 0 and len(self.item_dnn_feature_columns) > 0:

            score = fe_score(self.user_fe_rep, self.item_fe_rep, self.user_head,\
                self.item_head,[self.field_dim,self.field_dim,self.field_dim],[self.field_dim,
                                                                                                self.field_dim,self.field_dim])

            output = self.out(score)
            return output, self.user_dnn_embedding, self.item_dnn_embedding

        elif len(self.user_dnn_feature_columns) > 0:
            return self.user_dnn_embedding

        elif len(self.item_dnn_feature_columns) > 0:
            return self.item_dnn_embedding

        else:
            raise Exception("input Error! user and item feature columns are empty.")