import torch
import torch.nn as nn
from model.base_model import BaseModel
from layers.interaction import FM
from layers.core import DNN
from preprocessing.inputs import combined_dnn_input
from layers.interaction import SENETLayer
from layers.interaction import LightSE

class DeepFM(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, use_fm=True, dnn_hidden_units=(300,300,128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001,
                 seed=1024, dnn_dropout=0, dnn_activation='relu', dnn_use_bn=False, task='binary',
                 device='cpu', gpus=None):
        super(DeepFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus)

        self.filed_size = len(self.embedding_dict)
        self.SE = SENETLayer(self.filed_size, 3, seed, device)



        self.use_fm = use_fm
        self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
        if use_fm:
            self.fm = FM()

        if self.use_dnn:
            self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout,
                           use_bn=dnn_use_bn, init_std=init_std, device=device)
            self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)

            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        self.to(device)

    def forward(self, inputs):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(inputs, self.dnn_feature_columns,
                                                                               self.embedding_dict)
        # sparse_embedding_input = torch.cat(sparse_embedding_list, dim=1)

        # senet_output = self.SE(sparse_embedding_input)

        # print(sparse_embedding_input.shape)
        # print(senet_output.shape)
        logit = self.linear_model(inputs)


        if self.use_fm and len(sparse_embedding_list) > 0:
            fm_input = torch.cat(sparse_embedding_list, dim=1)
            # senet_output = self.sim_non_local(fm_input)
            # # print("use_se")
            # logit += self.fm(senet_output)
            logit += self.fm(fm_input)

        if self.use_dnn:
            dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
            # sparse_embedding = torch.cat(sparse_embedding_list, dim=1)
            # se_embedding = self.sim_non_local(sparse_embedding)
            # sparse_dnn_input = torch.flatten(se_embedding, start_dim=1)
            #
            # dense_dnn_input = torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1)
            # dnn_input = torch.cat([sparse_dnn_input, dense_dnn_input], axis=-1)

            # dnn_input = combined_dnn_input(senet_output, dense_value_list)
            dnn_output = self.dnn(dnn_input)
            dnn_logit = self.dnn_linear(dnn_output)
            logit += dnn_logit

        y_pred = self.out(logit)

        return y_pred
