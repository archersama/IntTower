import torch
import torch.nn as nn
from layers.activation import activation_layer
from tensorflow.python.keras.layers import Layer
import tensorflow as tf




class SampledSoftmaxLayer(Layer):
    def __init__(self, num_sampled=5, **kwargs):
        self.num_sampled = num_sampled
        super(SampledSoftmaxLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.size = input_shape[0][0]
        self.zero_bias = self.add_weight(shape=[self.size],
                                         initializer=Zeros,
                                         dtype=tf.float32,
                                         trainable=False,
                                         name="bias")
        super(SampledSoftmaxLayer, self).build(input_shape)

    def call(self, inputs_with_label_idx, training=None, **kwargs):
        """
        The first input should be the model as it were, and the second the
        target (i.e., a repeat of the training data) to compute the labels
        argument
        """
        embeddings, inputs, label_idx = inputs_with_label_idx

        loss = tf.nn.sampled_softmax_loss(weights=embeddings,  # self.item_embedding.
                                          biases=self.zero_bias,
                                          labels=label_idx,
                                          inputs=inputs,
                                          num_sampled=self.num_sampled,
                                          num_classes=self.size,  # self.target_song_size
                                          )
        return tf.expand_dims(loss, axis=1)

    def compute_output_shape(self, input_shape):
        return (None, 1)

    def get_config(self, ):
        config = {'num_sampled': self.num_sampled}
        base_config = super(SampledSoftmaxLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



def concat_fun(inputs, axis=-1):
    if len(inputs) == 1:
        return inputs[0]
    else:
        return torch.cat(inputs, dim=axis)



class PredictionLayer(nn.Module):
    def __init__(self, task='binary', use_bias=True, **kwargs):
        if task not in ["binary", "multiclass", "regression"]:
            raise ValueError("task must be binary,multiclass or regression")

        super(PredictionLayer, self).__init__()
        self.use_bias = use_bias
        self.task = task
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros((1,)))

    def forward(self, X):
        output = X
        if self.use_bias:
            output += self.bias
        if self.task == "binary":
            output = torch.sigmoid(X)
        return output






class DNN(nn.Module):
    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 init_std=0.0001, dice_dim=3, seed=1024, device='cpu'):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        if inputs_dim > 0:
            hidden_units = [inputs_dim] + list(hidden_units)
        else:
            hidden_units = list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i+1]) for i in range(len(hidden_units) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i+1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = nn.ModuleList(
            [activation_layer(activation, hidden_units[i+1], dice_dim) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)

    def forward(self, inputs):
        deep_input = inputs
        for i in range(len(self.linears)):
            fc = self.linears[i](deep_input)

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            deep_input = fc
        return deep_input


class User_Fe_DNN(nn.Module):
    def __init__(self, inputs_dim, field_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 init_std=0.0001, user_head =6,  dice_dim=3, seed=1024, device='cpu'):
        super(User_Fe_DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        self.user_head = user_head
        self.field_dim = field_dim
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        if inputs_dim > 0:
            hidden_units = [inputs_dim] + list(hidden_units)
        else:
            hidden_units = list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i+1]) for i in range(len(hidden_units) - 1)])


        self.Fe_linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], self.field_dim*self.user_head) for i in range(len(hidden_units) - 1)])


        # self.user_Fe_rep = []
        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i+1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = nn.ModuleList(
            [activation_layer(activation, hidden_units[i+1], dice_dim) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)

    def forward(self, inputs):
        deep_input = inputs
        self.user_fe_rep = []
        for i in range(len(self.linears)):

            fc = self.linears[i](deep_input)
            user_temp = self.Fe_linears[i](deep_input)

            self.user_fe_rep.append(user_temp)

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            deep_input = fc
        return self.user_fe_rep


class Item_Fe_DNN(nn.Module):
    def __init__(self, inputs_dim, field_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 init_std=0.0001, item_head = 3, dice_dim=3, seed=1024, device='cpu'):
        super(Item_Fe_DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        self.item_head = item_head
        self.field_dim = field_dim
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        if inputs_dim > 0:
            hidden_units = [inputs_dim] + list(hidden_units)
        else:
            hidden_units = list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        # self.Fe_linears = nn.ModuleList(
        #     [nn.Linear(hidden_units[-1], self.field_dim * self.item_head) for i in
        #      range(len(hidden_units) - 1)])

        self.Fe_linears = nn.ModuleList(
            [nn.Linear(hidden_units[-1], self.field_dim * self.item_head)])

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = nn.ModuleList(
            [activation_layer(activation, hidden_units[i + 1], dice_dim) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)

    def forward(self, inputs):
        self.item_fe_rep = []
        deep_input = inputs
        for i in range(len(self.linears)):
            # print("item_deep_input",deep_input.shape)
            fc = self.linears[i](deep_input)


            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)

            fc = self.dropout(fc)
            deep_input = fc
        for i in range(len(self.Fe_linears)):
            item_temp = self.Fe_linears[i](deep_input)
            # print("item_col_rep", item_temp.shape)
            self.item_fe_rep.append(item_temp)
        return self.item_fe_rep


class LocalActivationUnit(nn.Module):
    def __init__(self, hidden_units=(64, 32), embedding_dim=4, activation='sigmoid', dropout_rate=0,
                 dice_dim=3, l2_reg=0, use_bn=False):
        super(LocalActivationUnit, self).__init__()

        self.dnn = DNN(inputs_dim=4 * embedding_dim,
                       hidden_units=hidden_units,
                       activation=activation,
                       l2_reg=l2_reg,
                       dropout_rate=dropout_rate,
                       dice_dim=dice_dim,
                       use_bn=use_bn)

        self.dense = nn.Linear(hidden_units[-1], 1)

    def forward(self, query, user_behavier):
        # query ad            : size -> batch_size * 1 * embedding_size
        # user behavior       : size -> batch_size * time_seq_len * embedding_size
        user_behavier_len = user_behavier.size(1)

        queries = query.expand(-1, user_behavier_len, -1)

        attention_input = torch.cat([queries, user_behavier, queries - user_behavier, queries * user_behavier],
                                    dim=-1)    # [B, T, 4*E]
        attention_out = self.dnn(attention_input)

        attention_score = self.dense(attention_out)    # [B, T, 1]

        return attention_score







