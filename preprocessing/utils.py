import numpy as np
import torch


def slice_arrays(arrays, start=None, stop=None):
    if arrays is None:
        return [None]

    if isinstance(arrays, np.ndarray):
        arrays = [arrays]

    if isinstance(start, list) and stop is not None:
        raise ValueError('The stop argument has to be None if the value of start '
                         'is a list.')
    elif isinstance(arrays, list):
        if hasattr(start, '__len__'):
            # hdf5 datasets only support list objects as indices
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [None if x is None else x[start] for x in arrays]
        else:
            if len(arrays) == 1:
                return arrays[0][start:stop]
            return [None if x is None else x[start:stop] for x in arrays]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return arrays[start]
        elif hasattr(start, '__getitem__'):
            return arrays[start:stop]
        else:
            return [None]


def single_score(rep):
    score = torch.mean(rep, 1)
    return score


def Cosine_Similarity(query, candidate, gamma=1, dim=-1):
    query_norm = torch.norm(query, dim=dim)
    candidate_norm = torch.norm(candidate, dim=dim)
    cosine_score = torch.sum(torch.mul(query, candidate), dim=-1)
    cosine_score = torch.div(cosine_score, query_norm * candidate_norm + 1e-8)
    cosine_score = torch.clamp(cosine_score, -1, 1.0) * gamma
    return cosine_score


def dual_augmented_loss(y, user_embedding, item_embedding, user_augment_vector, item_augment_vector):
    user_augment_vector = torch.squeeze(user_augment_vector)
    item_augment_vector = torch.squeeze(item_augment_vector)
    loss_u = torch.mean(torch.pow(y * user_augment_vector + (1 - y) * item_embedding - item_embedding, 2))
    loss_v = torch.mean(torch.pow(y * item_augment_vector + (1 - y) * user_embedding - user_embedding, 2))

    return loss_u,loss_v



def contrast_loss(y, user_embedding, item_embedding):
    # print(user_embedding.shape)
    user_embedding = torch.nn.functional.normalize(user_embedding, dim=-1)
    item_embedding = torch.nn.functional.normalize(item_embedding, dim=-1)
    #
    pos = 0
    all = 0
    tau = 0.001
    pos_index = y.expand(y.shape[0], item_embedding.shape[1])

    m = torch.nn.ZeroPad2d((0, item_embedding.shape[1] - user_embedding.shape[1], 0, 0))

    user_embedding = m(user_embedding)

    # print(user_embedding.shape,item_embedding.shape,pos_index.shape)

    pos += torch.mean(user_embedding * item_embedding * pos_index) / tau

    all += torch.mean(user_embedding * item_embedding) / tau



    contras = -torch.log(torch.exp(pos) / torch.exp(all))
    # print(contras)
    return contras


def col_score(user_rep, item_rep, user_fea_col):
    # print(user_rep.shape)
    # print(item_rep.shape)
    query_norm = torch.norm(user_rep, dim=-1)
    temp = torch.zeros(size=item_rep.shape).cuda()

    candidate_norm = torch.norm(user_rep, dim=-1)

    split_user_rep = torch.split(user_rep, 128, 1)
    for i in range(user_fea_col):
        temp += split_user_rep[i] * item_rep
    score = torch.sum(temp, 1)
    score = torch.div(score, query_norm * candidate_norm + 1e-8)
    score = torch.clamp(score, -1, 1.0)
    return score


def col_score_2(user_rep, item_rep, user_fea_col, item_fea_col, embedding_dim):
    # print(user_rep.shape)
    # print(item_rep.shape)
    # print("col_score")
    user_rep = torch.reshape(user_rep, (-1, user_fea_col, embedding_dim))
    item_rep = torch.reshape(item_rep, (-1, item_fea_col, embedding_dim))

    return (user_rep @ item_rep.permute(0, 2, 1)).max(2).values.sum(1)


def fe_score(user_rep, item_rep, user_fea_col, item_fea_col, user_embedding_dim, item_embedding_dim):
    # print(user_rep.shape)
    # print(item_rep.shape)
    # print("col_score")
    score = []
    # user_embedding, item_embedding  = user_rep[0],item_rep[0]
    # user_rep = torch.reshape(user_embedding, (-1, user_fea_col, user_embedding_dim[0]))
    # item_rep = torch.reshape(item_embedding, (-1, item_fea_col, item_embedding_dim[0]))
    #
    # return (user_rep @ item_rep.permute(0, 2, 1)).max(2).values.sum(1)


    for i in range(len(user_embedding_dim)):
        # print(user_rep[i].shape)
        # print(item_rep[i].shape)
        user_temp = torch.reshape(user_rep[i], (-1, user_fea_col, user_embedding_dim[i]))
        item_temp = torch.reshape(item_rep[-1], (-1, item_fea_col, item_embedding_dim[i]))
        # print(user_temp.shape)
        # print(item_temp.shape)
        score.append((user_temp @ item_temp.permute(0, 2, 1)).max(2).values.sum(1))
    # all_score = 0.4 * score[0] + 0.2*score[1] + 0.4*score[2]
    score = torch.stack(score).transpose(1, 0)
    # # print(torch.sum(score,1))
    # all_score = all_score.unsqueeze(1)
    # # print(all_score.shape)
    # return all_score
    return torch.sum(score, 1)

