import torch
import numpy as np
import random
torch.set_printoptions(profile="full")


# prepare the dataset for autoregressive models (rnn, transformer-ar)
def prepare_ar_data(path, seq_len, vocab, start_token):
    f = open(path)
    all_seq = []
    for line in f:
        line = line.split()
        seq = []
        for i in range(seq_len + 1):
            seq.append(int(line[i]) if i < len(line) else (vocab + 1))
        all_seq.append(seq)
    target = np.array(all_seq)
    data_size, seq_len = target.shape
    input = np.zeros((data_size, seq_len))
    input[:, 0] = start_token
    input[:, 1:] = target[:, :seq_len-1]
    return input, target


# get authentic data for Transformer_Nonautoregressive model
def prepare_nar_data(path, seq_len, token_num):
    f = open(path)
    seq_list = []
    for line in f:
        line = line.split()
        n = len(line)
        seq = []
        for i in range(seq_len+1):
            if i < n:
                ind = int(line[i])
                seq.append(ind-1)
            else:
                seq.append(token_num)
        seq_list.append(seq)
    seqs = np.array(seq_list)
    return seqs


# prepare the one-hot format authentic data for discriminator of GAN models
def prepare_onehot_aut_data(path, ntoken, seq_len):
    end_token = ntoken + 1
    f = open(path)
    onehotdict = []
    for line in f:
        line = line.split()
        seq = []
        for i in range(seq_len+1):
            onehot = [0 for id in range(end_token)]
            if i < len(line):
                ind = int(line[i])
                onehot[ind-1] = 1
                seq.append(onehot)
            else:
                onehot[end_token-1] = 1
                seq.append(onehot)
        onehotdict.append(seq)
    onehot_data = np.array(onehotdict)
    return onehot_data


# prepare discriminator labels
def prepare_dis_label(size):
    pos_label = np.ones(size)
    neg_label = np.zeros(size)
    pos_label = torch.tensor(pos_label, dtype=torch.float32, requires_grad=False)
    neg_label = torch.tensor(neg_label, dtype=torch.float32, requires_grad=False)
    return pos_label, neg_label


# prepare the data for "off-the-shelf" classifier
def prepare_cls_data(path, type, length, vocab):
    f = open(path)
    all_seq = []
    label = []
    act_dist = []

    for line in f:
        line = line.split()
        n = len(line)
        seq = []
        act_dist_i = [0 for _ in range(vocab + 1)]
        for i in range(length+1):
            if i < n:
                ind = int(line[i])
                seq.append(ind-1)
                act_dist_i[ind-1] += 1
            else:
                seq.append(vocab)
        act_dist_i[vocab] = n
        all_seq.append(seq)
        act_dist.append(act_dist_i)
        if type == 'neg':
            label.append([0])
        else:
            label.append([1])

    all_seq = np.array(all_seq)
    label = np.array(label)
    act_dist = np.array(act_dist)

    return all_seq, label, act_dist
