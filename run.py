import os
import torch
import numpy as np
from nets.process_gan import ProcessGAN
from nets.rnn import RNNs
from nets.trans_ar import Transformer_AR
from nets.transformer_nar import Transformer_NAR
from eval.act_dist_eval import get_act_dif
from eval.length_eval import get_length_dif
from eval.variance_eval import get_variance_dif
from datetime import datetime

# set the random seed
seed = 88
np.random.seed(seed)


def get_config(data, model):
    # Consider the data privacy, only the public data configurations are shown
    # Users can set the configurations based on specific data

    # sepsis dataset
    config_sep_gan = {
        'train_size': 676,
        'test_size' : 85,
        'valid_size': 85,
        'w_a'       : 1,   # the weight of activity distribution loss
        'seq_len'   : 186, # the longest sequence length in data
        'vocab_num' : 17,  # the size of vocabulary
        'emb_size'  : 8,   # embedding dimension
        'n_hid'     : 32,  # the dimension of the feedforward network model in nn.TransformerEncoder
        'n_layer'   : 2,   # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        'n_head_g'  : 4,   # the number of heads in the multi-head-attention models of generator
        'n_head_d'  : 2,   # the number of heads in the multi-head-attention models of discriminator
        'drop_out'  : 0.1, # the dropout value
        'batch_size': 128,
        'gd_ratio'  : 2,       # k value: the generator updates k times and discriminator updates 1 time
        'lr_gen'    : 0.0001,  # generator learning rate
        'lr_dis'    : 0.0001,  # discriminator learning rate
        'epochs'    : 2000,    # total epochs
        'seed'      : seed,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    config_sep_rnn = {
        'seq_num'   : 846,
        'seq_len'   : 186,
        'vocab_num' : 17,  # the size of vocabulary
        'emb_size'  : 4,   # embedding dimension
        'n_hid'     : 16,  # the dimension of the feedforward network model in RNN models
        'batch_size': 128,
        'epochs'    : 1000,
        'seed'      : seed,
        'train_size': 676,
        'test_size' : 85,
        'valid_size': 85,
        'lr'        : 0.01,
    }

    config_sep_trans_ar = {
        'seq_num'   : 846,
        'seq_len'   : 186,
        'vocab_num' : 17,
        'emb_size'  : 8,  # embedding dimension
        'n_hid'     : 16, # the dimension of the feedforward network model in nn.TransformerEncoder
        'batch_size': 64,
        'epochs'    : 1000,
        'seed'      : seed,
        'train_size': 676,
        'test_size' : 85,
        'valid_size': 85,
        'device'    : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'drop_out'  : 0.1, # the dropout value
        'n_layer'   : 3,
        'n_head'    : 4,
        'lr_gen'    : 0.1,
        'step'      : 3,   # the step size of auto-regressively feed the training sequences
    }

    config_sep_trans_nar = {
        'train_size': 676,
        'test_size' : 85,
        'valid_size': 85,
        'seq_len'   : 186,
        'vocab_num' : 17,  # the size of vocabulary
        'emb_size'  : 8,  # embedding dimension
        'n_hid'     : 32,  # the dimension of the feedforward network model in nn.TransformerEncoder
        'n_layer'   : 3,  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        'n_head'    : 4,  # the number of heads in the multi-head-attention models
        'drop_out'  : 0.1,  # the dropout value
        'device'    : torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'batch_size': 32,
        'lr_gen'    : 0.1,
        'epochs'    : 1000,
        'seed'      : seed,
    }

    all_mode = {}
    all_mode['SEP'] = {}
    all_mode['SEP']['GAN'] = config_sep_gan
    all_mode['SEP']['RNN'] = config_sep_rnn
    all_mode['SEP']['Trans_ar'] = config_sep_trans_ar
    all_mode['SEP']['Trans_nar'] = config_sep_trans_nar

    return all_mode[data][model]


def get_dif(res_path, syn_path, save_path):
    get_act_dif(res_path, syn_path, save_path)
    get_length_dif(res_path, syn_path, save_path)
    get_variance_dif(res_path, syn_path, save_path)


def run_rnn(data, save_time, res_path, model, config, gen_num):
    save_path = 'result/' + data + '/' + save_time +'_'+ model + '/'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    rnn = RNNs(res_path, save_path, model, config, gen_num)
    rnn.run()


def run_transformer_ar(data, save_time, res_path, config, gen_num):
    save_path = 'result/' + data + '/' + save_time +'_'+ 'transformer_ar/'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    trans_ar = Transformer_AR(res_path, save_path, config, gen_num)
    trans_ar.run()


def run_transformer_nar(data, save_time, res_path, config, gen_num):
    save_path = 'result/' + data + '/' + save_time + '_' + 'transformer_nar/'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    trans_nar = Transformer_NAR(res_path, save_path, config, gen_num)
    trans_nar.run()


def run_gan(data, save_time, res_path, mode, config, gen_num):
    save_path = 'result/' + data + '/' + save_time + '_' + mode + '/'
    save_path_res = 'result/' + data + '/' + save_time + '_' + mode + '/' + 'stats/'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(save_path_res), exist_ok=True)

    if mode == 'GAN_ORIGINAL':
        gan = ProcessGAN(res_path, save_path, config, gen_num, 'Vanilla')
    if mode == 'MSE':
        gan = ProcessGAN(res_path, save_path, config, gen_num, 'MSE')
    if mode == 'KL':
        gan = ProcessGAN(res_path, save_path, config, gen_num, 'KL')

    gan.run()


if __name__ == '__main__':

    datasets = ['SEP']
    #             1

    models   = ['gru', 'lstm', 'trans_ar', 'trans_nar', 'GAN_ORIGINAL', 'MSE', 'KL']
    #             1      2        3           4               5           6      7

    data_id_list  = [1]  # The dataset you want to test
    model_id_list = [6]  # The model you want to test
    gen_num = 500  # Number of synthetic sequences you want to generate

    for data_id in data_id_list:

        seq_path = 'data/data_all/data_seq/' + datasets[data_id - 1] + '.txt'

        # save file name based on timestamp
        dateTimeObjlocal = datetime.now()
        print("System Timestamp: ", dateTimeObjlocal)
        save_time = str(dateTimeObjlocal.month) + '-' + str(dateTimeObjlocal.day) + '-' + str(dateTimeObjlocal.hour) + '-' + str(dateTimeObjlocal.minute)

        for model_id in model_id_list:

            if model_id == 1:
                config = get_config(datasets[data_id - 1], 'RNN')
                run_rnn(datasets[data_id - 1], save_time, seq_path, 'gru', config, gen_num)

            if model_id == 2:
                config = get_config(datasets[data_id - 1], 'RNN')
                run_rnn(datasets[data_id - 1], save_time, seq_path, 'lstm', config, gen_num)

            if model_id == 3:
                config = get_config(datasets[data_id - 1], 'Trans_ar')
                run_transformer_ar(datasets[data_id - 1], save_time, seq_path, config, gen_num)

            if model_id == 4:
                config = get_config(datasets[data_id - 1], 'Trans_nar')
                run_transformer_nar(datasets[data_id - 1], save_time, seq_path, config, gen_num)

            if model_id == 5:
                config = get_config(datasets[data_id - 1], 'GAN')
                run_gan(datasets[data_id - 1], save_time, seq_path, 'GAN_ORIGINAL', config, gen_num)

            if model_id == 6:
                config = get_config(datasets[data_id - 1], 'GAN')
                run_gan(datasets[data_id - 1], save_time, seq_path, 'MSE', config, gen_num)

            if model_id == 7:
                config = get_config(datasets[data_id - 1], 'GAN')
                run_gan(datasets[data_id - 1], save_time, seq_path, 'KL', config, gen_num)

