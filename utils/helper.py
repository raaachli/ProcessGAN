import random
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from eval.act_dist_eval import save_act_difference
from eval.length_eval import save_len_difference
from eval.variance_eval import save_variance_dif


def generate_random_data(bs, vocab_size, seq_len):
    rand_data = []
    end_token = vocab_size
    for i in range(bs):
        randomlist = random.choices(range(0, end_token + 1), k=seq_len + 1)
        rand_data.append(randomlist)
    return rand_data


def gen_data_from_rand(size, g_model, ntokens, device, result_file, save_path, seq_len):
    gen_list = []
    for gen in range(size):
        gen_rand_set = generate_random_data(1, ntokens, seq_len)
        gen_rand_set = torch.tensor(gen_rand_set, dtype=torch.int64).to(device)
        gen_rand_set = torch.transpose(gen_rand_set, 0, 1)
        mask_len = gen_rand_set.size()[0]
        src_mask = g_model.generate_square_subsequent_mask(mask_len).to(device)
        g_output = g_model(gen_rand_set, src_mask)
        g_output = g_output.permute(1, 0, 2)
        out = F.gumbel_softmax(g_output, tau=1, hard=True)
        out_list = out.tolist()
        seq = []
        for j in range(seq_len + 1):
            for k in range(ntokens + 1):
                if out_list[0][j][k] == 1:
                    seq.append(k)
        sub_samp = []
        n = len(seq)
        for j in range(n):
            tok = seq[j]
            if tok != ntokens:
                sub_samp.append(tok + 1)
            else:
                break
        gen_list.append(sub_samp)
    with open(save_path + result_file + '.txt', 'a') as f:
        f.writelines(' '.join(str(token) for token in list) + '\n' for list in gen_list)
    return gen_list


def write_log(save_path, log, file_name):
    with open(save_path + file_name, 'a') as filehandle:
        for listitem in log:
            filehandle.write('%s\n' % listitem)


def plot_loss(save_path, log, file_name, type):
    fig, ax = plt.subplots()
    losses = np.array(log)
    if len(losses.shape) == 2:
        plt.plot(losses.T[0], label='train loss')
        plt.plot(losses.T[1], label='val loss')
    else:
        plt.plot(losses, label=type)
    plt.xlabel('epochs')
    plt.ylabel(type)
    plt.legend()
    fig.savefig(save_path + file_name)


def eval_result(save_path, gen_list, test_list):
    save_len_difference(gen_list, test_list, save_path)
    save_act_difference(gen_list, test_list, save_path)
    save_variance_dif(gen_list, test_list, save_path)


def get_pad_mask(output, batch_size, seq_len, vocab_size, padding_ind, device):
    out_list = output.tolist()
    pad_mask = []
    for i in range(batch_size):
        pad = seq_len
        for j in range(seq_len):
            if out_list[i][j][padding_ind] == 1:
                pad = j
                break
        pad_mask.append(pad)
    n = len(pad_mask)
    pad_mask_mul = []
    pad_mask_add = []
    for i in range(n):
        seq_mul = []
        seq_add = []
        onehot_one = [1 for _ in range(vocab_size)]
        onehot_zero = [0 for _ in range(vocab_size)]
        onehot_pad = [0 for _ in range(vocab_size - 1)]
        onehot_pad.append(1)
        for j in range(seq_len):
            if j < pad_mask[i]:
                seq_mul.append(onehot_one)
                seq_add.append(onehot_zero)
            else:
                seq_mul.append(onehot_zero)
                seq_add.append(onehot_pad)
        pad_mask_mul.append(seq_mul)
        pad_mask_add.append(seq_add)
    pad_mask_mul = torch.tensor(pad_mask_mul, dtype=torch.int64)
    pad_mask_add = torch.tensor(pad_mask_add, dtype=torch.int64)
    return pad_mask_mul.to(device), pad_mask_add.to(device)


def pad_after_end_token(g_output_t, pad_mask_mul, pad_mask_add):
    g_output_t = g_output_t * pad_mask_mul
    g_output_t = g_output_t + pad_mask_add
    g_output_t = g_output_t.permute(1, 0, 2)
    return g_output_t


def get_act_distribution(g_output, aut_seqs):
    g_output_t_act = g_output.sum(0)
    g_output_t_act = g_output_t_act.sum(0)
    g_authentic_act = aut_seqs.sum(0)
    g_authentic_act = g_authentic_act.sum(0)
    return g_output_t_act, g_authentic_act


def reverse_torch_to_list(seqs, vocab_num):
    result = []
    for seq in seqs:
        seq_i = []
        for i in seq:
            if i != vocab_num:
                seq_i.append(i + 1)
            else:
                break
        result.append(seq_i)
    return result


def remove_end_token(seqs, vocab_num):
    result = []
    for seq in seqs:
        seq_i = []
        for i in seq:
            if i != vocab_num + 1 and i != 0:
                seq_i.append(i)
            else:
                break
        result.append(seq_i)
    return result


def write_generated_seqs(save_path, model, gen_seqs):
    with open(save_path + 'result_' + model + '.txt', 'a') as f:
        f.writelines(' '.join(str(token) for token in list) + '\n' for list in gen_seqs)
