import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from eval.act_dist_eval import save_act_difference
from eval.length_eval import save_len_difference
from eval.time_eval import save_time_difference

from datetime import datetime
from eval.variance_eval import save_variance_dif


def normalize_gen_timestamp(gen_list_time):
    gen_time_seqs_norm = []
    for i in range(len(gen_list_time)):
        if len(gen_list_time[i]) == 0:
            gen_time_seqs_norm.append([])
        else:
            gen_list_time[i][0] = 0
            all_seq_i = normalize(gen_list_time[i])
            gen_time_seqs_norm.append(all_seq_i)
    return gen_time_seqs_norm


def normalize(values):
    total = sum(values)+(1e-9)
    return [value / total for value in values]


def generate_random_data(bs, vocab_size, seq_len):
    rand_data = []
    end_token = vocab_size
    for i in range(bs):
        randomlist = random.choices(range(0, end_token + 1), k=seq_len)
        rand_data.append(randomlist)
    return rand_data


def gen_data_from_rand_baseline(size, g_model, ntokens, device, result_file, save_path, seq_len, aut_duration):
    gen_list = []
    gen_list_time = []

    for gen in range(size):
        random_duration = aut_duration[gen][0]
        random_duration = adjust_value(random_duration)
        random_duration_i = torch.tensor([[random_duration]]).to(device)
        gen_rand_set = generate_random_data(1, ntokens, seq_len)
        gen_rand_set = torch.tensor(gen_rand_set, dtype=torch.int64).to(device)
        gen_rand_set = torch.transpose(gen_rand_set, 0, 1)
        g_output, g_output_time = g_model(gen_rand_set, random_duration_i)
        g_output_time = g_output_time.squeeze()
        g_output_time = F.relu(g_output_time)
        g_output_time = g_output_time.tolist()
        g_output = g_output.permute(1, 0, 2)
        out = F.gumbel_softmax(g_output, tau=1, hard=True)
        out_list = out.tolist()
        seq = []
        for j in range(seq_len):
            for k in range(ntokens + 1):
                if out_list[0][j][k] == 1:
                    seq.append(k)
        sub_samp = []
        sub_time = []
        n = len(seq)
        for j in range(n):
            tok = seq[j]
            if tok != ntokens:
                sub_samp.append(tok + 1)
                sub_time.append(g_output_time[j])
            else:
                break
        gen_list.append(sub_samp)
        gen_list_time.append(sub_time)
    gen_list_time = normalize_gen_timestamp(gen_list_time)
    with open(save_path + result_file + '.txt', 'a') as f:
        f.writelines(' '.join(str(token) for token in list) + '\n' for list in gen_list)
    with open(save_path + result_file + '_time.txt', 'a') as f:
        f.writelines(' '.join(str(token) for token in list) + '\n' for list in gen_list_time)
    return gen_list, gen_list_time


def adjust_value(value):
    # Ensure the value is between 0 and 1
    if not (0 <= value <= 1):
        raise ValueError("Value should be between 0 and 1.")

    # Randomly choose a small value to adjust by
    delta = random.uniform(0.00, 0.01)  # Adjust this range as needed

    # Randomly decide to add or subtract
    if random.choice([True, False]):
        value += delta
    else:
        value -= delta

    # Ensure the adjusted value remains in [0,1]
    return min(1, max(0, value))


def gen_data_from_rand(size, g_model, ntokens, device, result_file, save_path, seq_len, aut_duration):
    gen_list = []
    gen_list_time = []

    for gen in range(size):
        random_duration = aut_duration[gen][0]
        random_duration = adjust_value(random_duration)
        random_duration_i = torch.tensor([[random_duration]]).to(device)
        gen_rand_set = generate_random_data(1, ntokens, seq_len)
        gen_rand_set = torch.tensor(gen_rand_set, dtype=torch.int64).to(device)
        gen_rand_set = torch.transpose(gen_rand_set, 0, 1)
        mask_len = gen_rand_set.size()[0]
        src_mask = g_model.generate_square_subsequent_mask(mask_len).to(device)
        g_output, g_output_time = g_model(gen_rand_set, src_mask, random_duration_i)
        g_output_time = g_output_time.squeeze()
        g_output_time = F.relu(g_output_time)
        g_output_time = g_output_time.tolist()
        g_output = g_output.permute(1, 0, 2)
        out = F.gumbel_softmax(g_output, tau=1, hard=True)
        out_list = out.tolist()
        seq = []
        for j in range(seq_len):
            for k in range(ntokens + 1):
                if out_list[0][j][k] == 1:
                    seq.append(k)
        sub_samp = []
        sub_time = []
        n = len(seq)
        for j in range(n):
            tok = seq[j]
            if tok != ntokens:
                sub_samp.append(tok + 1)
                sub_time.append(g_output_time[j])
            else:
                break
        gen_list.append(sub_samp)
        gen_list_time.append(sub_time)

    gen_list_time = normalize_gen_timestamp(gen_list_time)

    with open(save_path + result_file + '.txt', 'a') as f:
        f.writelines(' '.join(str(token) for token in list) + '\n' for list in gen_list)
    with open(save_path + result_file + '_time.txt', 'a') as f:
        f.writelines(' '.join(str(token) for token in list) + '\n' for list in gen_list_time)

    return gen_list, gen_list_time


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
    plt.close()


def get_act_dict(dict_path):
    df = pd.read_csv(dict_path, header=None)
    act_dict = {}
    ind = 1
    for ind in range(len(df)):
        act_dict[ind+1] = df[0][ind]
    return act_dict


def eval_result(save_path, gen_list, gen_list_time, test_list, test_list_time, vocab_num, data):
    act_dict = get_act_dict('data/data_info/'+data+'/act_dict.csv')
    save_len_difference(gen_list, test_list, save_path)
    save_act_difference(gen_list, test_list, save_path, vocab_num)
    save_variance_dif(gen_list, test_list, save_path)
    save_time_difference(test_list, test_list_time, gen_list, gen_list_time, save_path, act_dict)


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

def reverse_torch_to_list_time(seqs, seqs_time, vocab_num):
    result = []
    result_time = []

    for i in range(len(seqs)):
        seq = seqs[i]
        seq_time = seqs_time[i]
        seq_i = []
        seq_time_i = []
        for j in range(len(seq)):
            act_j = seq[j]
            if act_j != vocab_num:
                seq_i.append(act_j + 1)
                seq_time_i.append(seq_time[j])
            else:
                break
        result.append(seq_i)
        result_time.append(seq_time_i)
    return result, result_time

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

def get_timestamp():
    dateTimeObjlocal = datetime.now()
    currDateTime = (
                "Received Timestamp: = " + str(dateTimeObjlocal.year) + str(dateTimeObjlocal.month) +str(
            dateTimeObjlocal.day) + str(dateTimeObjlocal.hour) + str(dateTimeObjlocal.minute) + str(
            dateTimeObjlocal.second) + "\n")
    print("System Timestamp: ", dateTimeObjlocal)
    save_time = str(dateTimeObjlocal.month) + str(dateTimeObjlocal.day) + str(dateTimeObjlocal.hour) + str(dateTimeObjlocal.minute)

    return save_time

