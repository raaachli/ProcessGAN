import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def save_time_difference(aut_seqs, aut_time_seqs, gen_seqs, gen_time_seqs, save_path, act_dict):
    save_act_time_stat(aut_seqs, aut_time_seqs, gen_seqs, gen_time_seqs, save_path, act_dict)
    plot_positions_violin(gen_seqs, gen_time_seqs, save_path, act_dict)
    plot_positions_violin(aut_seqs, aut_time_seqs, save_path+'AUT_', act_dict)
    save_time_mse(aut_time_seqs, gen_time_seqs, save_path)


def get_act_time_stat(act_seqs, act_time_seqs, act_dict):
    data = {}
    for label_set, increment_set in zip(act_seqs, act_time_seqs):
        positions = np.cumsum(increment_set)
        for label, position in zip(label_set, positions):
            data.setdefault(act_dict[label], []).append(position)

    means = []
    quantile_90 = []
    for act in act_dict:
        if act_dict[act] in data:
            data_act = np.array(data[act_dict[act]])
            means.append(np.mean(data_act))
            quantile_90.append(np.quantile(data_act, 0.90))
        else:
            means.append(-1)
            quantile_90.append(-1)

    return means, quantile_90


def save_act_time_stat(aut_seqs, aut_time_seqs, gen_seqs, gen_time_seqs, save_path, act_dict):
    aut_means, aut_quantile_90 = get_act_time_stat(aut_seqs, aut_time_seqs, act_dict)
    gen_means, gen_quantile_90 = get_act_time_stat(gen_seqs, gen_time_seqs, act_dict)

    means_dif = np.sum(np.array([abs(aut_means[i]-gen_means[i]) for i in range(len(aut_means))]))
    quantile_dif = np.sum(np.array([abs(aut_quantile_90[i]-gen_quantile_90[i]) for i in range(len(aut_means))]))

    write('time means_dif: ' + str(means_dif), save_path)
    write('time quantile_dif: ' + str(quantile_dif), save_path)


def get_mse(aut_time_seqs, gen_time_seqs):

    def pad_to_same_length(a, b):
        """Pad arrays a and b to have the same length."""
        max_length = max(len(a), len(b))
        a_padded = np.pad(a, (0, max_length - len(a)))
        b_padded = np.pad(b, (0, max_length - len(b)))
        return a_padded, b_padded

    # Compute MSE for each pairo f lists
    mse_values = []
    for a, b in zip(aut_time_seqs, gen_time_seqs):
        a_padded, b_padded = pad_to_same_length(np.array(a), np.array(b))
        a_cumsum = a_padded.cumsum()
        b_cumsum = b_padded.cumsum()
        mse_values.append(np.mean((a_cumsum - b_cumsum) ** 2))

    # Compute the average MSE
    average_mse = np.mean(mse_values)

    return average_mse


def write(item, save_path):
    with open(save_path + 'dif_log.txt', 'a') as filehandle:
        filehandle.write('%s\n' % item)


def save_time_mse(aut_seqs, gen_seqs, save_path):
    mse = get_mse(aut_seqs, gen_seqs)
    write('time mse: ' + str(mse), save_path)


def plot_positions_violin(act_seqs, act_time_seqs, save_path, act_dict):
    data = []
    for label_set, increment_set in zip(act_seqs, act_time_seqs):
        positions = np.cumsum(increment_set)
        for label, position in zip(label_set, positions):
            data.append([act_dict[label], position])

    df = pd.DataFrame(data, columns=["Activity", "Position"])
    order = []
    for act in act_dict:
        order.append(act_dict[act])

    # Create violin plot
    plt.figure(figsize=(15, 25))
    ax = sns.violinplot(
                        x="Position",
                        y="Activity",
                        data=df,
                        order=order,
                        scale="width",
                        cut=0,
                        orient='h',
                        inner="box",
                        showmeans = True,
                        showmedians = True
                        )

    ax.set_ylabel(ax.get_ylabel(), fontsize=30)

    counts = df['Activity'].value_counts()
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax2 = ax.twinx()
    ax2.set_ylabel("Count", fontsize=30, rotation=270, labelpad=50)
    # Make sure it has the same limits and ticks as the primary y-axis
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(ax.get_yticks())

    # Set the labels of the secondary y-axis to the counts
    labels = [f'{counts.get(round(tick), 0)}' for tick in ax2.get_yticks()]
    ax2.set_yticklabels(labels, fontsize=20)

    # Hide the ticks of the secondary y-axis
    ax2.tick_params(axis='y', which='both', length=0)

    # plt.ylabel('Activity')
    ax.set_xlabel('Position', fontsize=30, labelpad=30)
    # plt.title('Positions of Activities', fontsize=50, pad=30)
    plt.tight_layout()
    # Ensure you have defined save_path elsewhere or provide a full path directly
    plt.savefig(save_path + 'timestamp_act_violin_base.png')
    # plt.show()
    plt.close()


def get_seqs_from_path(path):
    f = open(path)
    all_seq = [[int(ind) for ind in line.split()] for line in f]
    return all_seq


def get_time_seqs_from_path(path):
    f = open(path)
    all_seq = [[float(ind) for ind in line.split()] for line in f]
    all_seq_norm = []
    for i in range(len(all_seq)):
        if len(all_seq[i]) == 0:
            continue
        all_seq[i][0] = 0
        all_seq_i = normalize(all_seq[i])
        all_seq_norm.append(all_seq_i)
    return all_seq_norm


def normalize(values):
    total = sum(values)+(1e-9)
    return [value / total for value in values]


def get_time_dif(aut_path, aut_time_path, gen_path, gen_time_path, save_path, act_dict):
    aut_seqs = get_seqs_from_path(aut_path)
    gen_seqs = get_seqs_from_path(gen_path)
    aut_time_seqs = get_time_seqs_from_path(aut_time_path)
    gen_time_seqs = get_time_seqs_from_path(gen_time_path)

    save_time_difference(aut_seqs, aut_time_seqs, gen_seqs, gen_time_seqs, save_path, act_dict)


def get_act_dict(dict_path):
    df = pd.read_csv(dict_path, header=None)
    act_dict = {}
    ind = 1
    for ind in range(len(df)):
        act_dict[ind+1] = df[0][ind]
    return act_dict
