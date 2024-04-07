import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_freq_dict(seqs):
    freqdict = {}
    for line in seqs:
        n = len(line)
        if n not in freqdict:
            freqdict[n] = 1
        else:
            freqdict[n] += 1
    return freqdict


def get_length_stats(seqs):
    len_list = []
    for seq in seqs:
        length = len(seq)
        len_list.append(length)

    mean = sum(len_list) / len(seqs)
    variance = sum([((x - mean) ** 2) for x in len_list]) / len(seqs)
    stddev = variance ** 0.5
    max_len = max(len_list)

    return mean, stddev, max_len


def write(item, save_path):
    with open(save_path + 'dif_log.txt', 'a') as filehandle:
        filehandle.write('%s\n' % item)


def save_len_difference(gen_seqs, aut_seqs, save_path):
    gen_mean, gen_std, gen_max_len = get_length_stats(gen_seqs)
    aut_mean, aut_std, aut_max_len = get_length_stats(aut_seqs)

    write('aut_mean: ' + str(aut_mean), save_path)
    write('aut_std: ' + str(aut_std), save_path)
    write('syn_mean: ' + str(gen_mean), save_path)
    write('syn_std: ' + str(gen_std), save_path)
    seq_dif = abs(aut_mean-gen_mean)+abs(aut_std-gen_std)
    write('seq_dif: ' + str(seq_dif), save_path)

    print('syn_mean: ' +str(gen_mean))
    print('aut_mean: ' + str(aut_mean))
    print('syn_std: ' + str(gen_std))
    print('aut_std: ' +str(aut_std))

    save_len_diff_figure(gen_seqs, aut_seqs, save_path)


def save_len_diff_figure(gen_seqs, aut_seqs, save_path):
    gen_freq_dict = get_freq_dict(gen_seqs)
    aut_freq_dict = get_freq_dict(aut_seqs)
    all_freq_dict = {}
    _, _, gen_max_len = get_length_stats(gen_seqs)
    _, _, aut_max_len = get_length_stats(aut_seqs)
    max_len = max(gen_max_len, aut_max_len)
    for seq_len in aut_freq_dict:
        if seq_len not in gen_freq_dict:
            all_freq_dict[seq_len] = [aut_freq_dict[seq_len]/len(aut_seqs), 0]
        else:
            all_freq_dict[seq_len] = [aut_freq_dict[seq_len]/len(aut_seqs), gen_freq_dict[seq_len]/len(gen_seqs)]

    for seq_len in gen_freq_dict:
        if seq_len not in all_freq_dict:
            all_freq_dict[seq_len] = [0, gen_freq_dict[seq_len]/len(gen_seqs)]

    my_df = pd.DataFrame([[k, *v] for k, v in all_freq_dict.items()],
                         columns=['sequence length', 'authentic processes', 'semi-synthetic processes'])
    my_df = my_df.sort_values(by=['sequence length'])

    # draw histogram
    SMALL_SIZE = 15
    MEDIUM_SIZE = 15
    BIGGER_SIZE = 18

    plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    barWidth = 1

    bars1 = my_df['authentic processes']
    bars2 = my_df['semi-synthetic processes']

    max_height = max(bars1.max(), bars2.max()) + 0.02

    r1 = my_df['sequence length']
    r2 = [x + barWidth for x in r1]

    plt.figure(figsize=(10, 6))
    plt.bar(r1, bars1, label='Authentic', width=barWidth, color='blue', alpha=0.4)
    plt.bar(r2, bars2, label='Synthetic', width=barWidth, color='red', alpha=0.3)

    plt.title('Sequence Length Distribution')
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.xticks(np.arange(0, max_len + 1, step=5))
    plt.xticks(rotation=75)
    plt.ylim(0, max_height)
    plt.legend()
    plt.savefig(save_path + 'length_distribution.png')
    # plt.show()
    plt.close()


def get_seqs_from_path(path):
    f = open(path)
    all_seq = [[int(ind) for ind in line.split()] for line in f]
    return all_seq


def get_length_dif(aut_path, seq_path, save_path):
    aut_seqs = get_seqs_from_path(aut_path)
    gen_seqs = get_seqs_from_path(seq_path)
    save_len_difference(gen_seqs, aut_seqs, save_path)

