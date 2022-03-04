import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_freq_dict(seqs):
    freqdict = {}
    for line in seqs:
        for i in range(len(line)):
            ind = line[i]
            if ind not in freqdict:
                freqdict[ind] = 1
            else:
                freqdict[ind] += 1
    return freqdict


def write(item, save_path):
    with open(save_path + 'dif_log.txt', 'a') as filehandle:
        filehandle.write('%s\n' % item)


def get_act_stats(gen_seqs, aut_seqs):
    # build distribution dataframe
    gen_freq_dict = get_freq_dict(gen_seqs)
    aut_freq_dict = get_freq_dict(aut_seqs)

    all_freq_dict = {}

    for activity in aut_freq_dict:
        if activity not in all_freq_dict:
            all_freq_dict[activity] = [aut_freq_dict[activity]]

    for activity in all_freq_dict:
        if int(activity) in gen_freq_dict:
            all_freq_dict[activity].append(gen_freq_dict[int(activity)])
        else:
            all_freq_dict[activity].append(0)

    # build frequency dataframe
    frequency_dict = {}
    aut_count = 0
    gen_count = 0

    for act in all_freq_dict:
        aut_count += all_freq_dict[act][0]
        gen_count += all_freq_dict[act][1]

    act_type_distance = 0  # the summation of activity type fraction difference
    for act in all_freq_dict:
        if act not in frequency_dict:
            frequency_dict[act] = [all_freq_dict[act][0] / aut_count, all_freq_dict[act][1] / gen_count]
            act_type_distance += abs(all_freq_dict[act][0] / aut_count - all_freq_dict[act][1] / gen_count)
    print('act difference: ' + str(act_type_distance))

    return act_type_distance, frequency_dict


def save_act_dif_figure(frequency_dict, save_path):
    my_df = pd.DataFrame([[k, *v] for k, v in frequency_dict.items()], columns=['activity', 'real', 'syn'])

    # draw histogram
    barWidth = 0.25

    bars1 = my_df['real']
    bars2 = my_df['syn']

    max_height = max(bars1.max(), bars2.max()) + 0.05

    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]

    plt.figure(figsize=(30, 15))
    plt.bar(r1, bars1, label='authentic', color='g', width=barWidth)
    plt.bar(r2, bars2, label='synthetic', color='r', width=barWidth)

    plt.rc('font', size=30)

    plt.title('Activity Frequency')
    plt.xlabel('Activity Type', fontsize=30)
    plt.ylabel('Frequency', fontsize=30)
    plt.xticks([r + barWidth for r in range(len(bars1))], np.arange(start=1, stop=len(bars1) + 1), fontsize=20)
    plt.xticks(rotation=90, fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0, max_height)
    plt.legend()
    plt.savefig(save_path + 'act_distribution.png')
    plt.show()


def save_act_difference(gen_seqs, aut_seqs, save_path):
    act_type_distance, frequency_dict = get_act_stats(gen_seqs, aut_seqs)
    write('act difference: ' + str(act_type_distance), save_path)
    save_act_dif_figure(frequency_dict, save_path)


def get_seqs_from_path(path):
    f = open(path)
    all_seq = [[int(ind) for ind in line.split()] for line in f]
    return all_seq


def get_act_dif(aut_path, gen_path, save_path):
    aut_seqs = get_seqs_from_path(aut_path)
    gen_seqs = get_seqs_from_path(gen_path)
    save_act_difference(gen_seqs, aut_seqs, save_path)


if __name__ == '__main__':
    aut_path = 'data/data_all/data_seq/SEP.txt'
    seq_path = 'results/result_final/SEP/PGAN/result_trans.txt'
    save_path = 'results/'
    get_act_dif(aut_path, seq_path, save_path)
