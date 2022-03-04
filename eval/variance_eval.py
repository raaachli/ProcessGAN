import editdistance


def write(item, save_path):
    with open(save_path + 'dif_log.txt', 'a') as filehandle:
        filehandle.write('%s\n' % item)


def get_variance(seqs):
    num_seq = len(seqs)
    ed_m = [[0 for _ in range(0, num_seq)] for _ in range(0, num_seq)]
    ed_m_norm = [[0 for _ in range(0, num_seq)] for _ in range(0, num_seq)]

    variance = 0

    for i in range(0, num_seq):
        for j in range(0, num_seq):
            ed_m[i][j] = editdistance.eval(seqs[i], seqs[j])
            if len(seqs[i]) == 0 and len(seqs[j]) == 0:
                ed_m_norm[i][j] = 0
            else:
                ed_m_norm[i][j] = ed_m[i][j] / (len(seqs[i]) + len(seqs[j]))
            variance += ed_m_norm[i][j]
    variance = variance / (2 * num_seq * num_seq)

    return variance


def save_variance_dif(gen_seqs, aut_seqs, save_path):
    gen_variance = get_variance(gen_seqs)
    aut_variance = get_variance(aut_seqs)
    diff = aut_variance - gen_variance
    write('aut_variance: ' + str(aut_variance), save_path)
    write('syn_variance: ' + str(gen_variance), save_path)
    write('variance difference: ' + str(diff), save_path)
    write('\n', save_path)
    print('aut_variance: ' + str(aut_variance))
    print('syn_variance: ' + str(gen_variance))


def get_seq_from_path(path):
    f = open(path)
    all_seq = [[int(ind) for ind in line.split()] for line in f]
    return all_seq


def get_variance_dif(aut_path, gen_path, save_path):
    aut_seqs = get_seq_from_path(aut_path)
    gen_seqs = get_seq_from_path(gen_path)
    save_variance_dif(gen_seqs, aut_seqs, save_path)


if __name__ == '__main__':
    aut_path = 'data/data_all/data_seq/BPI_2.txt'
    seq_path = 'data/data_test/BPI_1.txt'
    save_path = 'result/'
    get_variance_dif(aut_path, seq_path, save_path)
