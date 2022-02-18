import random
import os


def get_data(path):
    f = open(path)
    all_seq = []
    for line in f:
        seq = []
        line = line.split()
        for token in line:
            seq.append(token)
        all_seq.append(seq)
    return all_seq


def add_tokens(error_rate, all_seqs, num_vocab):
    all_seq = []
    for sequence in all_seqs:
        n = len(sequence)
        pos = int(n * error_rate)
        random_tokens = random.choices(range(1, num_vocab+1), k=pos)
        random_positions = random.choices(range(0, n), k=pos)
        random_dict = {random_positions[k]: random_tokens[k] for k in range(len(random_tokens))}
        i = 0
        seq = []
        for token in sequence:
            if i in random_dict:
                seq.append(token)
                seq.append(str(random_dict[i]))
                i += 1
            else:
                seq.append(token)
                i += 1
        all_seq.append(seq)
    return all_seq


def delete_tokens(error_rate, all_seqs):
    all_seq = []
    for line in all_seqs:
        n = len(line)
        pos = int(n * error_rate)
        random_positions = random.choices(range(0, n), k=pos)
        i = 0
        seq = []
        for token in line:
            if i in random_positions:
                i += 1
            else:
                seq.append(token)
                i += 1
        all_seq.append(seq)
    return all_seq


def change_tokens(error_rate, all_seqs, num_vocab):
    all_seq = []
    for sequence in all_seqs:
        n = len(sequence)
        pos = int(n * error_rate)
        random_tokens = random.choices(range(1, num_vocab), k=pos)
        random_positions = random.choices(range(0, n), k=pos)
        random_dict = {random_positions[k]: random_tokens[k] for k in range(len(random_tokens))}
        i = 0
        seq = []
        for token in sequence:
            if i in random_dict:
                seq.append(str(random_dict[i]))
                i += 1
            else:
                seq.append(token)
                i += 1
        all_seq.append(seq)
    return all_seq


def switch_tokens(error_rate, all_seqs, num_vocab):
    all_seq = []
    for sequence in all_seqs:
        n = len(sequence)
        pos = int(n * error_rate)
        half_pos = int(pos/2)
        random_positions_1 = random.choices(range(0, n), k=half_pos)
        random_positions_2 = random.choices(range(0, n), k=half_pos)
        seq = []
        for i in range(half_pos):
            sequence[random_positions_1[i]], sequence[random_positions_2[i]] = sequence[random_positions_2[i]], sequence[random_positions_1[i]]
    return all_seqs


def write_file(seqs, save_path, result_file):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    for seq in seqs:
        with open(save_path + result_file + '.txt', 'a') as f:
            buffer = ' '.join(seq) + "\n"
            # print(buffer)
            f.write(buffer)



def main():
    error_rate = 0.1
    path = 'data/data_all/data_seq/BPI_2.txt'
    num_vocab = 37
    all_seq = get_data(path)
    save_path = 'data/data_hq/'
    result_file = 'BPI_2'

    random.shuffle(all_seq)

    all_seqes = []

    for add_time in range(5):
        all_seqs = []
        all_seqs = add_tokens(error_rate, all_seq, num_vocab)
        all_seqs = delete_tokens(error_rate, all_seqs)
        all_seqs = switch_tokens(error_rate, all_seqs, num_vocab)

        for time in range(2):
            all_seqs = add_tokens(error_rate, all_seqs, num_vocab)
            all_seqs = delete_tokens(error_rate, all_seqs)
            all_seqs = switch_tokens(error_rate, all_seqs, num_vocab)

        all_seqes.extend(all_seqs)

    random.shuffle(all_seqes)

    max_len = max([len(all_seqes[i]) for i in range(len(all_seqes))])
    print(max_len)
    write_file(all_seqes, save_path, result_file)


if __name__ == '__main__':
    main()
