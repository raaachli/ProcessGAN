import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset, DataLoader
from utils.data_loader import load_ar_data
import models.lstm as lstm
import models.gru as gru
from eval.variance_eval import save_variance_dif
from utils.data_prepare import prepare_ar_data
from eval.act_dist_eval import save_act_difference
from eval.length_eval import save_len_difference
# torch.set_printoptions(profile="full")


class RNNs:
    def __init__(self, res_path, save_path, model, config, gen_num):
        self.res_path = res_path
        self.save_path = save_path
        self.config = config
        self.model = model
        self.gen_num = gen_num
        self.seq_num = config['seq_num']
        self.seq_len = config['seq_len']
        self.vocab_num = config['vocab_num']
        self.emb_size = config['emb_size']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.n_hid = config['n_hid']
        self.lr = config['lr']
        self.seed = config['seed']
        self.train_size = config['train_size']
        self.test_size = config['test_size']
        self.valid_size = config['valid_size']

    def train_MLE(self, gen, gen_opt, input, target, epochs, model):

        dataset = load_ar_data(input, target)
        train_data, val_data, test_data = torch.utils.data.random_split(dataset, (self.train_size, self.valid_size, self.test_size), generator=torch.Generator().manual_seed(self.seed))
        train_dataloader = DataLoader(train_data, batch_size=self.batch_size, drop_last=False, shuffle=False, num_workers=1)
        val_dataloader = DataLoader(val_data, batch_size=self.batch_size, drop_last=False, shuffle=False, num_workers=1)
        test_dataloader = DataLoader(test_data, batch_size=self.batch_size, drop_last=False, shuffle=False, num_workers=1)

        loss_log = []

        patience = 20
        p_flag = 0
        test_loss_l = 100

        for epoch in range(epochs):
            total_loss = 0
            gen.train()

            for i, item in enumerate(train_dataloader):
                inp, target = item
                gen_opt.zero_grad()
                loss = gen.NLLLoss(inp, target)
                loss.backward()
                gen_opt.step()
                total_loss += loss.data.item()

            val_loss = self.evaluate(gen, val_dataloader)
            loss_log.append([total_loss/len(train_dataloader), val_loss/len(val_dataloader)])

            print('-' * 89)
            str_1 = ('| epoch {:3d} | training loss {:5.4f} | val loss {: 5.4f} | '
                  .format(epoch, total_loss/len(train_dataloader), val_loss/len(val_dataloader)))
            print(str_1)
            print('-' * 89)

            # early stopping
            test_loss_c = val_loss
            if test_loss_c > test_loss_l:
                p_flag += 1
            if p_flag > patience:
                break
            test_loss_l = test_loss_c

        test_loss = self.evaluate(gen, test_dataloader)

        str_2 = ('| test loss {: 5.4f} | '
              .format(test_loss/len(test_dataloader)))
        print(str_2)

        # plot the loss figure
        fig, ax = plt.subplots()
        losses = np.array(loss_log)
        plt.plot(losses.T[0], label='train loss')
        plt.plot(losses.T[1], label='val loss')
        plt.xlabel('epochs')
        plt.ylabel('losses')
        plt.legend()
        fig.savefig(self.save_path + model + '_loss.png')

        # generate samples
        sam = gen.sample(self.gen_num, start_letter=0)
        sam = sam.tolist()
        gen_list = self.remove_end_token(sam, self.vocab_num)

        # evaluate the synthetic samples
        test_seqs = next(iter(test_dataloader))[1].tolist()
        test_list = self.remove_end_token(test_seqs, self.vocab_num)
        self.result_eval(gen_list, test_list)

        return gen_list

    def remove_end_token(self, seqs, vocab_num):
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

    def evaluate(self, gen, test_dataloader):
        total_loss = 0
        gen.eval()
        with torch.no_grad():
            for i, item in enumerate(test_dataloader):
                inp, target = item
                loss = gen.NLLLoss(inp, target)
                total_loss += loss.data.item()
        return total_loss

    def write_gen_seqs(self, gen_seqs):
        with open(self.save_path + 'result_' + self.model + '.txt', 'a') as f:
            f.writelines(' '.join(str(token) for token in list) + '\n' for list in gen_seqs)

    def result_eval(self, gen_list, test_list):
        save_len_difference(gen_list, test_list, self.save_path)
        save_act_difference(gen_list, test_list, self.save_path)
        save_variance_dif(gen_list, test_list, self.save_path)

    def run(self):
        np.random.seed(self.seed)
        random.seed(self.seed)

        input_data, target_data = prepare_ar_data(self.res_path, self.seq_len,self.vocab_num, start_token=0)

        if self.model == 'gru':
            gen = gru.GRU_Generator(self.emb_size, self.n_hid, self.vocab_num, self.seq_len, gpu=False)
        if self.model == 'lstm':
            gen = lstm.LSTM_Generator(self.emb_size, self.n_hid, self.vocab_num, self.seq_len, gpu=False)

        gen_optimizer = optim.Adam(gen.parameters(), lr=self.lr)
        result_seqs = self.train_MLE(gen, gen_optimizer, input_data, target_data, self.epochs, self.model)
        self.write_gen_seqs(result_seqs)





