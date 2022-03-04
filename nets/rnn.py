import numpy as np
import torch
import torch.optim as optim
import random
import models.lstm as lstm
import models.gru as gru
from torch.utils.data import Dataset, DataLoader
from utils.data_loader import load_ar_data
from utils.data_prepare import prepare_ar_data
from utils.helper import write_generated_seqs, eval_result, plot_loss, remove_end_token


class RNNs:
    """RNN variants, including GRU model and LSTM model.

    Parameters:
        'vocab_num' : the size of vocabulary
        'emb_size'  : embedding dimension
        'n_hid'     : the dimension of the feedforward network model in RNN models
        'seq_len'   : the length of the longest sequence in dataset

    """
    def __init__(self, res_path, save_path, model, config, gen_num):
        self.res_path = res_path
        self.save_path = save_path
        self.config = config
        self.model = model
        self.gen_num = gen_num
        self.device = config['device']
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

    def train(self, input, target):
        # set the RNN based generator
        if self.model == 'gru':
            gen = gru.GRU_Generator(self.emb_size, self.n_hid, self.vocab_num, self.seq_len, self.device).to(self.device)
        if self.model == 'lstm':
            gen = lstm.LSTM_Generator(self.emb_size, self.n_hid, self.vocab_num, self.seq_len, self.device).to(self.device)
        gen_opt = optim.Adam(gen.parameters(), lr=self.lr)

        # set the dataloaders
        dataset = load_ar_data(input, target)
        train_data, val_data, test_data = torch.utils.data.random_split(dataset, (self.train_size, self.valid_size, self.test_size), generator=torch.Generator().manual_seed(self.seed))
        train_dataloader = DataLoader(train_data, batch_size=self.batch_size, drop_last=False, shuffle=False, num_workers=1)
        val_dataloader = DataLoader(val_data, batch_size=self.batch_size, drop_last=False, shuffle=False, num_workers=1)
        test_dataloader = DataLoader(test_data, batch_size=self.batch_size, drop_last=False, shuffle=False, num_workers=1)

        # set early stop patience
        patience = 20
        p_flag = 0
        val_loss_l = 100

        loss_log = []
        for epoch in range(self.epochs):
            total_loss = 0
            gen.train()

            for i, item in enumerate(train_dataloader):
                inp, target = item
                inp = inp.to(self.device)
                target = target.to(self.device)
                gen_opt.zero_grad()
                loss = gen.get_NLLLoss(inp, target)
                loss.backward()
                gen_opt.step()
                total_loss += loss.data.item()

            val_loss = self.evaluate(gen, val_dataloader)
            total_loss = total_loss/len(train_dataloader)
            val_loss = val_loss/len(val_dataloader)
            loss_log.append([total_loss, val_loss])

            print('| epoch {:3d} | training loss {:5.4f} | val loss {: 5.4f} | '
                  .format(epoch, total_loss, val_loss))

            # early stopping
            val_loss_c = val_loss
            if val_loss_c > val_loss_l:
                p_flag += 1
            if p_flag > patience:
                break
            val_loss_l = val_loss_c

        test_loss = self.evaluate(gen, test_dataloader)
        test_loss = test_loss/len(test_dataloader)
        print('| test loss {: 5.4f} | '
              .format(test_loss))

        # plot the loss figure
        plot_loss(self.save_path, loss_log, self.model + '_loss.png', "loss")

        # generate sequences
        samples = gen.generate(self.gen_num, start_letter=0)
        samples = samples.tolist()
        gen_list = remove_end_token(samples, self.vocab_num)

        # evaluate the synthetic samples
        test_seqs = next(iter(test_dataloader))[1].tolist()
        test_list = remove_end_token(test_seqs, self.vocab_num)
        eval_result(self.save_path, gen_list, test_list)

        # return the generated synthetic sequence list
        return gen_list

    def evaluate(self, gen, test_dataloader):
        total_loss = 0
        gen.eval()
        with torch.no_grad():
            for i, item in enumerate(test_dataloader):
                inp, target = item
                inp = inp.to(self.device)
                target = target.to(self.device)
                loss = gen.get_NLLLoss(inp, target)
                total_loss += loss.data.item()

        # return the total evaluation loss value
        return total_loss

    def run(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        input_data, target_data = prepare_ar_data(self.res_path, self.seq_len,self.vocab_num, start_token=0)
        result_seqs = self.train(input_data, target_data)
        write_generated_seqs(self.save_path, self.model, result_seqs)
