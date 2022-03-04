import datetime
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from models.transformer_ar import TransformerModel
from utils.data_prepare import prepare_ar_data
from utils.data_loader import load_ar_data
from utils.helper import eval_result, remove_end_token, write_generated_seqs


class Transformer_AR:
    """The autoregressive training of Transformer.

    Parameters:
        'vocab_num' : the size of vocabulary
        'emb_size'  : embedding dimension
        'n_hid'     : the dimension of the feedforward network model in nn.TransformerEncoder
        'drop_out'  : the dropout value
        'n_layer'   : the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        'n_head'    : the number of heads in the multi-head-attention models
        'lr_gen'    : learning rate
        'step'      : the step size that auto-regressively feed the sequences into the generator

    """
    def __init__(self, res_path, save_path, config, gen_num):
        self.res_path = res_path
        self.save_path = save_path
        self.config = config
        self.gen_num = gen_num
        self.vocab_num = config['vocab_num']
        self.n_inp = self.vocab_num + 2
        self.pad_ind = self.vocab_num + 1
        self.emb_size = config['emb_size']
        self.n_hid = config['n_hid']
        self.n_layer = config['n_layer']
        self.n_head = config['n_head']
        self.drop_out = config['drop_out']
        self.lr_gen = config['lr_gen']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.seed = config['seed']
        self.device = config['device']
        self.seq_len = config['seq_len']
        self.seq_num = config['seq_num']
        self.train_size = config['train_size']
        self.test_size = config['test_size']
        self.valid_size = config['valid_size']
        self.step = config['step']

    def train(self, input_data, target_data):
        # set Transformer based generator model
        model = TransformerModel(self.n_inp, self.emb_size, self.n_head, self.n_hid, self.n_layer, self.pad_ind, self.drop_out).to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr_gen)

        # set the dataloaders
        dataset = load_ar_data(input_data, target_data)
        train_data, val_data, test_data = torch.utils.data.random_split(dataset, (self.train_size, self.valid_size, self.test_size), generator=torch.Generator().manual_seed(self.seed))
        train_dataloader = DataLoader(train_data, batch_size=self.batch_size, drop_last=False, shuffle=False, num_workers=1)
        val_dataloader = DataLoader(val_data, batch_size=self.batch_size, drop_last=False, shuffle=False, num_workers=1)
        test_dataloader = DataLoader(test_data, batch_size=self.batch_size, drop_last=False, shuffle=False, num_workers=1)

        # set the early stopping patience
        patience = 20
        p_flag = 0
        val_loss_l = 100

        for epoch in range(self.epochs):
            model.train()
            total_loss = 0
            start = time.time()

            for i, item in enumerate(train_dataloader):
                data, targets = item
                data = data.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                loss = model.getLoss(data, targets, self.step)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            total_loss = total_loss/len(train_dataloader)

            val_loss = self.evaluate(model, val_dataloader)
            val_loss = val_loss/len(val_dataloader)
            end = time.time()
            es_remain_time = datetime.timedelta(seconds=(end-start)*(self.epochs-epoch))
            print("epoch {:3d} | train loss {:5.4f} | validation loss {:5.4f} | time {}"
                  .format(epoch, total_loss, val_loss, es_remain_time))

            val_loss_c = val_loss
            if val_loss_c > val_loss_l:
                p_flag += 1
            if p_flag > patience:
                break
            val_loss_l = val_loss_c

        test_loss = self.evaluate(model, test_dataloader)
        test_loss = test_loss/len(test_dataloader)
        print("test loss {:5.4f}".format(test_loss))

        gen_list = self.generate_seqs(model)
        test_seqs = next(iter(test_dataloader))[1].tolist()
        test_list = remove_end_token(test_seqs, self.vocab_num)

        # evaluate the generated sequences
        eval_result(self.save_path, gen_list, test_list)
        write_generated_seqs(self.save_path, 'result_transformer_ar', gen_list)

    def evaluate(self, eval_model, eval_dataloader):
        eval_model.eval()
        total_loss = 0.
        with torch.no_grad():
            for i, item in enumerate(eval_dataloader):
                data, targets = item
                data = data.to(self.device)
                targets = targets.to(self.device)
                loss = eval_model.getLoss(data, targets, self.step)
                total_loss += loss

        # return the total loss of evaluation data
        return total_loss

    def generate_next_token(self, start, model):
        start_tens = torch.Tensor(start).long().to(self.device)
        src_mask = model.generate_square_subsequent_mask(start_tens.size(0)).to(self.device)
        out = model.forward(start_tens, src_mask)
        out = out.view(-1, self.n_inp)
        out = F.log_softmax(out, 1)
        out_ind = torch.multinomial(torch.exp(out), 1).long()
        gen_full_seq = []
        for token in out_ind:
            gen_full_seq.append(token.item())
        gen_token = gen_full_seq[-1]

        # return the last generated token
        return gen_token

    def generate_seqs(self, model):
        model.eval()
        count = 0
        gen_list = []
        while count < self.gen_num:
            start = [[0]]
            sample = []
            i = 0
            while i < self.seq_len:
                next_token = self.generate_next_token(start, model)
                if next_token != self.vocab_num + 1 and next_token != 0:
                    sample.append(next_token)
                    start.append([next_token])
                    i += 1
                else:
                    break
            count += 1
            gen_list.append(sample)

        # return the generated sequences
        return gen_list

    def run(self):
        input_data, target_data = prepare_ar_data(self.res_path, self.seq_len, self.vocab_num, start_token=0)
        self.train(input_data, target_data)
