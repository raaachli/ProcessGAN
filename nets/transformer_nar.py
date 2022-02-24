import random
import torch
from torch import nn
from eval.act_dist_eval import save_act_difference
from eval.length_eval import save_len_difference
from eval.variance_eval import save_variance_dif
from models.transformer_gen import TransformerModel
import torch.nn.functional as F
from utils.data_prepare import prepare_nar_data
from utils.data_loader import load_nar_data
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class Transformer_NAR:
    def __init__(self, res_path, save_path, config, gen_num):

        self.res_path = res_path
        self.save_path = save_path
        self.config = config
        self.gen_num = gen_num
        self.seq_len = config['seq_len']
        self.vocab_num = config['vocab_num']
        self.emb_size = config['emb_size']
        self.n_hid = config['n_hid']
        self.n_layer = config['n_layer']
        self.n_head = config['n_head']
        self.drop_out = config['drop_out']
        self.lr_gen = config['lr_gen']
        self.train_size = config['train_size']
        self.test_size = config['test_size']
        self.valid_size = config['valid_size']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.seed = config['seed']
        self.device = config['device']
        self.n_inp = self.vocab_num + 1
        self.pad_ind = self.vocab_num

    def generate_random_data(self, bs, vocab_size, seq_len):
        rand_data = []
        end_token = vocab_size
        for i in range(bs):
            randomlist = random.choices(range(0, end_token + 1), k=seq_len + 1)
            rand_data.append(randomlist)
        return rand_data

    def gen_data_from_rand(self, size, g_model, ntokens, device, result_file):
        gen_list = []
        for gen in range(size):
            gen_rand_set = self.generate_random_data(1, ntokens, self.seq_len)
            gen_rand_set = torch.tensor(gen_rand_set, dtype=torch.int64).to(device)
            gen_rand_set = torch.transpose(gen_rand_set, 0, 1)

            mask_len = gen_rand_set.size()[0]
            src_mask = g_model.generate_square_subsequent_mask(mask_len).to(device)

            g_output = g_model(gen_rand_set, src_mask)
            g_output2 = g_output.permute(1, 0, 2)

            out = F.gumbel_softmax(g_output2, tau=1, hard=True)
            out_list = out.tolist()

            seq = []
            for j in range(self.seq_len + 1):
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
        with open(self.save_path + result_file + '.txt', 'a') as f:
            f.writelines(' '.join(str(token) for token in list) + '\n' for list in gen_list)

        return gen_list

    def draw_loss(self, loss_log):
        fig, ax = plt.subplots()
        losses = np.array(loss_log)
        plt.plot(losses, label='loss')
        plt.xlabel('epochs')
        plt.ylabel('losses')
        plt.legend()
        fig.savefig(self.save_path + 'trans_nar_loss.png')

    def train(self, target):
        random.seed(self.seed)
        np.random.seed(self.seed)

        g_model = TransformerModel(self.n_inp, self.emb_size, self.n_head, self.n_hid, self.n_layer, self.pad_ind, self.drop_out).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(g_model.parameters(), lr=self.lr_gen)

        dataset = load_nar_data(target)
        train_data, val_data, test_data = torch.utils.data.random_split(dataset, (self.train_size, self.valid_size, self.test_size), generator=torch.Generator().manual_seed(self.seed))
        train_dataloader = DataLoader(train_data, batch_size=self.batch_size, drop_last=False, shuffle=False,num_workers=1)
        val_dataloader = DataLoader(val_data, batch_size=self.batch_size, drop_last=False, shuffle=False, num_workers=1)
        test_dataloader = DataLoader(test_data, batch_size=self.batch_size, drop_last=False, shuffle=False,num_workers=1)

        test_seqs = next(iter(test_dataloader)).tolist()
        test_list = self.reverse_to_list(test_seqs, self.vocab_num)

        loss_log = []

        for pre_epoch in range(1, self.epochs+1):
            g_model.train()
            rand_set = self.generate_random_data(self.train_size, self.vocab_num, self.seq_len)
            rand_set = torch.tensor(rand_set, dtype=torch.int64).to(self.device)
            total_loss = 0
            for i, item in enumerate(train_dataloader):
                tar_data = item
                tar_data = tar_data.to(self.device)
                batch = tar_data.size()[0]
                input_rand_data = rand_set[i:i + batch]
                input_rand_data = torch.transpose(input_rand_data, 0, 1)

                mask_len = input_rand_data.size()[0]
                src_mask = g_model.generate_square_subsequent_mask(mask_len).to(self.device)

                optimizer.zero_grad()
                output = g_model(input_rand_data, src_mask)
                output = output.permute(1, 2, 0)
                loss = criterion(output, tar_data)

                total_loss += loss

                loss.backward()
                optimizer.step()

            validation_loss = self.evaluate(g_model, val_dataloader, criterion)
            print('pre_pos epoch {:3d} | gen_loss {:5.2f} | val_loss {:5.2f}'.format(pre_epoch, total_loss/len(train_dataloader), validation_loss/len(val_dataloader)))
            loss_log.append(total_loss.detach().cpu()/len(train_dataloader))

            if pre_epoch % 10 == 0:
                # torch.save(g_model, self.save_path + str(pre_epoch) + '_g_model.pt')
                g_model.eval()
                with torch.no_grad():
                    gen_list = self.gen_data_from_rand(self.gen_num, g_model, self.vocab_num, self.device, 'result_transformer_nar_'+str(pre_epoch))
                    self.result_eval(gen_list, test_list)

        self.draw_loss(loss_log)

    def reverse_to_list(self, seqs, vocab_num):
        result = []
        for seq in seqs:
            seq_i = []
            for i in seq:
                if i != vocab_num:
                    seq_i.append(i+1)
                else:
                    break
            result.append(seq_i)
        return result

    def result_eval(self, gen_list, test_list):
        save_len_difference(gen_list, test_list, self.save_path)
        save_act_difference(gen_list, test_list, self.save_path)
        save_variance_dif(gen_list, test_list, self.save_path)

    def evaluate(self, model, eval_dataloader, criterion):
        total_loss = 0
        model.eval()
        with torch.no_grad():
            for i, item in enumerate(eval_dataloader):
                target = item
                target = target.to(self.device)
                batch = target.size()[0]
                rand_input = self.generate_random_data(batch, self.vocab_num, self.seq_len)
                rand_input = torch.tensor(rand_input, dtype=torch.int64).to(self.device)
                rand_input = torch.transpose(rand_input, 0, 1)
                mask_len = rand_input.size()[0]
                src_mask = model.generate_square_subsequent_mask(mask_len).to(self.device)
                output = model(rand_input, src_mask)
                output = output.permute(1, 2, 0)
                loss = criterion(output, target)
                total_loss += loss.data.item()
        return total_loss

    def run(self):
        target = prepare_nar_data(self.res_path, self.seq_len, self.vocab_num)
        self.train(target)
