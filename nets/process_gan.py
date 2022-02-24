import datetime
import random
import torch
from torch import nn
from models.transformer_gen import TransformerModel
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from utils.data_loader import load_nar_data
from utils.data_prepare import prepare_onehot_aut_data, prepare_dis_label, prepare_nar_data
import time
from models.transformer_dis import TransformerModel_DIS
import numpy as np
import matplotlib.pyplot as plt
from eval.act_dist_eval import save_act_difference
from eval.length_eval import save_len_difference
from eval.variance_eval import save_variance_dif


class ProcessGAN:
    def __init__(self, res_path, save_path, config, gen_num, mode):
        self.res_path = res_path
        self.save_path = save_path
        self.config = config
        self.gen_num = gen_num
        self.mode = mode
        self.seq_len = config['seq_len']
        self.vocab_num = config['vocab_num']
        self.emb_size = config['emb_size']
        self.n_hid = config['n_hid']
        self.n_layer = config['n_layer']
        self.n_head_g = config['n_head_g']
        self.n_head_d = config['n_head_d']
        self.drop_out = config['drop_out']
        self.gd_ratio = config['gd_ratio']
        self.lr_gen = config['lr_gen']
        self.lr_dis = config['lr_dis']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.seed = config['seed']
        self.device = config['device']
        self.test_size = config['test_size']
        self.valid_size = config['valid_size']
        self.train_size = config['train_size']
        self.w_a = config['w_a']
        self.n_inp = self.vocab_num + 1
        self.pad_ind = self.vocab_num

    def generate_random_data(self, bs, vocab_size, seq_len):
        rand_data = []
        end_token = vocab_size
        for i in range(bs):
            randomlist = random.choices(range(0, end_token+1), k = seq_len+1)
            rand_data.append(randomlist)
        return rand_data

    def write_log(self, log, file_name):
        with open(self.save_path + file_name, 'a') as filehandle:
            for listitem in log:
                filehandle.write('%s\n' % listitem)

    def draw(self, log,file_name,type):
        fig, ax = plt.subplots()
        losses = np.array(log)
        plt.plot(losses, label=type)
        plt.xlabel('epochs')
        plt.ylabel(type)
        plt.legend()
        fig.savefig(self.save_path + file_name)

    def get_pad_mask(self, output, batch_size, seq_len, vocab_size, padding_ind):
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
        return pad_mask_mul.to(self.device), pad_mask_add.to(self.device)

    def pad_after_end_token(self, g_output_t, pad_mask_mul, pad_mask_add):
        g_output_t = g_output_t * pad_mask_mul
        g_output_t = g_output_t + pad_mask_add
        g_output_t = g_output_t.permute(1, 0, 2)
        return g_output_t

    def gen_data_from_rand(self, size, g_model, ntokens, device, result_file):
        gen_list = []
        for gen in range(size):
            gen_rand_set = self.generate_random_data(1, ntokens, self.seq_len)
            gen_rand_set = torch.tensor(gen_rand_set, dtype=torch.int64).to(device)
            gen_rand_set = torch.transpose(gen_rand_set, 0, 1)
            mask_len = gen_rand_set.size()[0]
            src_mask = g_model.generate_square_subsequent_mask(mask_len).to(device)
            g_output = g_model(gen_rand_set, src_mask)
            g_output = g_output.permute(1, 0, 2)
            out = F.gumbel_softmax(g_output, tau=1, hard=True)
            out_list = out.tolist()
            seq = []
            for j in range(self.seq_len+1):
                for k in range(ntokens + 1):
                    if out_list[0][j][k] == 1:
                        seq.append(k)
            sub_samp = []
            n = len(seq)
            for j in range(n):
                tok = seq[j]
                if tok != ntokens:
                    sub_samp.append(tok+1)
                else:
                    break
            gen_list.append(sub_samp)
        with open(self.save_path + result_file + '.txt', 'a') as f:
            f.writelines(' '.join(str(token) for token in list) + '\n' for list in gen_list)
        return gen_list

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

    def get_mse_loss(self, act_loss_criterion, g_output_t_act, g_authentic_act, batch_size):
        act_loss = act_loss_criterion(g_output_t_act.float(), g_authentic_act.float()) / (batch_size)
        return act_loss

    def get_kl_loss(self, act_loss_criterion, g_output_t_act, g_authentic_act, batch_size):
        g_output_t_act = g_output_t_act + 1
        total_out_act = g_output_t_act.sum(0)
        g_output_t_act = g_output_t_act / (total_out_act + self.vocab_num + 1)

        g_authentic_act = g_authentic_act + 1
        total_aut_act = g_authentic_act.sum(0)
        g_authentic_act = g_authentic_act / (total_aut_act + self.vocab_num + 1)

        act_loss = act_loss_criterion(g_output_t_act.log(), g_authentic_act) / (batch_size)
        return act_loss

    def get_act_distribution(self, g_output, aut_seqs):
        g_output_t_act = g_output.sum(0)
        g_output_t_act = g_output_t_act.sum(0)
        g_authentic_act = aut_seqs.sum(0)
        g_authentic_act = g_authentic_act.sum(0)
        return g_output_t_act, g_authentic_act

    def generator(self, data, g_model, d_model, batch):
        mask_len = data.size()[0]
        src_mask = g_model.generate_square_subsequent_mask(mask_len).to(self.device)
        g_output = g_model(data, src_mask)
        g_output_st = g_output.permute(1, 0, 2)
        # use straight-through Gumbel-softmax to obtain gradient from discriminator
        g_output_t = F.gumbel_softmax(g_output_st, tau=1, hard=True)
        pad_mask_mul, pad_mask_add = self.get_pad_mask(g_output_t, batch, self.seq_len + 1, self.vocab_num + 1, self.pad_ind)
        g_output_t = self.pad_after_end_token(g_output_t, pad_mask_mul, pad_mask_add)
        d_predict = d_model(g_output_t, src_mask)
        return d_predict, g_output_t

    def discrminator(self, dis_data_pos, g_output_t, g_model, d_model, d_criterion, batch):
        mask_len = dis_data_pos.size()[0]
        src_mask = g_model.generate_square_subsequent_mask(mask_len).to(self.device)
        dis_label_pos, dis_label_neg = prepare_dis_label(batch)
        dis_label_neg = dis_label_neg.to(self.device)
        dis_label_pos = dis_label_pos.to(self.device)

        dis_predict_pos = d_model(dis_data_pos, src_mask)
        dis_predict_neg = d_model(g_output_t.detach(), src_mask)

        dis_loss_pos = d_criterion(dis_predict_pos, dis_label_pos.reshape(-1, 1))
        dis_loss_neg = d_criterion(dis_predict_neg, dis_label_neg.reshape(-1, 1))

        predict_neg = (dis_predict_neg.flatten().round())
        gd_acc_neg = (predict_neg == dis_label_neg.flatten()).sum() / batch

        predict_pos = (dis_predict_pos.flatten().round())
        gd_acc_pos = (predict_pos == dis_label_pos.flatten()).sum() / batch

        dis_loss = dis_loss_pos + dis_loss_neg

        return dis_loss, gd_acc_neg, gd_acc_pos

    def get_pre_exp_loss(self, pre_epoch, g_model, d_model, train_dataloader):
        mean_act_loss = 0
        mean_gen_loss = 0

        d_criterion = nn.BCELoss()

        if self.mode == 'MSE':
            act_loss_criterion = nn.MSELoss()
        if self.mode == 'KL':
            act_loss_criterion = nn.KLDivLoss(size_average=False)

        for epoch in range(pre_epoch):
            print('pre epoch' + str(epoch))
            g_model.train()
            d_model.train()

            rand_set = self.generate_random_data(self.train_size, self.vocab_num, self.seq_len)
            rand_set = torch.tensor(rand_set, dtype=torch.int64).to(self.device)

            for i, item in enumerate(train_dataloader):
                dis_data_pos = item
                dis_data_pos = dis_data_pos.to(self.device)
                batch = dis_data_pos.size()[0]
                target_real = torch.ones(batch, 1).to(self.device)
                data = rand_set[i:i + batch]
                data = torch.transpose(data, 0, 1)
                d_predict, pre_g_output_t = self.generator(data, g_model, d_model, batch)
                gen_loss = d_criterion(d_predict, target_real)
                pre_g_output_t_act, g_authentic_act = self.get_act_distribution(pre_g_output_t, dis_data_pos)

                if self.mode == 'MSE':
                    act_loss = self.get_mse_loss(act_loss_criterion, pre_g_output_t_act, g_authentic_act, batch)

                if self.mode == 'KL':
                    act_loss = self.get_kl_loss(act_loss_criterion, pre_g_output_t_act, g_authentic_act, batch)

                mean_act_loss += act_loss.item()
                mean_gen_loss += gen_loss.item()

        mean_act = mean_act_loss / pre_epoch
        mean_gen = mean_gen_loss / pre_epoch

        return mean_act, mean_gen

    def train(self, target, aut_data):
        random.seed(self.seed)
        np.random.seed(self.seed)

        # initialize generator
        g_model = TransformerModel(self.n_inp, self.emb_size, self.n_head_g, self.n_hid, self.n_layer, self.pad_ind, self.drop_out).to(self.device)
        gd_optimizer = torch.optim.Adam(g_model.parameters(), lr=self.lr_gen, betas=(0.5, 0.999)) # the optimizer of generator

        # initialize discriminator
        emsize_d = self.vocab_num + 1
        dis_output_dim = 1
        d_model = TransformerModel_DIS(dis_output_dim, emsize_d, self.n_head_d, self.n_hid, self.n_layer, self.drop_out).to(self.device)
        d_criterion = nn.BCELoss()
        d_optimizer = torch.optim.Adam(d_model.parameters(), lr=self.lr_dis, betas=(0.5, 0.999))

        # record the parameters
        para_list = [self.vocab_num, self.emb_size, self.n_head_g, self.n_hid, self.n_layer, self.pad_ind, self.drop_out, self.n_head_d,  self.batch_size, self.epochs, self.lr_dis]
        self.write_log(para_list, 'parameter_log.txt')

        # load the one-hot format of training data
        dataset = load_nar_data(target)
        train_data, _, _ = torch.utils.data.random_split(dataset, (self.train_size, self.valid_size, self.test_size), generator=torch.Generator().manual_seed(self.seed))
        train_dataloader = DataLoader(train_data, batch_size=self.batch_size, drop_last=False, shuffle=False, num_workers=1)

        # load the original format of test data
        dataset_2 = load_nar_data(aut_data)
        _, _, test_data = torch.utils.data.random_split(dataset_2, (self.train_size, self.valid_size, self.test_size), generator=torch.Generator().manual_seed(self.seed))
        test_dataloader = DataLoader(test_data, batch_size=self.batch_size, drop_last=False, shuffle=False, num_workers=1)
        test_seqs = next(iter(test_dataloader)).tolist()
        test_list = self.reverse_to_list(test_seqs, self.vocab_num)

        # two process gan variants
        if self.mode == 'MSE':
            act_loss_criterion = nn.MSELoss()
        if self.mode == 'KL':
            act_loss_criterion = nn.KLDivLoss(size_average=False)

        # run pre epochs if add activity loss
        pre_epoch = 50
        if self.mode != 'Vanilla':
            mean_act_loss, mean_gen_loss = self.get_pre_exp_loss(pre_epoch, g_model, d_model, train_dataloader)

        d_acc = []
        act_loss = 0

        for big_epoch in range(1,  self.epochs + 1):
            start_time = time.time()
            g_model.train()
            d_model.train()

            dis_total_loss = 0
            gen_total_loss = 0

            rand_set = self.generate_random_data(self.train_size, self.vocab_num, self.seq_len)
            rand_set = torch.tensor(rand_set, dtype=torch.int64).to(self.device)

            acc_i = 0

            for i, item in enumerate(train_dataloader):
                # update generator
                dis_data_pos = item
                dis_data_pos = dis_data_pos.to(self.device)
                batch = dis_data_pos.size()[0]

                dis_data_pos = dis_data_pos.permute(1, 0, 2) # [LENGTH, BATCH_SIZE, VOCAB]
                target_real = torch.ones(batch, 1).to(self.device)

                random_data = rand_set[i:i + batch]
                random_data = torch.transpose(random_data, 0, 1)
                gd_optimizer.zero_grad()

                d_predict, g_output_t = self.generator(random_data, g_model, d_model, batch)
                gen_loss = d_criterion(d_predict, target_real)

                g_output_t_act, g_authentic_act = self.get_act_distribution(g_output_t, dis_data_pos)

                if self.mode == 'Vanilla':
                    act_loss = 0

                # supervised activity distribution loss (Mean square error loss)
                if self.mode == 'MSE':
                    act_loss = self.get_mse_loss(act_loss_criterion, g_output_t_act, g_authentic_act, batch)
                    act_loss = act_loss / (mean_act_loss)
                    gen_loss = gen_loss / (mean_gen_loss)

                # supervised activity distribution loss (KL divergence)
                if self.mode == 'KL':
                    act_loss = self.get_kl_loss(act_loss_criterion, g_output_t_act, g_authentic_act, batch)
                    act_loss = act_loss / (mean_act_loss)
                    gen_loss = gen_loss / (mean_gen_loss)

                g_loss = act_loss + gen_loss
                g_loss.backward()
                gd_optimizer.step()
                gen_total_loss += g_loss

                # update discriminator
                if big_epoch % self.gd_ratio == 0:
                    d_optimizer.zero_grad()
                    dis_loss, gd_acc_neg, gd_acc_pos = self.discrminator(dis_data_pos, g_output_t, g_model, d_model, d_criterion, batch)
                    dis_total_loss += dis_loss

                    # backprop
                    dis_loss.backward()
                    d_optimizer.step()
                    acc_i += (gd_acc_neg + gd_acc_pos)/2

            if big_epoch % self.gd_ratio == 0:
                end_time = time.time()
                acc = acc_i/len(train_dataloader)
                d_acc.append(acc.detach().cpu())

                print('ad epoch {:3d} |  g_loss {:5.4f} | d_loss {:5.4f} | d_acc_pos {:5.2f} | d_acc_neg {:5.2f} | d_acc {:5.2f} |time {}'
                        .format(big_epoch, gen_total_loss/len(train_dataloader), dis_total_loss/len(train_dataloader), gd_acc_pos, gd_acc_neg, acc, datetime.timedelta(seconds=(end_time - start_time)*(self.epochs-big_epoch))))

            if big_epoch % 100 == 0:
                # torch.save(g_model, self.save_path + str(big_epoch)+'g_model.pt')
                # torch.save(d_model, self.save_path + str(big_epoch)+'d_model.pt')
                self.draw(d_acc, str(big_epoch)+'d_acc.png', 'd_acc')

                g_model.eval()
                with torch.no_grad():
                    gen_list = self.gen_data_from_rand(self.gen_num, g_model, self.vocab_num, self.device, str(big_epoch) + '_result_trans')
                with open(self.save_path +'stats/' + 'dif_log.txt', 'a') as filehandle:
                    filehandle.write('%s\n' % big_epoch)
                self.result_eval(gen_list, test_list)

    def result_eval(self, gen_list, test_list):
        save_len_difference(gen_list, test_list, self.save_path+'stats/')
        save_act_difference(gen_list, test_list, self.save_path+'stats/')
        save_variance_dif(gen_list, test_list, self.save_path+'stats/')

    def run(self):
        aut_onehot_data = prepare_onehot_aut_data(self.res_path, self.vocab_num, self.seq_len)
        aut_data = prepare_nar_data(self.res_path, self.seq_len, self.vocab_num)
        self.train(aut_onehot_data, aut_data)
