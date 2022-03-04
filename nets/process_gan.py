import datetime
import random
import torch
import time
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from models.transformer_dis import TransformerModel_DIS
from models.transformer_gen import TransformerModel
from utils.data_loader import load_nar_data
from utils.data_prepare import prepare_onehot_aut_data, prepare_dis_label, prepare_nar_data
from utils.helper import reverse_torch_to_list, generate_random_data, gen_data_from_rand, \
    write_log, plot_loss, eval_result, get_pad_mask, \
    pad_after_end_token, get_act_distribution


class ProcessGAN:
    def __init__(self, res_path, save_path, config, gen_num, mode):
        """ProcessGAN Model and the variants

        Parameters:
            'seq_len'   : the longest sequence length in data
            'vocab_num' : the size of vocabulary
            'emb_size'  : embedding dimension
            'n_hid'     : the dimension of the feedforward network model in nn.TransformerEncoder
            'n_layer'   : the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
            'n_head_g'  : the number of heads in the multi-head-attention models of generator
            'n_head_d'  : the number of heads in the multi-head-attention models of discriminator
            'drop_out'  : the dropout value
            'gd_ratio'  : k value: the generator updates k times and discriminator updates 1 time
            'lr_gen'    : generator learning rate
            'lr_dis'    : discriminator learning rate
            'epochs'    : total epochs

        """
        self.res_path = res_path
        self.save_path = save_path
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
        self.n_inp = self.vocab_num + 1
        self.pad_ind = self.vocab_num

    def train(self, target, aut_data):
        random.seed(self.seed)
        np.random.seed(self.seed)

        # initialize generator
        g_model = TransformerModel(self.n_inp, self.emb_size, self.n_head_g, self.n_hid, self.n_layer, self.pad_ind, self.drop_out).to(self.device)
        # the optimizer of generator
        gd_optimizer = torch.optim.Adam(g_model.parameters(), lr=self.lr_gen, betas=(0.5, 0.999))

        # initialize discriminator
        emsize_d = self.vocab_num + 1
        dis_output_dim = 1
        d_model = TransformerModel_DIS(dis_output_dim, emsize_d, self.n_head_d, self.n_hid, self.n_layer, self.drop_out).to(self.device)
        d_criterion = nn.BCELoss()
        d_optimizer = torch.optim.Adam(d_model.parameters(), lr=self.lr_dis, betas=(0.5, 0.999))

        # record the parameters
        para_list = [self.vocab_num, self.emb_size, self.n_head_g, self.n_hid, self.n_layer, self.pad_ind, self.drop_out, self.n_head_d,  self.batch_size, self.epochs, self.lr_dis]
        write_log(self.save_path, para_list, 'parameter_log.txt')

        # load the one-hot format of training data
        dataset = load_nar_data(target)
        train_data, _, _ = torch.utils.data.random_split(dataset, (self.train_size, self.valid_size, self.test_size), generator=torch.Generator().manual_seed(self.seed))
        train_dataloader = DataLoader(train_data, batch_size=self.batch_size, drop_last=False, shuffle=False, num_workers=1)

        # load the original format of test data for activity loss calculation
        dataset_2 = load_nar_data(aut_data)
        _, _, test_data = torch.utils.data.random_split(dataset_2, (self.train_size, self.valid_size, self.test_size), generator=torch.Generator().manual_seed(self.seed))
        test_dataloader = DataLoader(test_data, batch_size=self.batch_size, drop_last=False, shuffle=False, num_workers=1)
        test_seqs = next(iter(test_dataloader)).tolist()
        test_list = reverse_torch_to_list(test_seqs, self.vocab_num)

        # run pre epochs if add activity loss
        pre_epoch = 50
        if self.mode != 'Vanilla':
            mean_act_loss, mean_gen_loss = self.get_pre_exp_loss(pre_epoch, g_model, d_model, train_dataloader)

        # log the discriminator's accuracies
        d_acc = []

        for big_epoch in range(1,  self.epochs + 1):
            start_time = time.time()
            g_model.train()
            d_model.train()

            dis_total_loss = 0
            gen_total_loss = 0

            # generate random sequences for generator input
            rand_set = generate_random_data(self.train_size, self.vocab_num, self.seq_len)
            rand_set = torch.tensor(rand_set, dtype=torch.int64).to(self.device)

            acc_i = 0

            for i, item in enumerate(train_dataloader):
                # update generator
                dis_data_pos = item
                dis_data_pos = dis_data_pos.to(self.device)
                batch = dis_data_pos.size()[0]

                # [LENGTH, BATCH_SIZE, VOCAB]
                dis_data_pos = dis_data_pos.permute(1, 0, 2)
                real_labels = torch.ones(batch, 1).to(self.device)
                random_data = rand_set[i:i + batch]
                random_data = torch.transpose(random_data, 0, 1)

                # generate sequences from random_data
                gd_optimizer.zero_grad()
                gen_loss, g_output_t = self.generator(random_data, g_model, d_model, batch, d_criterion, real_labels)
                g_output_t_act, g_authentic_act = get_act_distribution(g_output_t, dis_data_pos)
                act_loss = self.get_act_loss(g_output_t_act, g_authentic_act, batch)

                if self.mode != 'Vanilla':
                    act_loss = act_loss / (mean_act_loss)
                    gen_loss = gen_loss / (mean_gen_loss)

                # back-propagate the generator
                g_loss = act_loss + gen_loss
                g_loss.backward()
                gd_optimizer.step()
                gen_total_loss += g_loss

                # update discriminator
                if big_epoch % self.gd_ratio == 0:
                    d_optimizer.zero_grad()
                    dis_loss, gd_acc_neg, gd_acc_pos = self.discrminator(dis_data_pos, g_output_t, g_model, d_model, d_criterion, batch)
                    dis_total_loss += dis_loss

                    # back-propagate the discriminator
                    dis_loss.backward()
                    d_optimizer.step()
                    acc_i += (gd_acc_neg + gd_acc_pos)/2

            if big_epoch % self.gd_ratio == 0:
                end_time = time.time()
                acc = acc_i/len(train_dataloader)
                d_acc.append(acc.detach().cpu())
                gen_total_loss = gen_total_loss/len(train_dataloader)
                dis_total_loss = dis_total_loss/len(train_dataloader)
                es_remain_time = datetime.timedelta(seconds=(end_time - start_time)*(self.epochs-big_epoch))
                print('ad epoch {:3d} |  g_loss {:5.4f} | d_loss {:5.4f} | d_acc_pos {:5.2f} | d_acc_neg {:5.2f} | d_acc {:5.2f} |time {}'
                        .format(big_epoch, gen_total_loss, dis_total_loss, gd_acc_pos, gd_acc_neg, acc, es_remain_time))

            # generate and evaluate samples every 100 epochs
            if big_epoch % 100 == 0:
                torch.save(g_model, self.save_path + str(big_epoch)+'g_model.pt')
                torch.save(d_model, self.save_path + str(big_epoch)+'d_model.pt')
                plot_loss(self.save_path, d_acc, str(big_epoch)+'d_acc.png', 'd_acc')
                g_model.eval()

                # generate synthetic sequences using the generator and save the sequences
                with torch.no_grad():
                    gen_list = gen_data_from_rand(self.gen_num, g_model, self.vocab_num, self.device, str(big_epoch) + '_result_trans', self.save_path, self.seq_len)
                # evaluate and record the results
                with open(self.save_path +'stats/' + 'dif_log.txt', 'a') as filehandle:
                    filehandle.write('%s\n' % big_epoch)
                eval_result(self.save_path +'stats/', gen_list, test_list)

    def get_act_loss(self, g_output_t_act, g_authentic_act, batch_size):
        """get the additional activity distribution loss between generated sequences and real sequences"""
        if self.mode == "MSE":
            act_loss_criterion = nn.MSELoss()
            act_loss = act_loss_criterion(g_output_t_act.float(), g_authentic_act.float()) / (batch_size)
        elif self.mode == "KL":
            act_loss_criterion = nn.KLDivLoss(size_average=False)
            g_output_t_act = g_output_t_act + 1
            total_out_act = g_output_t_act.sum(0)
            g_output_t_act = g_output_t_act / (total_out_act + self.vocab_num + 1)
            g_authentic_act = g_authentic_act + 1
            total_aut_act = g_authentic_act.sum(0)
            g_authentic_act = g_authentic_act / (total_aut_act + self.vocab_num + 1)
            act_loss = act_loss_criterion(g_output_t_act.log(), g_authentic_act) / (batch_size)
        else:
            act_loss = 0

        # return the activity distribution loss
        return act_loss

    def generator(self, data, g_model, d_model, batch, d_criterion, real_labels):
        """Transformer encoder-based Generator"""
        mask_len = data.size()[0]
        src_mask = g_model.generate_square_subsequent_mask(mask_len).to(self.device)
        g_output = g_model(data, src_mask)
        g_output_st = g_output.permute(1, 0, 2)

        # use straight-through Gumbel-softmax to obtain gradient from discriminator
        g_output_t = F.gumbel_softmax(g_output_st, tau=1, hard=True)

        # the tokens generated after the end token will be padded
        pad_mask_mul, pad_mask_add = get_pad_mask(g_output_t, batch, self.seq_len + 1, self.vocab_num + 1, self.pad_ind, self.device)
        g_output_t = pad_after_end_token(g_output_t, pad_mask_mul, pad_mask_add)

        # generator loss is given by discriminator's prediction
        d_predict = d_model(g_output_t, src_mask)
        gen_loss = d_criterion(d_predict, real_labels)

        # return the generator loss, and the generated sequences
        return gen_loss, g_output_t

    def discrminator(self, dis_data_pos, g_output_t, g_model, d_model, d_criterion, batch):
        """Transformer encoder-based Discriminator"""
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

        # return the discriminator loss, the accuracy of negative samples and positive samples
        return dis_loss, gd_acc_neg, gd_acc_pos

    def get_pre_exp_loss(self, pre_epoch, g_model, d_model, train_dataloader):
        """Calculate the expectation loss values.
        Generator loss and activity distribution loss are expected in the same scale when added together.
        total_loss = weight*act_loss + gen_loss
        where:
        weight -> mean(gen_loss)/mean(act_loss)
        """
        total_act_loss = 0
        total_gen_loss = 0
        d_criterion = nn.BCELoss()
        for epoch in range(pre_epoch):
            print('pre epoch' + str(epoch))
            g_model.train()
            d_model.train()
            rand_set = generate_random_data(self.train_size, self.vocab_num, self.seq_len)
            rand_set = torch.tensor(rand_set, dtype=torch.int64).to(self.device)
            for i, item in enumerate(train_dataloader):
                dis_data_pos = item
                dis_data_pos = dis_data_pos.to(self.device)
                batch = dis_data_pos.size()[0]
                real_labels = torch.ones(batch, 1).to(self.device)
                data = rand_set[i:i + batch]
                data = torch.transpose(data, 0, 1)
                gen_loss, pre_g_output_t = self.generator(data, g_model, d_model, batch, d_criterion, real_labels)
                pre_g_output_t_act, g_authentic_act = get_act_distribution(pre_g_output_t, dis_data_pos)
                act_loss = self.get_act_loss(pre_g_output_t_act, g_authentic_act, batch)
                total_act_loss += act_loss.item()
                total_gen_loss += gen_loss.item()
        mean_act = total_act_loss / pre_epoch
        mean_gen = total_gen_loss / pre_epoch

        # return the mean value of the activity distribution loss, and the generator loss
        return mean_act, mean_gen

    def run(self):
        aut_onehot_data = prepare_onehot_aut_data(self.res_path, self.vocab_num, self.seq_len)
        aut_data = prepare_nar_data(self.res_path, self.seq_len, self.vocab_num)
        self.train(aut_onehot_data, aut_data)
