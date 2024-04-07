import datetime
import time
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils.data_loader import load_nar_time_data
from utils.data_prepare import *
from utils.helper import *
torch.set_printoptions(profile='full')

"""
ProcessGAN Model and the variants
"""


class ProcessGAN_Time:
    def __init__(self, res_path_time, res_path_act, res_path_duration, save_path,
                 config, gen_num, mode, model, model_mode):

        self.res_path_time = res_path_time
        self.res_path_act = res_path_act
        self.res_path_duration = res_path_duration
        self.model_name = model
        self.model_mode = model_mode

        self.save_path = save_path
        self.config = config
        self.gen_num = gen_num
        self.seq_len = config['seq_len']

        self.vocab_num_act = config['vocab_num_act']
        self.emb_size_act = config['emb_size_act']
        self.n_hid_act = config['n_hid_act']
        self.n_layer_act = config['n_layer_act']
        self.n_head_act = config['n_head_act']
        self.drop_out_act = config['drop_out_act']
        self.lr_gen_act = config['lr_gen_act']
        self.n_inp_act = self.vocab_num_act + 1
        self.pad_ind_act = self.vocab_num_act

        self.n_hid_d = config['n_hid_d']
        self.n_layer_d = config['n_layer_d']
        self.n_head_d = config['n_head_d']
        self.drop_out_d = config['drop_out_d']
        self.lr_dis = config['lr_dis']

        self.train_size = config['train_size']
        self.test_size = config['test_size']
        self.valid_size = config['valid_size']
        self.batch_size = config['batch_size']
        self.epochs_pre_train = config['epochs_pre_train']
        self.epochs = config['epochs']

        self.seed = config['seed']
        self.device = config['device']
        self.mode = mode
        self.gd_ratio = config['gd_ratio']
        self.data = config['data']

    def train(self, target, aut_data, aut_data_time, aut_data_duration):
        random.seed(self.seed)
        np.random.seed(self.seed)
        if self.model_name == 'trans':
            from models.transformer_dis_time import TransformerModel_DIS_Time
            from models.transformer_gen_time import TransformerModel_GEN_Time
        if self.model_name == 'trans_attn':
            from models.transformer_dis_time_attn import TransformerModel_DIS_Time
            from models.transformer_gen_time import TransformerModel_GEN_Time

        g_model = TransformerModel_GEN_Time(self.n_inp_act,
                                            self.emb_size_act,
                                            self.n_head_act,
                                            self.n_hid_act,
                                            self.n_layer_act,
                                            self.pad_ind_act,
                                            self.drop_out_act).to(self.device)
        gd_optimizer = torch.optim.Adam(g_model.parameters(), lr=self.lr_gen_act, betas=(0.5, 0.999))
        emsize_d = self.vocab_num_act + 1
        dis_output_dim = 1
        d_model = TransformerModel_DIS_Time(dis_output_dim,
                                            emsize_d,
                                            self.n_head_d,
                                            self.n_hid_d,
                                            self.seq_len,
                                            self.n_layer_d,
                                            self.drop_out_d).to(self.device)
        d_criterion = nn.BCELoss()
        d_optimizer = torch.optim.Adam(d_model.parameters(), lr=self.lr_dis, betas=(0.5, 0.999))
        num_parameters_g = sum(p.numel() for p in g_model.parameters() if p.requires_grad)
        num_parameters_d = sum(p.numel() for p in d_model.parameters() if p.requires_grad)
        print(
            'num_parameters_g {:3d} |  num_parameters_d {:3d} '
            .format(num_parameters_g, num_parameters_d))

        # record the parameters
        para_list = [self.vocab_num_act,
                     self.emb_size_act,
                     self.n_head_act,
                     self.n_hid_act,
                     self.n_layer_act,
                     self.pad_ind_act,
                     self.drop_out_act,

                     self.lr_gen_act,
                     self.n_hid_d,
                     self.n_layer_d,
                     self.drop_out_d,
                     self.lr_dis,
                     self.n_head_d,
                     num_parameters_g,
                     num_parameters_d,
                     self.batch_size,
                     self.epochs,

                     ]
        write_log(self.save_path, para_list, 'parameter_log.txt')

        # load the one-hot format of training data
        dataset = load_nar_time_data(target, aut_data_time, aut_data_duration)
        train_data, _, _ = torch.utils.data.random_split(dataset, (self.train_size, self.valid_size, self.test_size), generator=torch.Generator().manual_seed(self.seed))
        train_dataloader = DataLoader(train_data, batch_size=self.batch_size, drop_last=False, shuffle=False, num_workers=1)

        # load the original format of test data for activity loss calculation
        dataset_2 = load_nar_time_data(aut_data, aut_data_time, aut_data_duration)
        test_data, _, _ = torch.utils.data.random_split(dataset_2, (self.train_size, self.valid_size, self.test_size), generator=torch.Generator().manual_seed(self.seed))
        test_dataloader = DataLoader(test_data, batch_size=self.train_size, drop_last=False, shuffle=False, num_workers=1)
        test_seqs, test_time, test_duration = next(iter(test_dataloader))
        test_seqs = test_seqs.tolist()
        test_time = test_time.tolist()
        test_list, test_list_time = reverse_torch_to_list_time(test_seqs, test_time, self.vocab_num_act)

        # ckpt = torch.load(ckpt_path)
        # g_model.load_state_dict(ckpt['net'])
        # with torch.no_grad():
        #     gen_list, gen_list_time = gen_data_from_rand(self.gen_num, g_model, self.vocab_num_act, self.device,
        #                                   str(0) + '_result_trans', self.save_path, self.seq_len)
        # # evaluate and record the results
        # with open(self.save_path + 'stats/' + 'dif_log.txt', 'a') as filehandle:
        #     filehandle.write('%s\n' % 0)
        # eval_result(self.save_path + 'stats/', gen_list, gen_list_time, test_list, test_list_time, self.vocab_num_act)

        # run pre epochs if add activity loss
        pre_epoch = 50
        if self.mode != 'Vanilla':
            mean_act_loss, mean_gen_loss, mean_time_loss, mean_act_time_loss = self.get_pre_exp_loss(pre_epoch, g_model, d_model, train_dataloader)

        # log the discriminator's accuracies
        d_acc = []

        for big_epoch in range(1,  self.epochs + 1):
            start_time = time.time()
            g_model.train()
            d_model.train()

            dis_total_loss = 0
            gen_total_loss = 0

            # generate random sequences for generator input
            rand_set = generate_random_data(self.train_size, self.vocab_num_act, self.seq_len)
            rand_set = torch.tensor(rand_set, dtype=torch.int64).to(self.device)

            acc_i = 0
            iter_time = []
            for i, item in enumerate(train_dataloader):
                # update generator
                iter_start_time = time.time()
                dis_data_pos, dis_data_time, dis_data_duration = item
                dis_data_pos = dis_data_pos.to(self.device)
                dis_data_time = dis_data_time.to(self.device)
                dis_data_duration = dis_data_duration.to(self.device)

                batch = dis_data_pos.size()[0]

                # [LENGTH, BATCH_SIZE, VOCAB]
                dis_data_pos = dis_data_pos.permute(1, 0, 2)
                real_labels = torch.ones(batch, 1).to(self.device)
                random_data = rand_set[i:i + batch]
                random_data = torch.transpose(random_data, 0, 1)

                # generate sequences from random_data
                gd_optimizer.zero_grad()
                gen_loss, g_output_act, g_output_time = self.generator(random_data, g_model, d_model, batch, d_criterion, real_labels, dis_data_duration, dis_data_time)

                g_output_t_act, g_authentic_act = get_act_distribution(g_output_act, dis_data_pos)
                act_loss = self.get_act_loss(g_output_t_act, g_authentic_act, batch)
                act_time_loss = self.get_act_time_loss(dis_data_pos, dis_data_time, g_output_act, g_output_time, batch)
                time_loss = self.get_time_loss(dis_data_time, g_output_time, batch)

                if self.mode != 'Vanilla':
                    act_loss = act_loss / (mean_act_loss)
                    gen_loss = gen_loss / (mean_gen_loss)
                    time_loss = time_loss / (mean_time_loss)
                    act_time_loss = act_time_loss / (mean_act_time_loss)

                # back-propagate the generator
                if self.model_mode == 1:
                    g_loss = gen_loss
                if self.model_mode == 2:
                    g_loss = act_loss + gen_loss
                if self.model_mode == 3:
                    g_loss = gen_loss + time_loss + act_time_loss
                if self.model_mode == 4:
                    g_loss = act_loss + gen_loss + time_loss + act_time_loss

                g_loss.backward()
                gd_optimizer.step()
                gen_total_loss += g_loss

                # update discriminator
                if big_epoch % self.gd_ratio == 0:
                    d_optimizer.zero_grad()
                    g_output_time = g_output_time.permute(1, 0)
                    g_output_time = g_output_time.unsqueeze(dim=2)
                    dis_loss, gd_acc_neg, gd_acc_pos = self.discrminator(dis_data_pos, dis_data_time, g_output_act, g_output_time, g_model, d_model, d_criterion, batch)
                    dis_total_loss += dis_loss

                    # back-propagate the discriminator
                    dis_loss.backward()
                    d_optimizer.step()
                    acc_i += (gd_acc_neg + gd_acc_pos)/2

                iter_end_time = time.time()
                iter_time.append(iter_end_time-iter_start_time)

            if big_epoch % self.gd_ratio == 0:
                end_time = time.time()
                acc = acc_i/len(train_dataloader)
                d_acc.append(acc.detach().cpu())
                gen_total_loss = gen_total_loss/len(train_dataloader)
                dis_total_loss = dis_total_loss/len(train_dataloader)
                es_remain_time = datetime.timedelta(seconds=(end_time - start_time)*(self.epochs-big_epoch))
                per_epoch_time = sum(iter_time)/len(iter_time)
                print('epoch {:3d} |  '
                      'g_loss {:5.4f} | '
                      'd_loss {:5.4f} | '
                      'd_acc_pos {:5.2f} | '
                      'd_acc_neg {:5.2f} | '
                      'd_acc {:5.2f} | '
                      'per_time {:5.4f}s | '
                      'remain_time {}'
                      .format(big_epoch, gen_total_loss, dis_total_loss, gd_acc_pos, gd_acc_neg, acc, per_epoch_time, es_remain_time))

            # generate and evaluate samples every 100 epochs
            if big_epoch % 100 == 0:
                state_dict_g = {"net": g_model.state_dict(),
                                'optimizer': gd_optimizer.state_dict(),
                                'epoch': big_epoch}
                torch.save(state_dict_g, self.save_path + str(big_epoch) + '_g_model.pth')

                state_dict_d = {"net": d_model.state_dict(),
                                'optimizer': d_optimizer.state_dict(),
                                'epoch': big_epoch}
                torch.save(state_dict_d, self.save_path + str(big_epoch) + '_d_model.pth')

                # torch.save(g_model, self.save_path + str(big_epoch)+'g_model.pt')
                # torch.save(d_model, self.save_path + str(big_epoch)+'d_model.pt')
                plot_loss(self.save_path, d_acc, str(big_epoch)+'d_acc.png', 'd_acc')
                g_model.eval()

                # generate synthetic sequences using the generator and save the sequences
                with torch.no_grad():
                    gen_list, gen_list_time = gen_data_from_rand(self.gen_num, g_model, self.vocab_num_act, self.device, str(big_epoch) + '_result_trans', self.save_path, self.seq_len, aut_data_duration)
                # evaluate and record the results
                with open(self.save_path +'stats/' + 'dif_log.txt', 'a') as filehandle:
                    filehandle.write('%s\n' % big_epoch)
                eval_result(self.save_path +'stats/'+str(big_epoch)+'_', gen_list, gen_list_time, test_list, test_list_time, self.vocab_num_act, self.data)

    def get_act_loss(self, g_output_t_act, g_authentic_act, batch_size):
        """get the additional activity distribution loss between generated sequences and real sequences"""
        if self.mode == "MSE":
            act_loss_criterion = nn.MSELoss()
            act_loss = act_loss_criterion(g_output_t_act.float(), g_authentic_act.float()) / (batch_size)
        elif self.mode == "KL":
            act_loss_criterion = nn.KLDivLoss(size_average=False)
            g_output_t_act = g_output_t_act + 1
            total_out_act = g_output_t_act.sum(0)
            g_output_t_act = g_output_t_act / (total_out_act + self.vocab_num_act + 1)
            g_authentic_act = g_authentic_act + 1
            total_aut_act = g_authentic_act.sum(0)
            g_authentic_act = g_authentic_act / (total_aut_act + self.vocab_num_act + 1)
            act_loss = act_loss_criterion(g_output_t_act.log(), g_authentic_act) / (batch_size)
        else:
            act_loss = 0

        # return the activity distribution loss
        return act_loss

    def get_time_loss(self, dis_data_time, g_output_time, batch_size):
        """get the additional timestamp distribution loss between generated sequences and real sequences"""
        time_loss_criterion = nn.MSELoss()
        g_output_time = g_output_time.float()
        g_output_time = torch.cumsum(g_output_time, dim=1)
        dis_data_time = dis_data_time.float()
        dis_data_time = torch.cumsum(dis_data_time, dim=1)
        time_loss = time_loss_criterion(g_output_time.float(), dis_data_time.float()) / (batch_size)
        return time_loss

    def get_act_time_loss(self, dis_data_pos, dis_data_time, g_output_act, g_output_time, batch_size):
        """get the additional activity - timestamp loss between generated sequences and real sequences"""
        time_loss_criterion = nn.MSELoss()

        g_output_time = g_output_time.float()
        g_output_time = torch.cumsum(g_output_time, dim=1)
        dis_data_time = dis_data_time.float()
        dis_data_time = torch.cumsum(dis_data_time, dim=1)
        g_output_act = g_output_act.permute(1, 0, 2)

        means_syn, stds_syn, q_1_syn, q_2_syn, q_3_syn = self.get_act_time_stat(g_output_act, g_output_time)
        means_aut, stds_aut, q_1_aut, q_2_aut, q_3_aut = self.get_act_time_stat(dis_data_pos, dis_data_time)

        time_loss_1 = time_loss_criterion(means_syn.float(), means_aut.float()) / (batch_size)
        time_loss_2 = time_loss_criterion(q_3_syn.float(), q_3_aut.float()) / (batch_size)

        time_loss = time_loss_2 + time_loss_1
        return time_loss

    def get_act_time_stat(self, A_onehot, B):
        d = A_onehot.shape[2]
        index_tensor = torch.arange(d).float().to(A_onehot.device).unsqueeze(0).unsqueeze(0)
        A = (A_onehot * index_tensor).sum(dim=-1).long()
        # Flatten the first two dimensions
        A_flat = A.view(-1)
        B_flat = B.view(-1)
        q = torch.tensor([0.25, 0.5, 0.9]).to(A_onehot.device)
        means = []
        stds = []
        q_1 = []
        q_2 = []
        q_3 = []

        for label in range(d):
            mask = (A_flat == label)
            if mask.sum() > 0:  # Check if the label exists in the flattened tensor
                timestamp_values = torch.masked_select(B_flat, mask)

                means.append(torch.median(timestamp_values))
                stds.append(torch.var(timestamp_values, unbiased=False))
                quant = torch.quantile(timestamp_values, q, dim=0, keepdim=True)
                q_3.append(quant[2])
                q_2.append(quant[1])
                q_1.append(quant[0])

            else:
                # If a label doesn't exist in the batch, append a placeholder (e.g., -1)
                means.append(-1.0)
                stds.append(-1.0)
                q_3.append(-1.0)
                q_2.append(-1.0)
                q_1.append(-1.0)
        # Convert lists to tensors
        means = torch.tensor(means)
        stds = torch.tensor(stds)
        q_1 = torch.tensor(q_1)
        q_2 = torch.tensor(q_2)
        q_3 = torch.tensor(q_3)

        return means, stds, q_1, q_2, q_3

    def get_act_time_distribution(self, A_onehot, B):
        d = A_onehot.shape[2]
        index_tensor = torch.arange(d).float().to(A_onehot.device).unsqueeze(0).unsqueeze(0)
        A = (A_onehot * index_tensor).sum(dim=-1).long()
        # Flatten the first two dimensions
        A_flat = A.view(-1)
        B_flat = B.view(-1)
        q = torch.tensor([0.25, 0.5, 0.75]).to(A_onehot.device)
        means = []
        stds = []
        q_1 = []
        q_2 = []
        q_3 = []

        for label in range(d):
            mask = (A_flat == label)
            if mask.sum() > 0:  # Check if the label exists in the flattened tensor
                timestamp_values = torch.masked_select(B_flat, mask)
                means.append(torch.median(timestamp_values))
                stds.append(torch.var(timestamp_values, unbiased=False))
                quant = torch.quantile(timestamp_values, q, dim=0, keepdim=True)
                q_3.append(quant[2])
                q_2.append(quant[1])
                q_1.append(quant[0])

            else:
                # If a label doesn't exist in the batch, append a placeholder (e.g., -1)
                means.append(-1.0)
                stds.append(-1.0)
                q_3.append(-1.0)
                q_2.append(-1.0)
                q_1.append(-1.0)
        # Convert lists to tensors
        means = torch.tensor(means)
        stds = torch.tensor(stds)
        q_1 = torch.tensor(q_1)
        q_2 = torch.tensor(q_2)
        q_3 = torch.tensor(q_3)

        return means, stds, q_1, q_2, q_3

    def generator(self, data, g_model, d_model, batch, d_criterion, real_labels, duration, target_time):
        """Transformer encoder-based Generator"""
        mask_len = data.size()[0]
        src_mask = g_model.generate_square_subsequent_mask(mask_len).to(self.device)
        g_output, g_output_time = g_model(data, src_mask, duration)
        g_output_st = g_output.permute(1, 0, 2)

        # use straight-through Gumbel-softmax to obtain gradient from discriminator
        g_output_act = F.gumbel_softmax(g_output_st, tau=1, hard=True)
        # the tokens generated after the end token will be padded
        pad_mask_mul, pad_mask_add = get_pad_mask(g_output_act, batch, self.seq_len, self.vocab_num_act + 1, self.pad_ind_act, self.device)
        g_output_act = pad_after_end_token(g_output_act, pad_mask_mul, pad_mask_add)
        # generator loss is given by discriminator's prediction
        src_mask = d_model.generate_square_subsequent_mask(mask_len).to(self.device)

        d_predict = d_model(g_output_act, g_output_time, src_mask)
        gen_loss = d_criterion(d_predict, real_labels)
        g_output_time = g_output_time.squeeze()
        g_output_time = g_output_time.permute(1, 0)
        # return the generator loss, and the generated sequences
        return gen_loss, g_output_act, g_output_time

    def discrminator(self, dis_data_pos, dis_data_time, g_output_act, g_output_time, g_model, d_model, d_criterion, batch):
        """Transformer encoder-based Discriminator"""
        mask_len = dis_data_pos.size()[0]
        dis_label_pos, dis_label_neg = prepare_dis_label(batch)
        dis_label_neg = dis_label_neg.to(self.device)
        dis_label_pos = dis_label_pos.to(self.device)
        dis_data_time = dis_data_time.permute(1, 0)
        dis_data_time = dis_data_time.unsqueeze(dim=2)
        # print(dis_data_pos.shape)
        src_mask = d_model.generate_square_subsequent_mask(mask_len).to(self.device)
        dis_predict_pos = d_model(dis_data_pos, dis_data_time, src_mask)
        dis_predict_neg = d_model(g_output_act.detach(), g_output_time.detach(), src_mask)
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
        total_time_loss = 0
        total_act_time_loss = 0

        d_criterion = nn.BCELoss()
        for epoch in range(pre_epoch):
            print('pre epoch' + str(epoch))
            g_model.train()
            d_model.train()
            rand_set = generate_random_data(self.train_size, self.vocab_num_act, self.seq_len)
            rand_set = torch.tensor(rand_set, dtype=torch.int64).to(self.device)
            for i, item in enumerate(train_dataloader):
                dis_data_pos, dis_data_time, dis_data_duration = item
                dis_data_pos = dis_data_pos.to(self.device)
                dis_data_time = dis_data_time.to(self.device)
                dis_data_duration = dis_data_duration.to(self.device)
                batch = dis_data_pos.size()[0]
                real_labels = torch.ones(batch, 1).to(self.device)
                data = rand_set[i:i + batch]
                data = torch.transpose(data, 0, 1)
                gen_loss, pre_g_output_t, pre_g_output_time = self.generator(data, g_model, d_model, batch, d_criterion, real_labels, dis_data_duration, dis_data_time)
                pre_g_output_t_act, g_authentic_act = get_act_distribution(pre_g_output_t, dis_data_pos)
                act_loss = self.get_act_loss(pre_g_output_t_act, g_authentic_act, batch)
                time_loss = self.get_time_loss(dis_data_time, pre_g_output_time, batch)
                act_time_loss = self.get_act_time_loss(dis_data_pos, dis_data_time, pre_g_output_t, pre_g_output_time, batch)
                total_act_loss += act_loss.item()
                total_gen_loss += gen_loss.item()
                total_time_loss += time_loss.item()
                total_act_time_loss += act_time_loss.item()

        mean_act = total_act_loss / pre_epoch
        mean_gen = total_gen_loss / pre_epoch
        mean_time = total_time_loss / pre_epoch
        mean_act_time = total_act_time_loss / pre_epoch

        # return the mean value of the activity distribution loss, and the generator loss
        return mean_act, mean_gen, mean_time, mean_act_time

    def run(self):
        aut_onehot_data = prepare_onehot_aut_data(self.res_path_act, self.vocab_num_act, self.seq_len)
        aut_data = prepare_nar_data(self.res_path_act, self.seq_len, self.vocab_num_act)

        aut_data_time = prepare_cont_aut_data(self.res_path_time, self.seq_len)
        aut_data_duration = prepare_cont_aut_data(self.res_path_duration, 1)

        self.train(aut_onehot_data, aut_data, aut_data_time, aut_data_duration)
