import random
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from models.transformer_gen import TransformerModel
from utils.data_prepare import prepare_nar_data
from utils.data_loader import load_nar_data
from utils.helper import generate_random_data, gen_data_from_rand, plot_loss, reverse_torch_to_list, eval_result


class Transformer_NAR:
    def __init__(self, res_path, save_path, config, gen_num):
        """Non-aoturegressive training of Transformer

        Parameters:
        'vocab_num' : the size of vocabulary
        'emb_size'  : embedding dimension
        'n_hid'     : the dimension of the feedforward network model in nn.TransformerEncoder
        'n_layer'   : the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        'n_head'    : the number of heads in the multi-head-attention models
        'drop_out'  : the dropout value

        """
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
        test_list = reverse_torch_to_list(test_seqs, self.vocab_num)

        loss_log = []

        for epoch in range(1, self.epochs+1):
            g_model.train()
            rand_set = generate_random_data(self.train_size, self.vocab_num, self.seq_len)
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
            validation_loss = validation_loss/len(val_dataloader)
            total_loss = total_loss/len(train_dataloader)
            print('epoch {:3d} | gen_loss {:5.2f} | val_loss {:5.2f}'.format(epoch, total_loss, validation_loss))
            loss_log.append(total_loss.detach().cpu())

            if epoch % 100 == 0:
                torch.save(g_model, self.save_path + str(epoch) + '_g_model.pt')
                g_model.eval()
                with torch.no_grad():
                    gen_list = gen_data_from_rand(self.gen_num, g_model, self.vocab_num, self.device, 'result_transformer_nar_'+str(epoch), self.save_path, self.seq_len)
                    eval_result(self.save_path, gen_list, test_list)

        plot_loss(self.save_path, loss_log, "trans_nar_loss.png", "loss")

    def evaluate(self, model, eval_dataloader, criterion):
        total_loss = 0
        model.eval()
        with torch.no_grad():
            for i, item in enumerate(eval_dataloader):
                target = item
                target = target.to(self.device)
                batch = target.size()[0]
                rand_input = generate_random_data(batch, self.vocab_num, self.seq_len)
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
