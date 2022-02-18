import datetime
import time
from torch.utils.data import Dataset, DataLoader

from eval.act_dist_eval import save_act_difference
from eval.length_eval import save_len_difference
from eval.variance_eval import save_variance_dif
from utils.data_loader import load_ar_data
import torch
from models.transformer_ar import TransformerModel
import torch.nn.functional as F
from utils.data_prepare import prepare_ar_data


class Transformer_AR:
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


    def start_train(self, input_data, target_data):

        model = TransformerModel(self.n_inp, self.emb_size, self.n_head, self.n_hid, self.n_layer, self.pad_ind, self.drop_out).to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr_gen)

        dataset = load_ar_data(input_data, target_data)

        train_data, val_data, test_data = torch.utils.data.random_split(dataset, (
        self.train_size, self.valid_size, self.test_size), generator=torch.Generator().manual_seed(self.seed))

        train_dataloader = DataLoader(train_data, batch_size=self.batch_size, drop_last=False, shuffle=False,
                                      num_workers=1)
        val_dataloader = DataLoader(val_data, batch_size=self.batch_size, drop_last=False, shuffle=False, num_workers=1)
        test_dataloader = DataLoader(test_data, batch_size=self.batch_size, drop_last=False, shuffle=False,
                                     num_workers=1)

        patience = 20
        p_flag = 0

        test_loss_l = 100

        for epoch in range(self.epochs):
            model.train()  # Turn on the train mode
            total_loss = 0
            start  = time.time()

            for i, item in enumerate(train_dataloader):
                data, targets = item
                data = data.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                '''loss'''
                loss = model.getLoss(data, targets, self.step)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            total_loss = total_loss/len(train_dataloader)
            val_loss = self.evaluate(model, val_dataloader)
            end = time.time()
            print("epoch {:3d} | train loss {:5.4f} | validation loss {:5.4f} | time {}".format(epoch, total_loss, val_loss/len(val_dataloader), datetime.timedelta(seconds=(end-start)*(self.epochs-epoch))))

            test_loss_c = val_loss
            if test_loss_c > test_loss_l:
                p_flag += 1
            if p_flag > patience:
                break
            test_loss_l = test_loss_c

        test_loss = self.evaluate(model, test_dataloader)
        print("test loss {:5.4f}".format(test_loss/len(test_dataloader)))

        gen_list = self.generate_seqs(model)
        test_seqs = next(iter(test_dataloader))[1].tolist()
        test_list = self.remove_end_token(test_seqs, self.vocab_num)

        self.result_eval(gen_list, test_list)
        self.write_gen_seqs(gen_list)

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


    def generate_seqs(self, model):
        model.eval()
        count = 0
        gen_list = []
        while count < self.gen_num:
            start = [[0]]
            sample = []
            i = 0
            while i < self.seq_len:
                next_word = self.pick_next_word(start, model)
                if next_word != self.vocab_num + 1 and next_word != 0:
                    sample.append(next_word)
                    start.append([next_word])
                    i += 1
                else:
                    break
            count += 1
            gen_list.append(sample)
        return gen_list

    def result_eval(self, gen_list, test_list):
        save_len_difference(gen_list, test_list, self.save_path)
        save_act_difference(gen_list, test_list, self.save_path)
        save_variance_dif(gen_list, test_list, self.save_path)

    def write_gen_seqs(self, gen_seqs):
        with open(self.save_path + 'result_transformer_ar' + '.txt', 'a') as f:
            f.writelines(' '.join(str(token) for token in list) + '\n' for list in gen_seqs)


    def evaluate(self, eval_model, eval_dataloader):
        eval_model.eval()  # Turn on the evaluation mode
        total_loss = 0.
        with torch.no_grad():
            for i, item in enumerate(eval_dataloader):
                data, targets = item
                data = data.to(self.device)
                targets = targets.to(self.device)
                loss = eval_model.getLoss(data, targets, self.step)
                total_loss += loss
        return total_loss


    def pick_next_word(self, start, model):
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
        return gen_token

    '''start training'''
    def run(self):
        input_data, target_data = prepare_ar_data(self.res_path, self.seq_len,self.vocab_num, start_token=0)
        self.start_train(input_data, target_data)

