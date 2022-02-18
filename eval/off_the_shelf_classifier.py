from utils.data_prepare import get_cls_data
import numpy as np
from torch import nn
import torch
import os
from models.classifier import Classifier
import random
import time

class Transformer_Classifier:
    def __init__(self, pos_path, neg_path, save_path, config):
        self.pos_path = pos_path
        self.neg_path = neg_path
        self.save_path = save_path
        self.config = config

        self.seq_len = config['seq_len']
        self.vocab_num = config['vocab_num']
        self.pad_ind = self.vocab_num
        self.n_inp = self.vocab_num + 1
        self.emb_size = config['emb_size']
        self.n_hid = config['n_hid']
        self.n_layer = config['n_layer']
        self.n_head = config['n_head']
        self.seed = config['seed']
        self.drop_out = config['drop_out']
        self.ndsize = config['ndsize']
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.epochs = config['epochs']
        self.device = config['device']

        self.train_pos_num = config['train_pos_num']
        self.train_neg_num = config['train_neg_num']
        self.test_pos_num = config['test_pos_num']
        self.test_neg_num = config['test_neg_num']
        self.val_pos_num = config['val_pos_num']
        self.val_neg_num = config['val_neg_num']

    def run(self):
        self.train()


    def train(self):

        pos_data, pos_label, pos_act_dist = get_cls_data(self.pos_path, 'pos', self.seq_len, self.vocab_num)
        neg_data, neg_label, neg_act_dist = get_cls_data(self.neg_path, 'neg', self.seq_len, self.vocab_num)


        train_num = self.train_pos_num + self.train_neg_num
        test_num  = self.test_pos_num + self.test_neg_num
        val_num = self.val_pos_num + self.val_neg_num

        pos_num = self.train_pos_num + self.test_pos_num + self.val_pos_num
        neg_num = self.train_neg_num + self.test_neg_num + self.val_neg_num

        num_case = pos_num + neg_num

        x_train = np.concatenate((pos_data[:self.train_pos_num], neg_data[:self.train_neg_num]), axis=0)
        y_train = np.concatenate((pos_label[:self.train_pos_num], neg_label[:self.train_neg_num]), axis=0)
        z_train = np.concatenate((pos_act_dist[:self.train_pos_num], neg_act_dist[:self.train_neg_num]), axis=0)

        perm = np.random.permutation(train_num)
        x_train = x_train[perm]
        y_train = y_train[perm]
        z_train = z_train[perm]

        x_test = np.concatenate((pos_data[self.train_pos_num:self.train_pos_num+self.test_pos_num], neg_data[self.train_neg_num:self.train_neg_num+self.test_neg_num]), axis=0)
        y_test = np.concatenate((pos_label[self.train_pos_num:self.train_pos_num+self.test_pos_num], neg_label[self.train_neg_num:self.train_neg_num+self.test_neg_num]), axis=0)
        z_test = np.concatenate((pos_act_dist[self.train_pos_num:self.train_pos_num+self.test_pos_num], neg_act_dist[self.train_neg_num:self.train_neg_num+self.test_neg_num]), axis=0)

        perm = np.random.permutation(test_num)
        x_test = x_test[perm]
        y_test = y_test[perm]
        z_test = z_test[perm]

        x_val = np.concatenate((pos_data[self.train_pos_num+self.test_pos_num:pos_num], neg_data[self.train_neg_num+self.test_neg_num:neg_num]), axis=0)
        y_val = np.concatenate((pos_label[self.train_pos_num+self.test_pos_num:pos_num], neg_label[self.train_neg_num+self.test_neg_num:neg_num]), axis=0)
        z_val = np.concatenate((pos_act_dist[self.train_pos_num+self.test_pos_num:pos_num],
                                 neg_act_dist[self.train_neg_num+self.test_neg_num:neg_num]), axis=0)

        perm = np.random.permutation(val_num)
        x_val = x_val[perm]
        y_val = y_val[perm]
        z_val = z_val[perm]


        x_train = torch.tensor(x_train, dtype=torch.int64).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        z_train = torch.tensor(z_train, dtype=torch.float32).to(self.device)

        x_test = torch.tensor(x_test, dtype=torch.int64).to(self.device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(self.device)
        z_test = torch.tensor(z_test, dtype=torch.float32).to(self.device)

        x_val = torch.tensor(x_val, dtype=torch.int64).to(self.device)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)
        z_val = torch.tensor(z_val, dtype=torch.float32).to(self.device)


        random.seed(self.seed)
        np.random.seed(self.seed)

        dis_output = 1

        model = Classifier(dis_output, self.n_inp, self.emb_size, self.n_head,
                           self.n_hid, self.n_layer, self.pad_ind, self.ndsize, self.drop_out).to(self.device)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        x_test = torch.transpose(x_test, 0, 1)
        x_val = torch.transpose(x_val, 0, 1)

        patience = 20
        p_flag = 0

        test_loss_l = 100

        for big_epoch in range(1, self.epochs + 1):
            start_time = time.time()
            model.train()
            train_loss = 0
            train_acc = 0

            left_batch = train_num % self.batch_size
            batch_num = int(train_num / self.batch_size) + 1

            for i in range(0, train_num, self.batch_size):

                if i == train_num - left_batch:
                    batch = left_batch
                else:
                    batch = self.batch_size

                train_data = x_train[i:i + batch]
                train_data = torch.transpose(train_data, 0, 1)

                y = y_train[i:i + batch]
                z = z_train[i:i + batch]

                mask_len = train_data.size()[0]
                src_mask = model.generate_square_subsequent_mask(mask_len).to(self.device)

                optimizer.zero_grad()
                train_output = model(train_data, src_mask, z)

                train_output = torch.squeeze(train_output)
                y = torch.squeeze(y)

                loss = criterion(train_output, y)
                loss.backward()
                optimizer.step()

                train_predict = (train_output.flatten().round())
                train_acc = (train_predict == y.flatten()).sum() / batch

                train_loss += loss


            model.eval()
            with torch.no_grad():
                mask_len = x_test.size()[0]
                src_mask = model.generate_square_subsequent_mask(mask_len).to(self.device)
                test_output = model(x_test, src_mask, z_test)
                test_loss = criterion(test_output, y_test)
                test_predict = (test_output.flatten().round())
                test_tar = y_test.flatten()
                P_T = (test_predict == 1).sum()
                A_T = (test_tar == 1).sum()
                P_F = (test_predict == 0).sum()
                A_F = (test_tar == 0).sum()
                T = (test_predict*test_tar == 1).sum()
                recall = T/A_T
                precision = T/P_T
                f1_score = 2*recall*precision/(recall+precision)
                test_acc = (test_predict == y_test.flatten()).sum() / test_num

            test_loss_c = test_loss
            if test_loss_c > test_loss_l:
                p_flag += 1
            if p_flag > patience:
                break
            test_loss_l = test_loss_c

            end_time = time.time()

            print(
                'ad epoch {:3d} |  train_loss {:5.4f} | train_acc {:5.2f} | test_loss {:5.8f} | test_acc {:5.2f} |time {:5.4f}'
                    .format(big_epoch, train_loss/batch_num, train_acc, test_loss, test_acc, end_time - start_time))

            print(
                'recall {:5.4f} | precision {:5.4f} |f1_score {:5.4f}'
                    .format(recall, precision, f1_score))

        model.eval()
        with torch.no_grad():
            mask_len = x_val.size()[0]
            src_mask = model.generate_square_subsequent_mask(mask_len).to(self.device)
            val_output = model(x_val, src_mask, z_val)
            val_loss = criterion(val_output, y_val)
            val_predict = (val_output.flatten().round())
            val_tar = y_val.flatten()
            P_T = (val_predict == 1).sum()
            A_T = (val_tar == 1).sum()
            P_F = (val_predict == 0).sum()
            A_F = (val_tar == 0).sum()
            T = (val_predict * val_tar == 1).sum()
            recall = T / A_T
            precision = T / P_T
            f1_score = 2 * recall * precision / (recall + precision)
            val_acc = (val_predict == y_val.flatten()).sum() / val_num
        print(
            'Validation recall {:5.4f} | precision {:5.4f} |f1_score {:5.4f} | acc {:5.4f}'
                .format(recall, precision, f1_score, val_acc))

        save = self.save_path+'classifier.pt'
        os.makedirs(os.path.dirname(save), exist_ok=True)
        torch.save(model, save)


    def classify(self, model_path, synthetic_path, save_log, val_num):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        val_data, val_label, val_act_dist = get_cls_data(synthetic_path, 'neg', self.seq_len, self.vocab_num)
        x_val = torch.tensor(val_data, dtype=torch.int64).to(self.device)
        y_val = torch.tensor(val_label, dtype=torch.float32).to(self.device)
        z_val = torch.tensor(val_act_dist, dtype=torch.float32).to(self.device)

        model = torch.load(model_path)
        model.eval()
        val = torch.transpose(x_val, 0, 1)
        mask_len = val.size()[0]
        src_mask = model.generate_square_subsequent_mask(mask_len).to(self.device)
        val_output = model(val, src_mask, z_val)
        val_predict = (val_output.flatten().round())
        pred_list = val_predict.tolist()
        wrong_list = []
        wrong_num = 0
        for i in range(val_num):
            if pred_list[i] == 1:
                wrong_list.append(i + 1)
                wrong_num += 1
        if wrong_num/val_num > 0.0:
            print('predict pos ' + str(wrong_num/val_num))

        with open(save_log, 'a') as filehandle:
            filehandle.write('predict pos ' + str(wrong_num/val_num)+'\n')


if __name__ == '__main__':

    config_sep_classifier = {
        'seq_len': 186,
        'vocab_num': 17,  # the size of vocabulary
        'emb_size': 8,  # embedding dimension
        'n_hid': 32,  # the dimension of the feedforward network model in nn.TransformerEncoder
        'n_layer': 3,  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        'n_head': 4,  # the number of heads in the multiheadattention models
        'ndsize': 8,  # dense layer size
        'drop_out': 0.1,  # the dropout value
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'batch_size': 256,
        'lr': 0.001,
        'epochs': 1000,
        'seed': 88,
        'train_pos_num': 676,
        'train_neg_num': 3384,
        'test_pos_num': 85,
        'test_neg_num': 423,
        'val_pos_num': 85,
        'val_neg_num': 423,
    }

    data = 'SEP'
    res_path = 'data/data_all/data_seq/'+data+'.txt'
    save_path = 'results/classifier/'+data+'/'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    pos_path = res_path
    # the negative samples are generated by adding random noise to the synthetic data (using 'eval/add_noise.py')
    neg_path = 'data/data_hq/'+data+'.txt'

    train_classifier = Transformer_Classifier(pos_path, neg_path,
                                              save_path, config_sep_classifier)
    train_classifier.run()

    synthetic_paths = ['results/result_good/SEP/PGAN/2000_result_trans.txt']

    for synthetic_path in synthetic_paths:
        save_log = 'results/classifier/' + data + '/log.txt'
        model_path = 'results/classifier/' + data + '/classifier.pt'
        train_classifier.classify(model_path, synthetic_path, save_log, 500)

