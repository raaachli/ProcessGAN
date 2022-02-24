from utils.data_prepare import prepare_cls_data
import numpy as np
from torch import nn
import torch
import os
from eval.classifier_model import Classifier
import time
from torch.utils.data import Dataset, DataLoader
from utils.data_loader import load_cls_data


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
        pos_data, pos_label, pos_context = prepare_cls_data(self.pos_path, 'pos', self.seq_len, self.vocab_num)
        neg_data, neg_label, neg_context = prepare_cls_data(self.neg_path, 'neg', self.seq_len, self.vocab_num)

        seqs = np.concatenate((pos_data, neg_data), axis=0)
        label = np.concatenate((pos_label, neg_label), axis=0)
        context = np.concatenate((pos_context, neg_context), axis=0)
        self.train(seqs, label, context)

    def train(self, seqs, label, context):
        train_num = self.train_pos_num + self.train_neg_num
        test_num = self.test_pos_num + self.test_neg_num
        val_num = self.val_pos_num + self.val_neg_num

        dataset = load_cls_data(seqs, label, context)
        train_data, val_data, test_data = torch.utils.data.random_split(dataset, (train_num, val_num, test_num), generator=torch.Generator().manual_seed(self.seed))
        train_dataloader = DataLoader(train_data, batch_size=self.batch_size, drop_last=False, shuffle=False, num_workers=1)
        val_dataloader = DataLoader(val_data, batch_size=self.batch_size, drop_last=False, shuffle=False, num_workers=1)
        test_dataloader = DataLoader(test_data, batch_size=self.batch_size, drop_last=False, shuffle=False, num_workers=1)

        dis_output = 1
        model = Classifier(dis_output, self.n_inp, self.emb_size, self.n_head, self.n_hid, self.n_layer, self.pad_ind, self.ndsize, self.drop_out).to(self.device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        patience = 20
        p_flag = 0
        val_loss_l = 100

        for big_epoch in range(1, self.epochs + 1):
            start_time = time.time()
            model.train()
            train_loss = 0
            train_acc = 0

            for i, item in enumerate(train_dataloader):
                train_data, train_label, train_context = item
                train_data = train_data.to(self.device)
                train_label = train_label.to(self.device)
                train_context = train_context.to(self.device)
                train_data = torch.transpose(train_data, 0, 1)

                mask_len = train_data.size()[0]
                src_mask = model.generate_square_subsequent_mask(mask_len).to(self.device)

                optimizer.zero_grad()
                train_output = model(train_data, src_mask, train_context)
                train_output = torch.squeeze(train_output)
                train_label = torch.squeeze(train_label)

                loss = criterion(train_output, train_label)
                train_loss += loss

                loss.backward()
                optimizer.step()

                train_predict = (train_output.flatten().round())
                train_acc += (train_predict == train_label.flatten()).sum() / len(train_label)

            train_acc = train_acc / len(train_dataloader)
            val_loss, val_f1, val_recall, val_precision, val_acc = self.evaluate(model, val_dataloader, criterion)

            val_loss_c = val_loss
            if val_loss_c > val_loss_l:
                p_flag += 1
            if p_flag > patience:
                break
            val_loss_l = val_loss_c

            end_time = time.time()

            print('epoch {:3d} | train_loss {:5.4f} | train_acc {:5.2f} | test_loss {:5.8f} | test_acc {:5.2f} |recall {:5.4f} |precision {:5.4f} | f1_score {:5.4f}| time {:5.4f}'
                    .format(big_epoch, train_loss/len(train_dataloader), train_acc, val_loss, val_acc, val_recall, val_precision, val_f1, end_time - start_time))

        test_loss, test_f1, test_recall, test_precision, test_acc = self.evaluate(model, test_dataloader, criterion)
        print('Test recall {:5.4f} | precision {:5.4f} |f1_score {:5.4f} | acc {:5.4f}'
                .format(test_recall, test_precision, test_f1, test_acc))

        save = self.save_path+'classifier.pt'
        os.makedirs(os.path.dirname(save), exist_ok=True)
        torch.save(model, save)

    def evaluate(self, model, dataloader, criterion):
        model.eval()
        total_eval_loss = 0
        P_T = 0
        A_T = 0
        P_F = 0
        A_F = 0
        T = 0
        eval_acc = 0
        with torch.no_grad():
            for i, item in enumerate(dataloader):
                eval_data, eval_label, eval_context = item
                eval_data = eval_data.to(self.device)
                eval_label = eval_label.to(self.device)
                eval_context = eval_context.to(self.device)
                eval_data = torch.transpose(eval_data, 0, 1)
                mask_len = eval_data.size()[0]
                src_mask = model.generate_square_subsequent_mask(mask_len).to(self.device)
                eval_output = model(eval_data, src_mask, eval_context)
                eval_loss = criterion(eval_output, eval_label)
                eval_predict = (eval_output.flatten().round())
                eval_tar = eval_label.flatten()
                P_T += (eval_predict == 1).sum()
                A_T += (eval_tar == 1).sum()
                P_F += (eval_predict == 0).sum()
                A_F += (eval_tar == 0).sum()
                T += (eval_predict * eval_tar == 1).sum()
                total_eval_loss += eval_loss
                eval_acc += (eval_predict == eval_tar.flatten()).sum()/len(eval_label)
            eval_acc = eval_acc / len(dataloader)
            recall = T / A_T
            precision = T / P_T
            f1_score = 2 * recall * precision / (recall + precision)
            return total_eval_loss/len(dataloader), f1_score, recall, precision, eval_acc

    def classify(self, model_path, synthetic_path, save_log, val_num):
        model = torch.load(model_path)
        model.eval()
        val_data, val_label, val_act_dist = prepare_cls_data(synthetic_path, 'neg', self.seq_len, self.vocab_num)
        dataset = load_cls_data(val_data, val_label, val_act_dist)
        dataloader = DataLoader(dataset, batch_size=len(val_data), drop_last=False, shuffle=False, num_workers=1)
        for i, item in enumerate(dataloader):
            seqs, label, context = item
            seqs = seqs.to(self.device)
            label = label.to(self.device)
            context = context.to(self.device)
            val = torch.transpose(seqs, 0, 1)
            mask_len = val.size()[0]
            src_mask = model.generate_square_subsequent_mask(mask_len).to(self.device)
            val_output = model(val, src_mask, context)
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
        'epochs': 10,
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

    # the negative samples are generated by adding random noise to the synthetic data (run 'eval/add_noise.py' first)
    neg_path = 'data/data_hq/'+data+'.txt'
    train_classifier = Transformer_Classifier(pos_path, neg_path, save_path, config_sep_classifier)
    train_classifier.run()

    synthetic_paths = ['results/result_good/SEP/PGAN/2000_result_trans.txt']

    for synthetic_path in synthetic_paths:
        save_log = 'results/classifier/' + data + '/log.txt'
        model_path = 'results/classifier/' + data + '/classifier.pt'
        train_classifier.classify(model_path, synthetic_path, save_log, 500)

