import os
import time

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils.data_loader import LoadClsTimeData
from torch.utils.data import DataLoader, Dataset
from eval.classifier_model_time import Classifier
from utils.data_loader import load_cls_time_data
from utils.data_prepare import prepare_cls_time_data


class Transformer_Classifier:
    def __init__(self, pos_path_act, pos_path_time, neg_path_act, neg_path_time, save_path, config):
        """
        Transformer-based binary classifier
        """
        self.pos_path_act = pos_path_act
        self.pos_path_time = pos_path_time

        self.neg_path_act = neg_path_act
        self.neg_path_time = neg_path_time

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
        pos_data_act, pos_data_time, pos_label, pos_context = prepare_cls_time_data(self.pos_path_act, self.pos_path_time, 'pos', self.seq_len, self.vocab_num)
        neg_data_act, neg_data_time, neg_label, neg_context = prepare_cls_time_data(self.neg_path_act, self.neg_path_time, 'neg', self.seq_len, self.vocab_num)

        seqs_act = np.concatenate((pos_data_act, neg_data_act), axis=0)
        seqs_time = np.concatenate((pos_data_time, neg_data_time), axis=0)
        label = np.concatenate((pos_label, neg_label), axis=0)
        context = np.concatenate((pos_context, neg_context), axis=0)
        self.train(seqs_act, seqs_time, label, context)

    def train(self, seqs_act, seqs_time, label, context):
        train_num = self.train_pos_num + self.train_neg_num
        test_num = self.test_pos_num + self.test_neg_num
        val_num = self.val_pos_num + self.val_neg_num

        dataset = LoadClsTimeData(seqs_act, seqs_time, label, context)
        train_data, val_data, test_data = torch.utils.data.random_split(dataset, (train_num, val_num, test_num), generator=torch.Generator().manual_seed(self.seed))

        train_dataloader = DataLoader(train_data, batch_size=self.batch_size, drop_last=False, shuffle=False, num_workers=1)
        val_dataloader = DataLoader(val_data, batch_size=self.batch_size, drop_last=False, shuffle=False, num_workers=1)
        test_dataloader = DataLoader(test_data, batch_size=self.batch_size, drop_last=False, shuffle=False, num_workers=1)

        dis_output = 1
        # dim_out, input_size, head_num, hidden_size, nseq, nlayers, dropout
        model = Classifier(dis_output, self.n_inp, self.emb_size, self.n_head, self.n_hid, self.n_layer, self.pad_ind, self.ndsize, self.seq_len+1, self.drop_out).to(self.device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        patience = 20
        p_flag = 0
        val_loss_l = 100
        best_test_loss = 1000
        for big_epoch in range(1, self.epochs + 1):
            start_time = time.time()
            model.train()
            train_loss = 0
            train_acc = 0

            for i, item in enumerate(train_dataloader):
                train_data_act, train_data_time, train_label, train_context = item
                train_data_act = train_data_act.to(self.device)
                train_data_time = train_data_time.to(self.device)

                train_label = train_label.to(self.device)
                train_context = train_context.to(self.device)
                train_data_act = torch.transpose(train_data_act, 0, 1)
                train_data_time = torch.transpose(train_data_time, 0, 1)

                mask_len = train_data_act.size()[0]
                src_mask = model.generate_square_subsequent_mask(mask_len).to(self.device)

                optimizer.zero_grad()
                train_output = model(train_data_act, train_data_time, src_mask, train_context)
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

            if val_f1 >= 0.8:
                break
            val_loss_c = val_loss
            if val_loss_c > val_loss_l:
                p_flag += 1
            if p_flag > patience:
                break
            val_loss_l = val_loss_c

            if val_loss_c < best_test_loss:
                best_test_loss = val_loss_c
                best_model = model

            train_loss = train_loss/len(train_dataloader)

            end_time = time.time()

            print('epoch {:3d} | train_loss {:5.4f} | train_acc {:5.2f} | test_loss {:5.8f} | test_acc {:5.2f} |recall {:5.4f} |precision {:5.4f} | f1_score {:5.4f}| time {:5.4f}'
                    .format(big_epoch, train_loss, train_acc, val_loss, val_acc, val_recall, val_precision, val_f1, end_time - start_time))

        test_loss, test_f1, test_recall, test_precision, test_acc = self.evaluate(model, test_dataloader, criterion)
        print('Test recall {:5.4f} | precision {:5.4f} |f1_score {:5.4f} | acc {:5.4f}'
                .format(test_recall, test_precision, test_f1, test_acc))

        save = self.save_path+'classifier.pth'
        os.makedirs(os.path.dirname(save), exist_ok=True)
        state_dict_d = {"net": best_model.state_dict()}
        torch.save(state_dict_d, save)

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
                eval_data_act, eval_data_time, eval_label, eval_context = item
                eval_data_act = eval_data_act.to(self.device)
                eval_data_time = eval_data_time.to(self.device)

                eval_label = eval_label.to(self.device)
                eval_context = eval_context.to(self.device)
                eval_data_act = torch.transpose(eval_data_act, 0, 1)
                eval_data_time = torch.transpose(eval_data_time, 0, 1)

                mask_len = eval_data_act.size()[0]
                src_mask = model.generate_square_subsequent_mask(mask_len).to(self.device)
                eval_output = model(eval_data_act, eval_data_time, src_mask, eval_context)
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
            total_eval_loss = total_eval_loss/len(dataloader)
            return total_eval_loss, f1_score, recall, precision, eval_acc

    def classify(self, model_path, synthetic_paths_act, synthetic_paths_time, save_log, val_num):
        # model = torch.load(model_path)
        dis_output = 1
        model = Classifier(dis_output, self.n_inp, self.emb_size, self.n_head, self.n_hid, self.n_layer, self.pad_ind, self.ndsize, self.seq_len+1, self.drop_out).to(self.device)
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['net'])
        model.eval()
        val_data_act, val_data_time, val_label, val_act_dist = prepare_cls_time_data(synthetic_paths_act, synthetic_paths_time, 'neg', self.seq_len, self.vocab_num)
        dataset = LoadClsTimeData(val_data_act, val_data_time, val_label, val_act_dist)
        dataloader = DataLoader(dataset, batch_size=len(val_data_act), drop_last=False, shuffle=False, num_workers=1)
        for i, item in enumerate(dataloader):
            seqs_act, seqs_time, label, context = item
            seqs_act = seqs_act.to(self.device)
            seqs_time = seqs_time.to(self.device)

            label = label.to(self.device)
            context = context.to(self.device)
            val_act = torch.transpose(seqs_act, 0, 1)
            val_time = torch.transpose(seqs_time, 0, 1)

            mask_len = val_act.size()[0]
            src_mask = model.generate_square_subsequent_mask(mask_len).to(self.device)
            val_output = model(val_act, val_time, src_mask, context)
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


def get_config(data):
    file_path = 'eval/classifier_configs/' + data + '_classifier.yaml'
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == '__main__':
    data = 'SEP'
    base_data_path = os.path.join('data', 'data_time', data, 'data_seq')
    pos_path_act = os.path.join(base_data_path, f'{data}.txt')
    pos_path_time = os.path.join(base_data_path, f'{data}_time_dif_norm.txt')

    base_neg_path = os.path.join('data', 'data_hq_time')
    neg_path_act = os.path.join(base_neg_path, f'{data}.txt')
    neg_path_time = os.path.join(base_neg_path, f'{data}_time.txt')

    save_path = os.path.join('classifier_result', data)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    config = get_config(data)

    # the negative samples are generated by adding random noise to the synthetic data (run 'eval/add_noise.py' first)
    train_classifier = Transformer_Classifier(pos_path_act, pos_path_time, neg_path_act, neg_path_time, save_path, config)
    train_classifier.run()


