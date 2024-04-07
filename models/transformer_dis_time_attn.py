# The implementation of the time_interval based attention is based on https://github.com/JiachengLi1995/TiSASRec

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.PositionalEncoding import PositionalEncoding


class TransformerModel_DIS_Time(nn.Module):
    def __init__(self, dim_out, input_size, head_num, hidden_size, nseq, nlayers, dropout):
        super(TransformerModel_DIS_Time, self).__init__()
        self.decoder = nn.Linear(hidden_size, dim_out)
        self.init_weights()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.softmax = nn.Softmax()

        self.seq_emb = torch.nn.Linear(input_size, hidden_size)
        # self.K_emb = torch.nn.Linear(input_size, hidden_size)
        # self.V_emb = torch.nn.Linear(input_size, hidden_size)

        self.time_embed_K = torch.nn.Linear(1, hidden_size)
        self.time_embed_V = torch.nn.Linear(1, hidden_size)
        self.time_inter_embed_V = torch.nn.Linear(1, hidden_size)
        self.time_inter_embed_K = torch.nn.Linear(1, hidden_size)

        self.item_emb_dropout = torch.nn.Dropout(p=dropout)
        self.abs_pos_K_emb_dropout = torch.nn.Dropout(p=dropout)
        self.abs_pos_V_emb_dropout = torch.nn.Dropout(p=dropout)
        self.time_matrix_K_dropout = torch.nn.Dropout(p=dropout)
        self.time_matrix_V_dropout = torch.nn.Dropout(p=dropout)

        self.attention_layers = torch.nn.ModuleList()
        self.attention_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(hidden_size, eps=1e-8)

        for _ in range(nlayers):
            new_attn_layernorm = torch.nn.LayerNorm(hidden_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            new_attn_layer = TimeAwareAttention(hidden_size, head_num, dropout)
            self.attention_layers.append(new_attn_layer)
            new_fwd_layernorm = torch.nn.LayerNorm(hidden_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            new_fwd_layer = PointWiseFeedForward(hidden_size, dropout)
            self.forward_layers.append(new_fwd_layer)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def generate_timeline_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, bool(1)).masked_fill(mask == 1, float(0.0))
        mask = mask.bool()
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src_inp, time, src_mask):
        src = src_inp.permute(1, 0, 2)
        time = time.permute(1, 0, 2)

        length = src.shape[1]

        time_interval = self.get_time_interval(time)
        time_interval = time_interval.unsqueeze(dim=3)

        K_time_interval = self.time_inter_embed_K(time_interval)
        V_time_interval = self.time_inter_embed_V(time_interval)
        K_time_interval = self.time_matrix_K_dropout(K_time_interval)
        V_time_interval = self.time_matrix_V_dropout(V_time_interval)

        K_time = self.time_embed_K(time)
        V_time = self.time_embed_V(time)
        K_time = self.abs_pos_K_emb_dropout(K_time)
        V_time = self.abs_pos_V_emb_dropout(V_time)

        src = src.float()
        src = self.seq_emb(src)
        src = self.item_emb_dropout(src)

        tl = src.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool))
        attention_mask = attention_mask.to(src)

        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](src)
            mha_outputs = self.attention_layers[i](Q, src, attention_mask, K_time_interval, V_time_interval, K_time, V_time)
            src = Q + mha_outputs
            src = self.forward_layernorms[i](src)
            src = self.forward_layers[i](src)

        src = self.last_layernorm(src)
        output_d = self.decoder(src)
        output_d = self.sigmoid(output_d)
        output_d = output_d.mean(dim=1)
        return output_d

    def get_time_interval(self, time):
        time = self.relu(time)
        # time = time.permute(1, 0, 2)
        time = time.squeeze()
        positions = torch.cumsum(time, dim=1)
        expanded_a = positions.unsqueeze(2)
        expanded_b = positions.unsqueeze(1)
        distance_matrices = torch.abs(expanded_a - expanded_b)
        return distance_matrices


class TimeAwareAttention(torch.nn.Module):
    def __init__(self, hidden_size, head_num, dropout_rate):
        super(TimeAwareAttention, self).__init__()

        self.Q_w = torch.nn.Linear(hidden_size, hidden_size)
        self.K_w = torch.nn.Linear(hidden_size, hidden_size)
        self.V_w = torch.nn.Linear(hidden_size, hidden_size)

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_size = hidden_size // head_num
        self.dropout_rate = dropout_rate

    def forward(self, queries, keys, attn_mask, K_time_interval, V_time_interval, K_time, V_time):
        Q, K, V = self.Q_w(queries), self.K_w(keys), self.V_w(keys)

        Q_ = torch.cat(torch.split(Q, self.head_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, self.head_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, self.head_size, dim=2), dim=0)

        time_matrix_K_ = torch.cat(torch.split(K_time_interval, self.head_size, dim=3), dim=0)
        time_matrix_V_ = torch.cat(torch.split(V_time_interval, self.head_size, dim=3), dim=0)
        abs_pos_K_ = torch.cat(torch.split(K_time, self.head_size, dim=2), dim=0)
        abs_pos_V_ = torch.cat(torch.split(V_time, self.head_size, dim=2), dim=0)

        attn_weights = Q_.matmul(torch.transpose(K_, 1, 2))
        attn_weights += Q_.matmul(torch.transpose(abs_pos_K_, 1, 2))
        attn_weights += time_matrix_K_.matmul(Q_.unsqueeze(-1)).squeeze(-1)

        # seq length adaptive scaling
        attn_weights = attn_weights / (K_.shape[-1] ** 0.5)

        attn_mask = attn_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)
        attn_mask = attn_mask.bool()
        paddings = torch.ones(attn_weights.shape) * (-2 ** 32 + 1)  # -1e23 # float('-inf')
        paddings = paddings.to(Q)
        attn_weights = torch.where(attn_mask, paddings, attn_weights)  # enforcing causality
        attn_weights = self.softmax(attn_weights)  # code as below invalids pytorch backward rules
        attn_weights = self.dropout(attn_weights)

        outputs = attn_weights.matmul(V_)
        outputs += attn_weights.matmul(abs_pos_V_)
        outputs += attn_weights.unsqueeze(2).matmul(time_matrix_V_).reshape(outputs.shape).squeeze(2)

        # (num_head * N, T, C / num_head) -> (N, T, C)
        outputs = torch.cat(torch.split(outputs, Q.shape[0], dim=0), dim=2)

        return outputs


class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs