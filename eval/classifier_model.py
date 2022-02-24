import torch
import torch.nn as nn
from models.PositionalEncoding import PositionalEncoding
import math


class Classifier(nn.Module):

    def __init__(self, dim_out, ntoken, ninp, nhead, nhid, nlayers, padding_index, ndsize, dropout=0.5):
        torch.set_printoptions(profile="full")
        super(Classifier, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp, padding_idx=padding_index)
        self.ninp = ninp

        self.linear_1 = nn.Linear((padding_index+1)*2, ndsize)
        self.linear_2 = nn.Linear(ndsize, ndsize)
        self.linear_out = nn.Linear(ninp, padding_index+1)
        self.batchnorm = nn.BatchNorm1d(ndsize)
        self.batchnorm_2 = nn.BatchNorm1d(ndsize)

        self.decoder = nn.Linear(ndsize, dim_out)
        self.relu = nn.ReLU()

        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask, act_dist):
        src = self.encoder(src)
        src = src * math.sqrt(self.ninp)
        src = self.pos_encoder(src)

        output = self.transformer_encoder(src, src_mask)
        output = output.permute(1, 0, 2)
        output = output[:, -1, :]
        output = self.linear_out(output)

        # concat the sequence with contexts (activity distribution and length)
        output = torch.cat((output, act_dist), 1)
        output = self.relu(self.linear_1(output))
        output = self.batchnorm(output)
        output = self.relu(self.linear_2(output))
        output = self.batchnorm_2(output)

        output = self.decoder(output)
        output = self.sigmoid(output)

        return output
