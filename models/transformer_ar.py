import math
import torch
import torch.nn as nn
from models.PositionalEncoding import PositionalEncoding


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, padding_index, dropout=0.5):
        torch.set_printoptions(profile="full")
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp, padding_idx=padding_index)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
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

    def forward(self, src, src_mask):

        src = self.encoder(src)
        src = src * math.sqrt(self.ninp)
        src = self.pos_encoder(src)

        output = self.transformer_encoder(src, src_mask)
        # output = self.transformer_encoder(src)
        output = self.decoder(output)

        return output

    def getLoss(self, input, target, step):

        criterion = nn.CrossEntropyLoss()
        input = torch.transpose(input, 0, 1)
        seq_len, batch_size = input.size()
        target = torch.transpose(target, 0, 1)
        loss = 0

        for i in range(0, seq_len, step):
            input_mask = self.generate_square_subsequent_mask(i+1)
            input_mask = input_mask.cuda()
            data = input[:i+1]
            tar = target[:i+1]
            tar = tar.permute(1,0)
            output = self.forward(data, input_mask)
            output = output.permute(1, 2, 0)
            loss += criterion(output, tar)

        loss = loss/(int(seq_len/step)*batch_size)

        return loss

