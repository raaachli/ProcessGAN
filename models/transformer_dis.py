import torch
import torch.nn as nn
from models.PositionalEncoding import PositionalEncoding

class TransformerModel_DIS(nn.Module):

    def __init__(self, dim_out, ninp, nhead, nhid, nlayers, dropout):
        torch.set_printoptions(profile="full")
        super(TransformerModel_DIS, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, dim_out)
        self.init_weights()
        self.sigmoid = nn.Sigmoid()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        # output = self.transformer_encoder(src)
        output = output.permute(1,0,2)
        output = output[:,-1,:]
        # output = self.transformer_encoder(src)
        output = self.decoder(output)
        output = self.sigmoid(output)
        return output

        # return output
