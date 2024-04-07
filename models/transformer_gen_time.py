# BSD 3-Clause License
#
# Copyright (c) 2017, Pytorch contributors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ===============================================================================

import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.PositionalEncoding import PositionalEncoding


class TransformerModel_GEN_Time(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, padding_index, dropout=0.5):
        super(TransformerModel_GEN_Time, self).__init__()
        self.ninp = ninp+4
        self.pos_encoder = PositionalEncoding(self.ninp, dropout)
        encoder_layers = TransformerEncoderLayer(self.ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, self.ninp-4, padding_idx=padding_index)
        self.decoder_1 = nn.Linear(self.ninp, ntoken)
        self.decoder_2 = nn.Linear(self.ninp, 1)
        self.relu = nn.ReLU()
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder_1.bias.data.zero_()
        self.decoder_1.weight.data.uniform_(-initrange, initrange)

        self.decoder_2.bias.data.zero_()
        self.decoder_2.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask, duration):
        src = self.encoder(src)
        duration = duration.unsqueeze(dim=0)
        duration = duration.repeat(src.size(0), 1, 4)
        src = torch.cat((src, duration), dim=2)
        src = src * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        src = src.to(torch.float32)
        output = self.transformer_encoder(src, src_mask)
        output_act = self.decoder_1(output)
        output_time = self.decoder_2(output)
        # output_time = self.relu(output_time)
        return output_act, output_time
