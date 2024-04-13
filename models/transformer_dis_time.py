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

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from models.PositionalEncoding import PositionalEncoding


class TransformerModel_DIS_Time(nn.Module):
    def __init__(self, dim_out, ninp, nhead, nhid, nseq, nlayers, dropout):
        super(TransformerModel_DIS_Time, self).__init__()
        self.ninp = ninp
        self.pos_encoder = PositionalEncoding(nhid, dropout)
        encoder_layers = TransformerEncoderLayer(nhid, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.linear = nn.Linear(ninp+nseq+1, nhid)
        self.decoder = nn.Linear(nhid*nseq, dim_out)
        self.init_weights()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, time, src_mask):
        repeat = 1
        time_interval = self.get_time_interval(time)
        time_interval = time_interval.permute(1, 0, 2)
        time = time.repeat(1, 1, repeat)
        src = torch.cat((src, time, time_interval), dim=2)
        src = self.linear(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = output.permute(1,0,2)
        bs = output.size()[0]
        output_d = output.reshape(bs, -1)
        output_d = self.decoder(output_d)
        output_d = self.sigmoid(output_d)
        return output_d

    def get_time_interval(self, time):
        time = self.relu(time)
        time = time.permute(1, 0, 2)
        time = time.squeeze()
        positions = torch.cumsum(time, dim=1)
        expanded_a = positions.unsqueeze(2)
        expanded_b = positions.unsqueeze(1)
        distance_matrices = torch.abs(expanded_a - expanded_b)
        return distance_matrices
