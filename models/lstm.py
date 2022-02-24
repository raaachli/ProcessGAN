import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class LSTM_Generator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, gpu=False):
        super(LSTM_Generator, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size+2
        self.gpu = gpu

        self.embeddings = nn.Embedding(vocab_size+2, embedding_dim, padding_idx=vocab_size+1)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.lstm2out = nn.Linear(hidden_dim, vocab_size+2)

        self.dropout_layer = nn.Dropout(p=0.1)

    def init_hidden(self, batch_size=1):
        h = (autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)),
             autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim)))
        return h

    def forward(self, inp, hidden):

        emb = self.embeddings(inp)
        emb = emb.view(1, -1, self.embedding_dim)
        out, hidden = self.lstm(emb, hidden)
        out = self.lstm2out(out.view(-1, self.hidden_dim))
        out = self.dropout_layer(out)
        out = F.log_softmax(out, dim=1)
        return out, hidden

    def sample(self, num_samples, start_letter=0):

        samples = torch.zeros(num_samples, self.max_seq_len).type(torch.LongTensor)

        h = self.init_hidden(num_samples)
        inp = autograd.Variable(torch.LongTensor([start_letter] * num_samples))

        if self.gpu:
            samples = samples.cuda()
            inp = inp.cuda()

        for i in range(self.max_seq_len):
            out, h = self.forward(inp, h)
            out = torch.multinomial(torch.exp(out), 1)
            samples[:, i] = out.view(-1).data

            inp = out.view(-1)

        return samples

    def NLLLoss(self, inp, target):

        loss_fn = nn.NLLLoss()
        batch_size, seq_len = inp.size()
        inp = inp.permute(1, 0)
        target = target.permute(1, 0)
        h = self.init_hidden(batch_size)
        loss = 0

        for i in range(seq_len):

            out, h = self.forward(inp[i], h)
            loss += loss_fn(out, target[i])

        loss = loss / (batch_size*seq_len)

        return loss
