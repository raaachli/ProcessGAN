import torch
from torch.utils.data import Dataset


class load_nar_time_data(Dataset):
    """Dataloader for autoregressive models: RNNs, Transformer-ar"""

    def __init__(self, Target, Time, Duration):
        self.Target = Target
        self.Time = Time
        self.Duration = Duration

    def __len__(self):
        return len(self.Target)

    def __getitem__(self, index):
        tar = torch.IntTensor(self.Target[index])
        time = torch.FloatTensor(self.Time[index])
        duration = torch.Tensor(self.Duration[index])

        return tar.long(), time, duration


class load_cls_time_data(Dataset):
    """Dataloader for off-the-shelf classifier"""

    def __init__(self, Seqs_Act, Seqs_Time, Label, Context):
        self.Seqs_Act = Seqs_Act
        self.Seqs_Time = Seqs_Time

        self.Label = Label
        self.Context = Context

    def __len__(self):
        return len(self.Seqs_Time)

    def __getitem__(self, index):
        seqs_time = torch.Tensor(self.Seqs_Time[index])
        seqs_act = torch.Tensor(self.Seqs_Act[index])

        label = torch.IntTensor(self.Label[index])
        context = torch.IntTensor(self.Context[index])
        return seqs_act.long(), seqs_time.float(), label.float(), context.long()
