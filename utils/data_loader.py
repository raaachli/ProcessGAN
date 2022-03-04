from torch.utils.data import Dataset
import torch


class load_ar_data(Dataset):
    """Dataloader for autoregressive models: RNNs, Transformer-ar"""

    def __init__(self, Input, Target):
        self.Input = Input
        self.Target = Target

    def __len__(self):
        return len(self.Input)

    def __getitem__(self, index):
        inp = torch.Tensor(self.Input[index])
        tar = torch.IntTensor(self.Target[index])
        return inp.long(), tar.long()


class load_nar_data(Dataset):
    """Dataloader for non-autoregressive models: Transformer-nar, ProcessGAN and variants"""

    def __init__(self, Target):
        self.Target = Target

    def __len__(self):
        return len(self.Target)

    def __getitem__(self, index):
        tar = torch.IntTensor(self.Target[index])
        return tar.long()


class load_cls_data(Dataset):
    """Dataloader for off-the-shelf classifier"""

    def __init__(self, Seqs, Label, Context):
        self.Seqs = Seqs
        self.Label = Label
        self.Context = Context

    def __len__(self):
        return len(self.Seqs)

    def __getitem__(self, index):
        seqs = torch.Tensor(self.Seqs[index])
        label = torch.IntTensor(self.Label[index])
        context = torch.IntTensor(self.Context[index])
        return seqs.long(), label.float(), context.long()
