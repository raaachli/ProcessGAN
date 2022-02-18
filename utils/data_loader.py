from torch.utils.data import Dataset
import torch


class load_ar_data(Dataset):

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

    def __init__(self, Target):
        self.Target = Target

    def __len__(self):
        return len(self.Target)

    def __getitem__(self, index):
        tar = torch.IntTensor(self.Target[index])
        return tar.long()

