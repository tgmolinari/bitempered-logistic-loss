import torch.nn as nn

from bitempered_funcs import tempered_log_loss, tempered_softmax

class BitemperedLogLoss(nn.Module):
    def __init__(self, t1):
        super(BitemperedLogLoss, self).__init__()
        self.t1 = t1

    def forward(self, input, target):
        return tempered_log_loss(input, target, self.t1)


class BitemperedSoftmax(nn.Module):
    def __init__(self, t2):
        super(BitemperedSoftmax, self).__init__()
        self.t2 = t2

    def forward(self, input):
        return tempered_softmax(input, self.t2)