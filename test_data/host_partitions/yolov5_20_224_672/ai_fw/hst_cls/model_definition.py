# Copyright (c) 2022 Analog Inference, Inc.

import torch.nn as nn
import torch


class ReconstructedYoloV5(nn.Module):
    def __init__(self,
                 m_reconstructed: nn.Module,
                 mb_ref: nn.Module,
                 device: torch.device = torch.device('cpu')):
        super().__init__()
        self.device = device
        self.cart = m_reconstructed.float().to(device)
        self.detect = mb_ref.model[-1].to(device)
        self.names = mb_ref.names
        self.stride = mb_ref.stride

    def forward(self, x, augment=False):
        y_list = self.cart(x)
        y_list_d = [yl.to(self.device) for yl in y_list[::-1]]
        output = self.detect(y_list_d, skip_conv=True)
        return output
