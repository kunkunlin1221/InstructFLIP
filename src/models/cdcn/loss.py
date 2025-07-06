import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastDepthLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion_MSE = nn.MSELoss()

        kernel_filter_list = [
            [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
            [[0, 1, 0], [0, -1, 0], [0, 0, 0]],
            [[0, 0, 1], [0, -1, 0], [0, 0, 0]],
            [[0, 0, 0], [1, -1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
            [[0, 0, 0], [0, -1, 0], [1, 0, 0]],
            [[0, 0, 0], [0, -1, 0], [0, 1, 0]],
            [[0, 0, 0], [0, -1, 0], [0, 0, 1]],
        ]

        kernel_filter = torch.tensor(kernel_filter_list, dtype=torch.float).unsqueeze(1)
        self.register_buffer('kernel_filter', kernel_filter)

    def contrast_depth_conv(self, input):
        input = input.expand(-1, 8, -1, -1)
        return F.conv2d(input, weight=self.kernel_filter, groups=8)  # depthwise conv

    def forward(self, out, label):
        '''
        compute contrast depth in both of (out, label),
        then get the loss of them
        tf.atrous_convd match tf-versions: 1.4
        '''
        contrast_out = self.contrast_depth_conv(out)
        contrast_label = self.contrast_depth_conv(label)

        loss = self.criterion_MSE(contrast_out, contrast_label)

        return loss
