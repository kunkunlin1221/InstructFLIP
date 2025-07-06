import random
from pathlib import Path

import mmcv
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.utils import make_grid

from ..base import BaseModelInterface
from .components import Conv2d_cd, SpatialAttention
from .loss import ContrastDepthLoss


class CDCNpp(BaseModelInterface):

    def __init__(self, theta=0.7, in_channels=3,  **kwargs):
        super().__init__(**kwargs)
        self._build_model(in_channels, theta)
        self.mse_loss = nn.MSELoss()
        self.contrast_depth_loss = ContrastDepthLoss()

    def _build_model(self, in_channels, theta):
        self.conv1 = nn.Sequential(
            Conv2d_cd(in_channels, 80, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(80),
            nn.ReLU(),
        )
        self.Block1 = nn.Sequential(
            Conv2d_cd(80, 160, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            Conv2d_cd(160, int(160*1.6), kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(int(160*1.6)),
            nn.ReLU(),
            Conv2d_cd(int(160*1.6), 160, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.Block2 = nn.Sequential(
            Conv2d_cd(160, int(160*1.2), kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(int(160*1.2)),
            nn.ReLU(),
            Conv2d_cd(int(160*1.2), 160, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            Conv2d_cd(160, int(160*1.4), kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(int(160*1.4)),
            nn.ReLU(),
            Conv2d_cd(int(160*1.4), 160, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.Block3 = nn.Sequential(
            Conv2d_cd(160, 160, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            Conv2d_cd(160, int(160*1.2), kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(int(160*1.2)),
            nn.ReLU(),
            Conv2d_cd(int(160*1.2), 160, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.lastconv1 = nn.Sequential(
            Conv2d_cd(160*3, 160, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            Conv2d_cd(160, 1, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.Sigmoid(),  # nn.ReLU(),
        )
        self.sa1 = SpatialAttention(kernel=7)
        self.sa2 = SpatialAttention(kernel=5)
        self.sa3 = SpatialAttention(kernel=3)
        self.downsample32x32 = nn.Upsample(size=(32, 32), mode='bilinear')

    def forward(self, x):
        x = self.conv1(x)

        x_Block1 = self.Block1(x)
        attention1 = self.sa1(x_Block1)
        x_Block1_SA = attention1 * x_Block1
        x_Block1_32x32 = self.downsample32x32(x_Block1_SA)

        x_Block2 = self.Block2(x_Block1)
        attention2 = self.sa2(x_Block2)
        x_Block2_SA = attention2 * x_Block2
        x_Block2_32x32 = self.downsample32x32(x_Block2_SA)

        x_Block3 = self.Block3(x_Block2)
        attention3 = self.sa3(x_Block3)
        x_Block3_SA = attention3 * x_Block3
        x_Block3_32x32 = self.downsample32x32(x_Block3_SA)

        x_concat = torch.cat((x_Block1_32x32, x_Block2_32x32, x_Block3_32x32), dim=1)

        pred_masks = self.lastconv1(x_concat)
        return pred_masks

    def forward_train(self, batch):
        imgs = batch['img']
        depths = batch['depth']
        irs = batch['ir']
        binary_mask = batch['binary_mask']
        xs = torch.cat((imgs, irs, depths), dim=1)
        pred_masks = self(xs)
        binary_mask = F.interpolate(binary_mask, size=pred_masks.shape[2:], mode='bilinear')
        # loss
        absolute_loss = self.mse_loss(pred_masks, binary_mask)
        contrastive_loss = self.contrast_depth_loss(pred_masks, binary_mask)
        total_loss = absolute_loss + contrastive_loss
        return {
            'loss': total_loss,
            'absolute_loss': absolute_loss,
            'contrastive_loss': contrastive_loss,
        }

    @torch.no_grad()
    def forward_test(self, batch):
        imgs = batch['img']
        depths = batch['depth']
        irs = batch['ir']
        binary_mask = batch['binary_mask']
        xs = torch.cat((imgs, irs, depths), dim=1)
        pred_masks = self(xs)
        binary_mask = F.interpolate(binary_mask, size=pred_masks.shape[2:], mode='bilinear')
        map_score = torch.sum(pred_masks) / torch.sum(binary_mask)
        return {
            'scores': map_score,
            'pred_masks': pred_masks,
        }

    @torch.no_grad()
    def show_detail(self, batch, batch_idx, epoch, mode='train', logger=None, gpu_id=0):
        p = 0.001 if mode == 'train' else 0.01
        if random.random() <= p:
            is_training = self.training
            if is_training:
                self.eval()

            imgs = batch['img']
            depths = batch['depth']
            irs = batch['ir']
            binary_masks = batch['binary_mask']
            xs = torch.cat((imgs, irs, depths), dim=1)
            pred_masks = self(xs)
            pred_masks = F.interpolate(pred_masks, size=binary_masks.shape[2:], mode='bilinear')

            image = (imgs[0] * 128 + 127.5).to(torch.uint8)
            depth = (depths[0] * 128 + 127.5).to(torch.uint8).expand(3, -1, -1)
            ir = (irs[0] * 128 + 127.5).to(torch.uint8).expand(3, -1, -1)
            mask = (binary_masks[0] * 255).to(torch.uint8).expand(3, -1, -1)
            pred_mask = (pred_masks[0] * 255).to(torch.uint8).expand(3, -1, -1)

            plotted = [image, depth, ir, mask, pred_mask]
            plotted = make_grid(plotted, pad_value=255)

            if logger is not None:
                log_dir = Path(logger.log_dir, 'images')
                np_plotted = plotted.cpu().numpy().transpose(1, 2, 0)
                mmcv.imwrite(np_plotted, log_dir / f'{mode}/gpu_id={gpu_id}step={epoch}_ind={batch_idx}.jpg')
                logger.add_image(f'{mode}_step={epoch}_ind={batch_idx}', plotted, epoch)

            if is_training:
                self.train()
