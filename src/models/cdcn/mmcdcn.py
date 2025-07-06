import random
from pathlib import Path

import mmcv
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.utils import make_grid

from ..base import BaseModelInterface
from .components import Conv2d_cd
from .loss import ContrastDepthLoss


class MMCDCN(BaseModelInterface):

    def __init__(self, theta=0.7, **kwargs):
        super().__init__(**kwargs)
        self._build_model(theta=theta)
        self.mse_loss = nn.MSELoss()
        self.contrast_depth_loss = ContrastDepthLoss()

    def _build_model(self, theta):
        self.conv1_M1 = nn.Sequential(
            Conv2d_cd(3, 64, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.Block1_M1 = nn.Sequential(
            Conv2d_cd(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Conv2d_cd(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            Conv2d_cd(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.Block2_M1 = nn.Sequential(
            Conv2d_cd(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Conv2d_cd(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            Conv2d_cd(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.Block3_M1 = nn.Sequential(
            Conv2d_cd(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Conv2d_cd(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            Conv2d_cd(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.conv1_M2 = nn.Sequential(
            Conv2d_cd(1, 64, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.Block1_M2 = nn.Sequential(
            Conv2d_cd(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Conv2d_cd(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            Conv2d_cd(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.Block2_M2 = nn.Sequential(
            Conv2d_cd(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Conv2d_cd(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            Conv2d_cd(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.Block3_M2 = nn.Sequential(
            Conv2d_cd(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Conv2d_cd(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            Conv2d_cd(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.conv1_M3 = nn.Sequential(
            Conv2d_cd(1, 64, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.Block1_M3 = nn.Sequential(
            Conv2d_cd(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Conv2d_cd(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            Conv2d_cd(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

        )

        self.Block2_M3 = nn.Sequential(
            Conv2d_cd(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Conv2d_cd(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            Conv2d_cd(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.Block3_M3 = nn.Sequential(
            Conv2d_cd(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            Conv2d_cd(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            Conv2d_cd(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.lastconv1_M1 = nn.Sequential(
            Conv2d_cd(128*3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.lastconv1_M2 = nn.Sequential(
            Conv2d_cd(128*3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.lastconv1_M3 = nn.Sequential(
            Conv2d_cd(128*3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.lastconv2 = nn.Sequential(
            Conv2d_cd(128*3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.lastconv3 = nn.Sequential(
            Conv2d_cd(128, 1, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.Sigmoid(),  # nn.ReLU(),
        )

        self.downsample32x32 = nn.Upsample(size=(32, 32), mode='bilinear')

    def forward(self, *args, **kargs):

        # RGB
        x = self.conv1_M1(x1)

        x_Block1_M1 = self.Block1_M1(x)
        x_Block1_32x32_M1 = self.downsample32x32(x_Block1_M1)

        x_Block2_M1 = self.Block2_M1(x_Block1_M1)
        x_Block2_32x32_M1 = self.downsample32x32(x_Block2_M1)

        x_Block3_M1 = self.Block3_M1(x_Block2_M1)
        x_Block3_32x32_M1 = self.downsample32x32(x_Block3_M1)

        x_concat_M1 = torch.cat((x_Block1_32x32_M1, x_Block2_32x32_M1, x_Block3_32x32_M1), dim=1)

        # IR
        x = self.conv1_M2(x2)

        x_Block1_M2 = self.Block1_M2(x)
        x_Block1_32x32_M2 = self.downsample32x32(x_Block1_M2)

        x_Block2_M2 = self.Block2_M2(x_Block1_M2)
        x_Block2_32x32_M2 = self.downsample32x32(x_Block2_M2)

        x_Block3_M2 = self.Block3_M2(x_Block2_M2)
        x_Block3_32x32_M2 = self.downsample32x32(x_Block3_M2)

        x_concat_M2 = torch.cat((x_Block1_32x32_M2, x_Block2_32x32_M2, x_Block3_32x32_M2), dim=1)

        # Depth
        x = self.conv1_M3(x3)

        x_Block1_M3 = self.Block1_M3(x)
        x_Block1_32x32_M3 = self.downsample32x32(x_Block1_M3)

        x_Block2_M3 = self.Block2_M3(x_Block1_M3)
        x_Block2_32x32_M3 = self.downsample32x32(x_Block2_M1)

        x_Block3_M3 = self.Block3_M3(x_Block2_M3)
        x_Block3_32x32_M3 = self.downsample32x32(x_Block3_M3)

        x_concat_M3 = torch.cat((x_Block1_32x32_M3, x_Block2_32x32_M3, x_Block3_32x32_M3), dim=1)

        x_M1 = self.lastconv1_M1(x_concat_M1)
        x_M2 = self.lastconv1_M2(x_concat_M2)
        x_M3 = self.lastconv1_M3(x_concat_M3)

        x = torch.cat((x_M1, x_M2, x_M3), dim=1)

        x = self.lastconv2(x)
        pred_masks = self.lastconv3(x)

        return pred_masks

    def forward_train(self, batch):
        imgs = batch['img']
        depths = batch['depth']
        irs = batch['ir']
        binary_mask = batch['binary_mask']
        pred_masks = self(imgs, irs, depths)
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
        pred_masks = self(imgs, irs, depths)
        binary_mask = F.interpolate(binary_mask, size=pred_masks.shape[2:], mode='bilinear')
        map_score = torch.sum(pred_masks) / torch.sum(binary_mask)
        return {
            'scores': map_score,
            'pred_masks': pred_masks,
        }

    @torch.no_grad()
    def show_detail(self, batch, batch_idx, steps, mode='train', logger=None, gpu_id=0):
        p = 0.02 if mode == 'train' else 0.2
        if random.random() <= p:
            is_training = self.training
            if is_training:
                self.eval()

            imgs = batch['img']
            depths = batch['depth']
            irs = batch['ir']
            binary_masks = batch['binary_mask']
            pred_masks = self(imgs, irs, depths)
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
                mmcv.imwrite(np_plotted, log_dir / f'{mode}/gpu_id={gpu_id}step={steps}_ind={batch_idx}.jpg')
                logger.add_image(f'{mode}_step={steps}_ind={batch_idx}', plotted, steps)

            if is_training:
                self.train()
