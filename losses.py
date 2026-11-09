import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp

class HFENLoss(nn.Module):
    def __init__(self, device, ndim=3, kernel_size=None, sigma=None, norm_factor=1.0):
        super().__init__()
        self.ndim = ndim
        self.norm_factor = norm_factor
        self.kernel_size = kernel_size or (15 if ndim == 2 else 7)
        sigma = sigma or (2.5 if ndim == 2 else 1.5)
        ks = self.kernel_size

        if ndim == 2:
            xs = torch.arange(ks, dtype=torch.float32) - (ks - 1) / 2
            xx, yy = torch.meshgrid(xs, xs, indexing='ij')
            std2 = sigma ** 2
            g = torch.exp(-(xx**2 + yy**2) / (2 * std2))
            log = -(1 - (xx**2 + yy**2) / (2 * std2)) * g / (np.pi * std2**2)
            log = log - log.mean()
            log = log / (log.norm() + 1e-8)
            self.filter = nn.Conv2d(1, 1, ks, padding=ks // 2, bias=False)
            self.filter.weight.data = log.view(1, 1, ks, ks).to(device)
        else:
            xs = torch.arange(ks, dtype=torch.float32) - (ks - 1) / 2
            xx, yy, zz = torch.meshgrid(xs, xs, xs, indexing='ij')
            std2 = sigma ** 2
            r2 = xx**2 + yy**2 + zz**2
            g = torch.exp(-r2 / (2 * std2))
            log = -(1 - r2 / (2 * std2)) * g / (np.pi * std2**3)
            log = log - log.mean()
            log = log / (log.norm() + 1e-8)
            self.filter = nn.Conv3d(1, 1, ks, padding=0, bias=False)
            self.filter.weight.data = log.view(1, 1, ks, ks, ks).to(device)

        self.filter.weight.requires_grad = False
        self.criterion = nn.L1Loss()

    def forward(self, sr, hr):
        pad = self.kernel_size // 2
        sr_e = self.filter(F.pad(sr, (pad,) * 6, mode='replicate'))
        hr_e = self.filter(F.pad(hr, (pad,) * 6, mode='replicate'))
        return self.criterion(sr_e, hr_e) * self.norm_factor

class SSIM3DLoss(nn.Module):
    def __init__(self, window_size=7, sigma=1.5, channel=1, data_range=1.0, normalize=True):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.data_range = data_range
        self.normalize = normalize
        self.window = nn.Parameter(
            self._build_window(window_size, channel, sigma), requires_grad=False
        )

    @staticmethod
    def _gaussian(window_size, sigma):
        x = torch.arange(window_size, dtype=torch.float32)
        g = torch.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
        return g / g.sum()

    def _build_window(self, ws, channel, sigma):
        g1d = self._gaussian(ws, sigma).unsqueeze(1)
        g2d = g1d.mm(g1d.t())
        g3d = g1d.mm(g2d.reshape(1, -1)).reshape(ws, ws, ws)
        return g3d.float().unsqueeze(0).unsqueeze(0).expand(channel, 1, ws, ws, ws).contiguous()

    def forward(self, img1, img2):
        w = self.window.to(img1.device)
        pad = self.window_size // 2
        img1 = F.pad(img1, (pad,) * 6, mode='replicate')
        img2 = F.pad(img2, (pad,) * 6, mode='replicate')

        mu1 = F.conv3d(img1, w, padding=0, groups=self.channel)
        mu2 = F.conv3d(img2, w, padding=0, groups=self.channel)

        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        s1 = F.conv3d(img1 * img1, w, padding=0, groups=self.channel) - mu1_sq
        s2 = F.conv3d(img2 * img2, w, padding=0, groups=self.channel) - mu2_sq
        s12 = F.conv3d(img1 * img2, w, padding=0, groups=self.channel) - mu1_mu2

        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * s12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))

        loss = 1.0 - ssim_map.mean()
        return loss * 0.1 if self.normalize else loss

def get_loss_weights(epoch, cfg):
    ls = cfg['loss_schedule']
    s1 = ls['stage1_epochs']
    s2 = ls['stage2_epochs']

    if epoch < s1:
        return ls['stage1_l1'], 0.0, 0.0

    if epoch < s1 + s2:
        p = (epoch - s1) / s2
        l1 = ls['stage1_l1'] - (ls['stage1_l1'] - 0.8) * p
        hfen = ls['stage2_hfen_max'] * p
        return l1, hfen, 0.0

    p = (epoch - s1 - s2) / ls['stage3_epochs']
    return 0.8, ls['stage2_hfen_max'], ls['stage3_ssim_max'] * p
