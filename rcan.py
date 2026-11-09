import torch
import torch.nn as nn
import torch.nn.functional as F

def _conv_nd(ndim):
    return nn.Conv2d if ndim == 2 else nn.Conv3d
def _pool_nd(ndim):
    return nn.AdaptiveAvgPool2d if ndim == 2 else nn.AdaptiveAvgPool3d

class ChannelAttentionNd(nn.Module):
    def __init__(self, c, r, ndim):
        super().__init__()
        Conv, Pool = _conv_nd(ndim), _pool_nd(ndim)

        # Dual pooling
        self.avg_pool = Pool(1)
        self.max_pool = Pool(1)

        self.fc = nn.Sequential(
            Conv(c, c // r, 1, bias=False),
            nn.ReLU(inplace=True),
            Conv(c // r, c, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)

class RCABlockNd(nn.Module):
    def __init__(self, c, r, ndim, res_scale=0.1):
        super().__init__()
        Conv = _conv_nd(ndim)
        self.res_scale = res_scale
        self.body = nn.Sequential(
            Conv(c, c, 3, 1, 1),
            nn.ReLU(inplace=True),
            Conv(c, c, 3, 1, 1),
            ChannelAttentionNd(c, r, ndim)
        )

    def forward(self, x):
        return x + self.body(x) * self.res_scale

class ResidualGroupNd(nn.Module):
    def __init__(self, c, r, n_blk, ndim, res_scale=0.1):
        super().__init__()
        Conv = _conv_nd(ndim)
        self.body = nn.Sequential(
            *[RCABlockNd(c, r, ndim, res_scale=res_scale) for _ in range(n_blk)],
            Conv(c, c, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.body(x)

class PixelShuffle3D(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        B, C, D, H, W = x.shape
        r = self.scale
        out_c = C // (r ** 3)
        x = x.view(B, out_c, r, r, r, D, H, W)
        return x.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(B, out_c, D * r, H * r, W * r)

class RCAN(nn.Module):
    def __init__(self, ndim, scale, n_ch=32, n_gr=3, n_blk=3, res_scale=0.1):
        super().__init__()
        Conv = _conv_nd(ndim)
        r = 16

        self.head = Conv(1, n_ch, 3, 1, 1)

        self.body = nn.Sequential(
            *[ResidualGroupNd(n_ch, r, n_blk, ndim, res_scale=res_scale) for _ in range(n_gr)],
            Conv(n_ch, n_ch, 3, 1, 1)
        )

        if ndim == 2:
            self.tail = nn.Sequential(
                Conv(n_ch, n_ch * (scale ** 2), 3, 1, 1),
                nn.PixelShuffle(scale),
                Conv(n_ch, 1, 3, 1, 1)
            )
        else:
            self.tail = nn.Sequential(
                Conv(n_ch, n_ch * (scale ** 3), 3, 1, 1),
                PixelShuffle3D(scale),
                Conv(n_ch, 1, 3, 1, 1)
            )

    def forward(self, x):
        h = self.head(x)
        h = h + self.body(h)
        return self.tail(h)

def build_rcan(cfg, ndim):
    if ndim == 2:
        return RCAN(
            ndim=ndim,
            scale=cfg['model']['scale'],
            n_ch=cfg['model']['channels_2d'],
            n_gr=cfg['model']['groups_2d'],
            n_blk=cfg['model']['blocks_2d'],
            res_scale=cfg['model']['res_scale']
        )
    else:
        return RCAN(
            ndim=ndim,
            scale=cfg['model']['scale'],
            n_ch=cfg['model']['channels_3d'],
            n_gr=cfg['model']['groups_3d'],
            n_blk=cfg['model']['blocks_3d'],
            res_scale=cfg['model']['res_scale']
        )
