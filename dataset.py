import os
import random
import glob
import pickle
from enum import Enum

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class DegradationType(Enum):
    CLEAN = 0
    NOISE = 1
    BLUR = 2
    NOISE_BLUR = 3

class CTDegradationPipeline:
    def __init__(self, scale=2, mode='train', blur_sigma_range=(0.3, 0.7),
                 noise_std_range=(0.005, 0.020)):
        self.scale = scale
        self.mode = mode
        self.blur_sigma_range = blur_sigma_range
        self.noise_std_range = noise_std_range
        self._counter = 0

    def _get_degradation_type(self, batch_idx=None):
        if self.mode == 'train':
            idx = batch_idx if batch_idx is not None else self._counter
            self._counter += 1
            return DegradationType(idx % 4)
        return DegradationType.NOISE_BLUR

    def _apply_blur(self, img, sigma=None):
        device = img.device
        ndim = 3 if img.dim() == 5 else 2
        if sigma is None:
            sigma = random.uniform(*self.blur_sigma_range)

        ks = int(2 * np.ceil(3 * sigma) + 1)
        x = torch.arange(ks, dtype=torch.float32, device=device) - ks // 2
        g = torch.exp(-x ** 2 / (2 * sigma ** 2))
        g = g / g.sum()

        if ndim == 2:
            kernel = (g[:, None] * g[None, :]).view(1, 1, ks, ks)
            return F.conv2d(img, kernel, padding=ks // 2)
        else:
            kernel = (g[:, None, None] * g[None, :, None] * g[None, None, :]).view(1, 1, ks, ks, ks)
            return F.conv3d(img, kernel, padding=ks // 2)

    def _apply_noise(self, img, std=None):
        if std is None:
            std = random.uniform(*self.noise_std_range)
        return torch.clamp(img + torch.randn_like(img) * std, 0, 1)

    def _downsample(self, img):
        if img.dim() == 5:
            return F.avg_pool3d(img, kernel_size=self.scale, stride=self.scale)
        return F.avg_pool2d(img, kernel_size=self.scale, stride=self.scale)

    def __call__(self, hr, batch_idx=None):
        deg = self._get_degradation_type(batch_idx)

        if deg == DegradationType.CLEAN:
            lr = self._downsample(hr)
        elif deg == DegradationType.NOISE:
            lr = self._apply_noise(self._downsample(hr))
        elif deg == DegradationType.BLUR:
            lr = self._downsample(self._apply_blur(hr))
        else:  # NOISE_BLUR
            lr = self._apply_noise(self._downsample(self._apply_blur(hr)))

        return lr, deg


def apply_worst_case_degradation(hr, scale, blur_sigma_range=(0.3, 0.7),
                                 noise_std_range=(0.005, 0.020)):
    pipeline = CTDegradationPipeline(
        scale=scale, mode='val',
        blur_sigma_range=blur_sigma_range,
        noise_std_range=noise_std_range
    )
    lr, _ = pipeline(hr)
    return lr

#2D
class SRTrainDataset2D(Dataset):
    def __init__(self, img_dir, scale=2, hr_patch=128, repeat=1,
                 degradation_pipeline=None):
        self.paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        self.scale = scale
        self.hr_patch = hr_patch
        self.repeat = repeat
        self.to_tensor = T.ToTensor()
        self.degradation = degradation_pipeline

    def __len__(self):
        return len(self.paths) * self.repeat

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx % len(self.paths)]).convert("L")

        for _ in range(10):
            i, j, h, w = T.RandomCrop.get_params(img, (self.hr_patch, self.hr_patch))
            hr = TF.crop(img, i, j, h, w)
            if np.array(hr).mean() > 0.05:
                break

        if random.random() > 0.5: hr = TF.hflip(hr)
        if random.random() > 0.5: hr = TF.vflip(hr)
        if random.random() > 0.5: hr = TF.rotate(hr, 90)

        hr_t = self.to_tensor(hr).unsqueeze(0)
        if self.degradation is not None:
            lr_t, _ = self.degradation(hr_t, batch_idx=idx)
        else:
            lr_t = apply_worst_case_degradation(hr_t, self.scale)

        return lr_t[0], hr_t[0]

class SREvalDataset2D(Dataset):
    def __init__(self, img_dir, scale=2):
        self.paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        self.scale = scale
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        hr_pil = Image.open(path).convert("L")
        w, h = hr_pil.size
        hr_pil = TF.center_crop(hr_pil, (h - h % self.scale, w - w % self.scale))

        hr_t = self.to_tensor(hr_pil).unsqueeze(0)
        lr_t = apply_worst_case_degradation(hr_t, self.scale)
        return lr_t[0], hr_t[0], path

#3D
def _load_volume(path, crop_size):
    vol = np.load(path).astype(np.float32)
    D, H, W = vol.shape
    pad = [(0, max(0, crop_size - s)) for s in (D, H, W)]
    if any(p[1] > 0 for p in pad):
        vol = np.pad(vol, pad, mode='constant')

    base = os.path.basename(path).replace(".npy", "")
    meta_path = os.path.join(os.path.dirname(path), f"{base}_metadata.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
    else:
        meta = {'window_center': 0, 'window_width': 2000,
                'affine': np.eye(4), 'header': None}
    return vol, meta

class SRTrainDataset3D(Dataset):
    def __init__(self, volume_dir, scale=2, hr_patch=64, repeat=1,
                 min_foreground_ratio=0.05, max_retries=50,
                 degradation_pipeline=None):
        self.paths = sorted(glob.glob(os.path.join(volume_dir, "*.npy")))
        self.scale = scale
        self.hr_patch = hr_patch
        self.repeat = repeat
        self.min_fg_ratio = min_foreground_ratio
        self.max_retries = max_retries
        self.degradation = degradation_pipeline or CTDegradationPipeline(scale=scale, mode='train')

        print(f"\nPreprocessing {len(self.paths)} volumi...")
        self.volumes, self.metadata = [], []
        for path in tqdm(self.paths, desc="Caching"):
            vol, meta = _load_volume(path, hr_patch)
            self.volumes.append(vol)
            self.metadata.append(meta)
        print(f"Caricati {len(self.volumes)} volumi\n")

    def __len__(self):
        return len(self.paths) * self.repeat

    def _sample_patch(self, vol):
        D, H, W = vol.shape
        ph = self.hr_patch

        for attempt in range(self.max_retries):
            d0 = random.randint(0, max(0, D - ph))
            h0 = random.randint(0, max(0, H - ph))
            w0 = random.randint(0, max(0, W - ph))
            patch = vol[d0:d0+ph, h0:h0+ph, w0:w0+ph]

            fg_mask = patch > 0.05
            fg_ratio = fg_mask.sum() / patch.size
            contrast = patch[fg_mask].std() if fg_ratio > 0 and fg_mask.sum() > 10 else 0

            if fg_ratio >= self.min_fg_ratio and contrast > 0.02:
                return patch

        # fallback al centro
        d0, h0, w0 = D // 2 - ph // 2, H // 2 - ph // 2, W // 2 - ph // 2
        return vol[d0:d0+ph, h0:h0+ph, w0:w0+ph]

    def __getitem__(self, idx):
        vol = self.volumes[idx % len(self.paths)]
        hr = self._sample_patch(vol)

        # Augmentation
        for axis in range(3):
            if random.random() > 0.5:
                hr = np.flip(hr, axis=axis).copy()

        hr_t = torch.from_numpy(hr).unsqueeze(0).unsqueeze(0)
        lr_t, deg_type = self.degradation(hr_t, batch_idx=idx)
        return lr_t[0], hr_t[0], deg_type.value

class SREvalDataset3D(Dataset):
    def __init__(self, vol_dir, scale=2, crop_size=64, mode='val'):
        self.paths = sorted(glob.glob(os.path.join(vol_dir, "*.npy")))
        self.scale = scale
        self.crop_size = crop_size
        self.degradation = CTDegradationPipeline(scale=scale, mode=mode)

        print(f"\nCaricamento {mode}...")
        self.volumes, self.metadata = [], []
        for path in tqdm(self.paths, desc=mode.capitalize()):
            vol, meta = _load_volume(path, crop_size)
            self.volumes.append(vol)
            self.metadata.append(meta)
        print(f"{mode.capitalize()} pronto\n")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        vol = self.volumes[idx]
        D, H, W = vol.shape
        s = self.crop_size // 2
        dc, hc, wc = D // 2, H // 2, W // 2

        hr_crop = vol[dc-s:dc+s, hc-s:hc+s, wc-s:wc+s]
        hr_t = torch.from_numpy(hr_crop).unsqueeze(0).unsqueeze(0)
        lr_t = apply_worst_case_degradation(hr_t, self.scale)
        return lr_t[0], hr_t[0], self.paths[idx]
