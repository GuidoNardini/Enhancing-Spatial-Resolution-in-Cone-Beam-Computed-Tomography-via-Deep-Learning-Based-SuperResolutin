import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pytorch_msssim import ssim

from dataset import apply_worst_case_degradation

def psnr(sr, hr, data_range=1.0):
    mse = F.mse_loss(sr, hr)
    if mse == 0:
        return torch.tensor(float('inf'))
    return 10.0 * torch.log10((data_range ** 2) / mse)

def get_gaussian_kernel_3d(size, sigma=None):
    if sigma is None:
        sigma = size / 6
    coords = np.arange(size) - (size - 1) / 2
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing='ij')
    kernel = np.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    return (kernel / kernel.max()).astype(np.float32)

@torch.no_grad()
def calculate_baseline(loader, scale, ndim, device):
    mode = 'bicubic' if ndim == 2 else 'trilinear'
    total_psnr = total_ssim = count = 0

    for lr, hr, *_ in tqdm(loader, desc="Baseline"):
        lr, hr = lr.to(device), hr.to(device)
        sr = F.interpolate(lr, scale_factor=scale, mode=mode, align_corners=False)

        # crop se le dimensioni non coincidono dopo interpolazione
        slices = tuple(slice(0, min(a, b)) for a, b in zip(sr.shape[2:], hr.shape[2:]))
        sr = sr[(..., *slices)]
        hr = hr[(..., *slices)]

        total_psnr += psnr(sr, hr).item()
        try:
            min_dim = min(sr.shape[2:])
            win = 7 if min_dim < 64 else 11
            total_ssim += ssim(sr.float(), hr.float(), data_range=1.0,
                               size_average=True, win_size=win).item()
        except Exception:
            pass
        count += 1

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    print(f"Baseline {mode}: PSNR={avg_psnr:.2f} dB | SSIM={avg_ssim:.4f}")
    return avg_psnr, avg_ssim

@torch.no_grad()
def _reconstruct_volume_sliding_window(model_fn, vol_lr, scale, crop_size, overlap):
    stride = crop_size - overlap
    D_lr, H_lr, W_lr = vol_lr.shape
    gaussian_kernel = get_gaussian_kernel_3d(crop_size * scale)

    D_out, H_out, W_out = D_lr * scale, H_lr * scale, W_lr * scale
    output_vol = np.zeros((D_out, H_out, W_out), dtype=np.float32)
    count_map = np.zeros((D_out, H_out, W_out), dtype=np.float32)

    for d in range(0, D_lr, stride):
        for h in range(0, H_lr, stride):
            for w in range(0, W_lr, stride):
                d_end = min(d + crop_size, D_lr)
                h_end = min(h + crop_size, H_lr)
                w_end = min(w + crop_size, W_lr)
                d_s = max(0, d_end - crop_size)
                h_s = max(0, h_end - crop_size)
                w_s = max(0, w_end - crop_size)

                patch = vol_lr[d_s:d_end, h_s:h_end, w_s:w_end]
                pd, ph, pw = patch.shape
                if pd < crop_size or ph < crop_size or pw < crop_size:
                    patch = np.pad(patch,
                                   ((0, crop_size-pd), (0, crop_size-ph), (0, crop_size-pw)),
                                   mode='reflect')

                sr_patch = model_fn(patch).squeeze()

                vd = (d_end - d_s) * scale
                vh = (h_end - h_s) * scale
                vw = (w_end - w_s) * scale
                k = gaussian_kernel[:vd, :vh, :vw]

                output_vol[d_s*scale:d_s*scale+vd,
                           h_s*scale:h_s*scale+vh,
                           w_s*scale:w_s*scale+vw] += sr_patch[:vd, :vh, :vw] * k
                count_map[d_s*scale:d_s*scale+vd,
                          h_s*scale:h_s*scale+vh,
                          w_s*scale:w_s*scale+vw] += k

    return output_vol / np.maximum(count_map, 1e-8)

def validate_sliding_window(model, val_dir, device, cfg, max_volumes=2):
    scale = cfg['model']['scale']
    crop_size, overlap = 80, 16

    vol_paths = sorted(glob.glob(os.path.join(val_dir, "*.npy")))[:max_volumes]
    if not vol_paths:
        print(f"Nessun volume in {val_dir}")
        return {'psnr': 0.0, 'ssim': 0.0}

    total_psnr = total_ssim = 0.0

    for vol_path in vol_paths:
        vol_hr = np.load(vol_path).astype(np.float32)

        # padding multiplo di scale
        D, H, W = vol_hr.shape
        pad_shape = tuple(((s + scale - 1) // scale) * scale for s in (D, H, W))
        vol_hr_pad = np.pad(vol_hr,
                            [(0, p-s) for p, s in zip(pad_shape, (D, H, W))],
                            mode='reflect')

        hr_t = torch.from_numpy(vol_hr_pad).unsqueeze(0).unsqueeze(0).to(device)
        lr_t = apply_worst_case_degradation(hr_t, scale)
        vol_lr = lr_t.cpu().numpy().squeeze()

        def model_fn(patch):
            t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                return torch.clamp(model(t), 0, 1).cpu().numpy()

        sr_vol = _reconstruct_volume_sliding_window(model_fn, vol_lr, scale, crop_size, overlap)
        sr_vol = sr_vol[:D, :H, :W]

        sr_t = torch.from_numpy(sr_vol).unsqueeze(0).unsqueeze(0).to(device)
        hr_t = torch.from_numpy(vol_hr).unsqueeze(0).unsqueeze(0).to(device)

        vol_psnr = psnr(sr_t, hr_t).item()
        vol_ssim = ssim(sr_t, hr_t, data_range=1.0, size_average=True).item()
        print(f"  {os.path.basename(vol_path)}: PSNR={vol_psnr:.3f} dB | SSIM={vol_ssim:.4f}")

        total_psnr += vol_psnr
        total_ssim += vol_ssim

    n = len(vol_paths)
    return {'psnr': total_psnr / n, 'ssim': total_ssim / n}

def calculate_baseline_sliding_window(test_dir, device, cfg, max_volumes=3, seed=42):
    scale = cfg['model']['scale']
    crop_size, overlap = 80, 16

    vol_paths = sorted(glob.glob(os.path.join(test_dir, "*.npy")))
    if max_volumes and len(vol_paths) > max_volumes:
        np.random.seed(seed)
        idx = np.random.choice(len(vol_paths), size=max_volumes, replace=False)
        vol_paths = [vol_paths[i] for i in sorted(idx)]

    total_psnr = total_ssim = 0.0

    for vol_path in vol_paths:
        vol_hr = np.load(vol_path).astype(np.float32)
        D, H, W = vol_hr.shape
        pad_shape = tuple(((s + scale - 1) // scale) * scale for s in (D, H, W))
        vol_hr_pad = np.pad(vol_hr, [(0, p-s) for p, s in zip(pad_shape, (D, H, W))], mode='reflect')

        hr_t = torch.from_numpy(vol_hr_pad).unsqueeze(0).unsqueeze(0).to(device)
        lr_t = apply_worst_case_degradation(hr_t, scale)
        vol_lr = lr_t.cpu().numpy().squeeze()

        def baseline_fn(patch):
            t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                sr = F.interpolate(t, scale_factor=scale, mode='trilinear', align_corners=False)
                return torch.clamp(sr, 0, 1).cpu().numpy()

        sr_vol = _reconstruct_volume_sliding_window(baseline_fn, vol_lr, scale, crop_size, overlap)
        sr_vol = sr_vol[:D, :H, :W]

        sr_t = torch.from_numpy(sr_vol).unsqueeze(0).unsqueeze(0).to(device)
        hr_t = torch.from_numpy(vol_hr).unsqueeze(0).unsqueeze(0).to(device)

        vol_psnr = psnr(sr_t, hr_t).item()
        vol_ssim = ssim(sr_t, hr_t, data_range=1.0, size_average=True).item()
        print(f"  {os.path.basename(vol_path)}: PSNR={vol_psnr:.3f} dB | SSIM={vol_ssim:.4f}")

        total_psnr += vol_psnr
        total_ssim += vol_ssim

    n = len(vol_paths)
    avg_psnr = total_psnr / n if n > 0 else 0.0
    avg_ssim = total_ssim / n if n > 0 else 0.0
    print(f"Baseline trilinear SW: PSNR={avg_psnr:.3f} dB | SSIM={avg_ssim:.4f}")
    return {'psnr': avg_psnr, 'ssim': avg_ssim}
