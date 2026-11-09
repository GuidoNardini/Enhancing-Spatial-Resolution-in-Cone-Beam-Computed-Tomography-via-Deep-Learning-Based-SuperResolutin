import os
import glob
import random
import argparse
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm
MODEL_DIR = "/percorso_modello" 
from pytorch_msssim import ssim
from rcan import build_rcan
from dataset import (CTDegradationPipeline, SRTrainDataset2D, SREvalDataset2D,
                  SRTrainDataset3D, SREvalDataset3D)
from losses import HFENLoss, SSIM3DLoss, get_loss_weights
from metrics import psnr, calculate_baseline, validate_sliding_window

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

#Validation
@torch.no_grad()
def validate(model, loader, device, cfg, epoch=None):
    model.eval()
    total_psnr = total_ssim = count = 0

    for lr, hr, *_ in loader:
        lr, hr = lr.to(device), hr.to(device)
        sr = torch.clamp(model(lr), 0, 1)
        total_psnr += psnr(sr, hr).item()
        total_ssim += ssim(sr, hr, data_range=1.0, size_average=True).item()
        count += 1

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count

    # Sliding window validation every 10 epochs
    if epoch and epoch % 10 == 0:
        print(f"\nSliding window validation - epoch {epoch}")
        sw = validate_sliding_window(model, cfg['paths']['valid_data'], device, cfg, max_volumes=2)
        print(f"  Crop centrale  → PSNR: {avg_psnr:.3f} | SSIM: {avg_ssim:.4f}")
        print(f"  Volume intero  → PSNR: {sw['psnr']:.3f} | SSIM: {sw['ssim']:.4f}")

    return avg_psnr, avg_ssim

#Visual Debug
def save_debug_images(model, loader, device, ndim, scale, save_dir, epoch, n=3):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    interp = 'bicubic' if ndim == 2 else 'trilinear'

    with torch.no_grad():
        for i, (lr, hr, paths) in enumerate(loader):
            if i >= n:
                break
            lr, hr = lr.to(device), hr.to(device)
            name = os.path.splitext(os.path.basename(paths[0]))[0]

            sr_base = F.interpolate(lr, scale_factor=scale, mode=interp, align_corners=False)
            sr = torch.clamp(model(lr), 0, 1)

            def to_img(t):
                a = t[0, 0].cpu().numpy()
                return a[a.shape[0] // 2] if ndim == 3 else a

            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            for ax, img, title in zip(axes,
                [to_img(lr), to_img(sr_base), to_img(sr), to_img(hr)],
                ["LR Input", f"{interp} (PSNR {psnr(sr_base, hr).item():.2f})",
                 f"RCAN (PSNR {psnr(sr, hr).item():.2f})", "Ground Truth"]):
                ax.imshow(img, cmap='gray', vmin=0, vmax=1)
                ax.set_title(title)
                ax.axis('off')

            plt.suptitle(f"Epoch {epoch} — {name}")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{name}_E{epoch}.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)

#Training
def train_one_epoch(model, loader, optimizer, scaler, cfg, device, epoch,
                    l1_fn, hfen_fn, ssim_fn):
    model.train()
    lw_l1, lw_hfen, lw_ssim = get_loss_weights(epoch, cfg)
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"E{epoch} [l1={lw_l1:.2f} hfen={lw_hfen:.2f} ssim={lw_ssim:.2f}]")
    for batch in pbar:
        lr, hr = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=cfg['training']['use_amp']):
            sr = model(lr)
            loss = lw_l1 * l1_fn(sr, hr)
            if lw_hfen > 0:
                loss = loss + lw_hfen * hfen_fn(sr, hr)
            if lw_ssim > 0:
                loss = loss + lw_ssim * ssim_fn(sr, hr)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['gradient_clip'])
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        if len(pbar) > 0:
            pbar.set_postfix({'loss': f"{loss.item():.4f}",
                              'mem': f"{torch.cuda.memory_allocated()/1e9:.1f}GB"})

    return total_loss / len(loader)

#Setup
def detect_ndim(cfg):
    if cfg['paths'].get('mode', 'auto') != 'auto':
        return 3 if cfg['paths']['mode'] == '3d' else 2
    train_dir = os.path.join(cfg['paths']['data_dir'], 'train')
    return 3 if glob.glob(os.path.join(train_dir, '*.npy')) else 2

def build_dataloaders(cfg, ndim):
    t = cfg['training']
    dl = cfg['dataloader']
    deg_cfg = cfg['degradation']
    train_deg = CTDegradationPipeline(
        scale=cfg['model']['scale'], mode='train',
        blur_sigma_range=tuple(deg_cfg['train_blur_sigma_range']),
        noise_std_range=tuple(deg_cfg['train_noise_std_range'])
    )

    train_dir = os.path.join(cfg['paths']['data_dir'], 'train')
    valid_dir = os.path.join(cfg['paths']['data_dir'], 'valid')

    if ndim == 2:
        train_ds = SRTrainDataset2D(train_dir, cfg['model']['scale'],
                                    t['hr_patch_2d'], t['train_repeat'], train_deg)
        val_ds = SREvalDataset2D(valid_dir, cfg['model']['scale'])
        bs = t['batch_size_2d']
    else:
        train_ds = SRTrainDataset3D(train_dir, cfg['model']['scale'],
                                    t['hr_patch_3d'], t['train_repeat'],
                                    degradation_pipeline=train_deg)
        val_ds = SREvalDataset3D(valid_dir, cfg['model']['scale'],
                                  crop_size=t['hr_patch_3d'], mode='val')
        bs = t['batch_size_3d']

    nw = dl['num_workers']
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw,
        pin_memory=dl['pin_memory'],
        prefetch_factor=dl['prefetch_factor'] if nw > 0 else None,
        persistent_workers=dl['persistent_workers'] if nw > 0 else False,
        drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, pin_memory=True)
    return train_loader, val_loader

#Main
def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    cfg['paths']['train_data'] = os.path.join(cfg['paths']['data_dir'], 'train')
    cfg['paths']['valid_data'] = os.path.join(cfg['paths']['data_dir'], 'valid')

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_str = cfg['paths'].get('device') or ('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)
    ndim = detect_ndim(cfg)

    exp_dir = os.path.join("experiments", cfg['paths']['exp_name'])
    os.makedirs(exp_dir, exist_ok=True)

    train_loader, val_loader = build_dataloaders(cfg, ndim)

    model = build_rcan(cfg, ndim).to(device)
    print(f"Parametri modello: {sum(p.numel() for p in model.parameters()):,}")

    t = cfg['training']
    optimizer = torch.optim.AdamW(model.parameters(), lr=t['lr'], weight_decay=1e-5, fused=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=t['restart_period'], T_mult=1, eta_min=t['min_lr']
    )
    scaler = GradScaler(enabled=t['use_amp'])

    l1_fn = nn.L1Loss()
    hfen_fn = HFENLoss(device, ndim=ndim, kernel_size=7, sigma=1.0)
    ssim_fn = SSIM3DLoss(window_size=7, data_range=1.0).to(device)

    baseline_psnr, baseline_ssim = calculate_baseline(val_loader, cfg['model']['scale'], ndim, device)

    log_path = os.path.join(exp_dir, "training_log.txt")
    with open(log_path, 'w') as f:
        f.write(f"Exp: {cfg['paths']['exp_name']}  ndim={ndim}  device={device}\n")
        f.write(f"Baseline PSNR={baseline_psnr:.2f} dB  SSIM={baseline_ssim:.4f}\n")
        f.write("=" * 60 + "\n")

    best_psnr = 0.0
    patience_counter = 0

    for epoch in range(1, t['epochs'] + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, scaler, cfg,
                                   device, epoch, l1_fn, hfen_fn, ssim_fn)
        curr_psnr, curr_ssim = validate(model, val_loader, device, cfg, epoch=epoch)
        scheduler.step()

        stage = ("Stage1" if epoch <= cfg['loss_schedule']['stage1_epochs']
                 else "Stage2" if epoch <= cfg['loss_schedule']['stage1_epochs'] + cfg['loss_schedule']['stage2_epochs']
                 else "Stage3")

        print(f"[{stage}] Epoch {epoch}/{t['epochs']}  loss={avg_loss:.5f}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}  "
              f"PSNR={curr_psnr:.2f} (+{curr_psnr - baseline_psnr:.2f})  "
              f"SSIM={curr_ssim:.4f}")

        with open(log_path, 'a') as f:
            f.write(f"{epoch},{avg_loss:.5f},{curr_psnr:.4f},{curr_ssim:.4f}\n")

        if curr_psnr > best_psnr:
            best_psnr = curr_psnr
            patience_counter = 0
            torch.save({
                'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'psnr': curr_psnr, 'ssim': curr_ssim, 'config': cfg
            }, os.path.join(exp_dir, "best_model.pth"))
            print(f"  → nuovo best salvato (PSNR={curr_psnr:.2f})")
        else:
            patience_counter += 1
            if patience_counter >= t['patience']:
                print(f"Early stopping dopo {t['patience']} epoche senza miglioramenti.")
                break

        if epoch % t['debug_freq'] == 0:
            save_debug_images(model, val_loader, device, ndim, cfg['model']['scale'],
                              os.path.join(exp_dir, "debug"), epoch)

        if epoch % 5 == 0:
            torch.cuda.empty_cache()

    print(f"\nTraining completato.  Best PSNR={best_psnr:.2f} dB  "
          f"(+{best_psnr - baseline_psnr:.2f} vs baseline)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args([]) # Pass an empty list to parse_args
    main(os.path.join(MODEL_DIR, args.config))
