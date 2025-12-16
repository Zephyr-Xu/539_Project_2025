import os
import json
import gc
import math
import random
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import h5py
import cv2
import warnings
warnings.filterwarnings("ignore")

######################################
######## Parameters Setting###########
######################################

# System Configurations
class Config:
    dataset_path = "../../dataset/"
    H_matrix_path = "../../dataset/H_official_raw.npy"
    img_size = 128
    stride = 64

    batch_size = 4
    accumulate_grad_batches = 1
    num_workers = 4
    seed = 42

    in_channels = 3
    out_channels = 31
    base_channels = 64

    num_epochs = 300

    base_lr = 1e-4
    min_lr = 1e-7
    weight_decay = 1e-4
    grad_clip = 1.0
    use_amp = True

    val_interval = 5
    save_interval = 20
    log_interval = 20
    patience = 50

    use_physics_guidance = True

    use_ema = True
    ema_decay = 0.999

    psnr_scale = 20.0  
    clamp_lo = -1.5
    clamp_hi = 1.5
    l2_reg = 1e-4


# System Functions
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def get_parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calc_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    mse = F.mse_loss(torch.clamp(pred, 0, 1), torch.clamp(target, 0, 1))
    return 10.0 * torch.log10(1.0 / (mse + eps))


def calc_sam_deg(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    B, C, H, W = pred.shape
    p = pred.view(B, C, -1)
    t = target.view(B, C, -1)
    p = F.normalize(p, dim=1, p=2)
    t = F.normalize(t, dim=1, p=2)
    cos = (p * t).sum(dim=1).clamp(-1 + 1e-6, 1 - 1e-6)
    ang = torch.acos(cos)
    return ang.mean() * (180.0 / math.pi)


def calc_mrae(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    rel = torch.abs(pred - target) / (target + eps)
    return rel.mean()


######################################
########## Data Loading ##############
######################################

class ARAD1KDataset(Dataset):
    """ARAD_1K Dataset Loader for RGB->HSI task (patch-based)."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        img_size: int = 128,
        augment: bool = True,
        stride: int = 64
    ):
        self.data_root = Path(data_root)
        self.img_size = img_size
        self.augment = augment and (split == "train")
        self.split = split
        self.stride = stride

        self.rgbs = []
        self.hsis = []

        if split == "train":
            rgb_dir = self.data_root / "Train_RGB"
            spec_dir = self.data_root / "Train_Spec"
        else:
            rgb_dir = self.data_root / "Valid_RGB"
            spec_dir = self.data_root / "Valid_Spec"

        if spec_dir.exists():
            mat_files = sorted(spec_dir.glob("*.mat"))
            for i, mat_path in enumerate(mat_files):
                name = mat_path.stem
                rgb_path = rgb_dir / f"{name}.jpg"
                if not rgb_path.exists():
                    continue
                try:
                    bgr = cv2.imread(str(rgb_path))
                    if bgr is None:
                        continue
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    rgb = np.float32(rgb) / 255.0
                    rgb = np.transpose(rgb, [2, 0, 1])  

                    with h5py.File(str(mat_path), "r") as f:
                        hsi = np.float32(np.array(f["cube"]))
                    hsi = np.transpose(hsi, [0, 2, 1])  
                    hsi = np.clip(hsi, 0, 1)

                    self.rgbs.append(rgb)
                    self.hsis.append(hsi)

                    if i % 100 == 0:
                        print(f"Loaded {split} scene {i}: {name}")
                except (OSError, KeyError) as e:
                    print(f"Warning: Error loading {mat_path}: {e}")

        self.img_num = len(self.rgbs)
        if self.img_num > 0:
            h, w = self.rgbs[0].shape[1], self.rgbs[0].shape[2]
            self.patch_per_line = (w - img_size) // stride + 1
            self.patch_per_colum = (h - img_size) // stride + 1
            self.patch_per_img = self.patch_per_line * self.patch_per_colum
            self.length = self.patch_per_img * self.img_num
        else:
            self.patch_per_line = 0
            self.patch_per_colum = 0
            self.patch_per_img = 0
            self.length = 0

        print(f"{split.upper()} Dataset: {self.img_num} images, {self.length} patches")

    def augment_fn(self, img, rot_times, v_flip, h_flip):
        for _ in range(rot_times):
            img = np.rot90(img.copy(), axes=(1, 2))
        for _ in range(v_flip):
            img = img[:, :, ::-1].copy()
        for _ in range(h_flip):
            img = img[:, ::-1, :].copy()
        return img

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        stride = self.stride
        crop_size = self.img_size

        img_idx = idx // self.patch_per_img
        patch_idx = idx % self.patch_per_img
        h_idx = patch_idx // self.patch_per_line
        w_idx = patch_idx % self.patch_per_line

        rgb = self.rgbs[img_idx]
        hsi = self.hsis[img_idx]

        rgb = rgb[:, h_idx * stride:h_idx * stride + crop_size,
                  w_idx * stride:w_idx * stride + crop_size]
        hsi = hsi[:, h_idx * stride:h_idx * stride + crop_size,
                  w_idx * stride:w_idx * stride + crop_size]

        if self.augment:
            rot_times = random.randint(0, 3)
            v_flip = random.randint(0, 1)
            h_flip = random.randint(0, 1)
            rgb = self.augment_fn(rgb, rot_times, v_flip, h_flip)
            hsi = self.augment_fn(hsi, rot_times, v_flip, h_flip)

        return np.ascontiguousarray(rgb), np.ascontiguousarray(hsi)


######################################
########## Training Model ############
######################################

# Group normalization
def gn(c, g=8):
    """GroupNorm with automatic group selection"""
    g = min(g, c)
    while c % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, c)

# Two layers convolution
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False), gn(out_c), nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False), gn(out_c), nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False), gn(c), nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1, bias=False), gn(c)
        )

    def forward(self, x):
        return F.silu(self.block(x) + x)

# Encode process in U-Net CNN
class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=2, padding=1, bias=False), gn(out_c), nn.SiLU(inplace=True),
            DoubleConv(out_c, out_c)
        )

    def forward(self, x):
        return self.down(x)

# Decode process in U-Net CNN
class Up(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        self.conv = DoubleConv(out_c + skip_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class SpectralAttention(nn.Module):
    """Channel attention for spectral features"""
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1), nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg_pool(x))


# Transformer module in the CNN bottleneck 
class SpatialTransformer(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2) 
        h = self.norm(x_flat)
        attn_out, _ = self.attn(h, h, h)
        x_flat = x_flat + attn_out
        x_flat = x_flat + self.ffn(self.norm2(x_flat))

        return x_flat.transpose(1, 2).reshape(B, C, H, W)

# MSFF Module
class MSFF(nn.Module):
    """Multi-Scale Feature Fusion"""
    def __init__(self, ch4, ch3, ch2, ch1, base=64):
        super().__init__()
        self.p4 = nn.Conv2d(ch4, base, 1)
        self.p3 = nn.Conv2d(ch3, base, 1)
        self.p2 = nn.Conv2d(ch2, base, 1)
        self.p1 = nn.Conv2d(ch1, base, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(base * 4, base * 2, 3, padding=1, bias=False), gn(base * 2), nn.SiLU(inplace=True),
            ResidualBlock(base * 2), ResidualBlock(base * 2)
        )

    def forward(self, f4, f3, f2, f1):
        size = f1.shape[-2:]
        p4 = self.p4(F.interpolate(f4, size=size, mode='bilinear', align_corners=False))
        p3 = self.p3(F.interpolate(f3, size=size, mode='bilinear', align_corners=False))
        p2 = self.p2(F.interpolate(f2, size=size, mode='bilinear', align_corners=False))
        p1 = self.p1(f1)
        return self.fuse(torch.cat([p4, p3, p2, p1], dim=1))

# Detail enhancement module
class DetailEnhancement(nn.Module):
    """Edge + channel/spatial attention + residual refine"""
    def __init__(self, ch):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // 4, 1), nn.SiLU(inplace=True),
            nn.Conv2d(ch // 4, ch, 1), nn.Sigmoid()
        )
        self.sa = nn.Sequential(nn.Conv2d(ch, 1, 7, padding=3), nn.Sigmoid())
        self.conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), gn(ch), nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x * self.ca(x) * self.sa(x)) + x


# Physics guided head module
class PhysicsGuidedHead(nn.Module):
    """H pseudo-inverse + exposure estimation"""
    def __init__(self, H_matrix: np.ndarray, out_channels: int = 31):
        super().__init__()
        H_tensor = torch.tensor(H_matrix, dtype=torch.float32)
        H_pinv = torch.linalg.pinv(H_tensor)
        self.register_buffer("H", H_tensor)
        self.register_buffer("H_pinv", H_pinv)

        self.exposure_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Softplus()
        )

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, rgb: torch.Tensor):
        B, C, H, W = rgb.shape
        scale = self.exposure_net(rgb).view(B, 1, 1, 1)
        rgb_corrected = rgb * scale

        rgb_flat = rgb_corrected.permute(0, 2, 3, 1)  
        hsi_init = torch.matmul(rgb_flat, self.H_pinv.T)  
        hsi_init = hsi_init.permute(0, 3, 1, 2) 
        hsi_init = torch.clamp(hsi_init, 0, 1)

        return hsi_init, scale

    def get_fusion_weight(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha)


# Main training module
class SpectralSRNet(nn.Module):
    """
    U-Net backbone + MSFF + DetailEnhancement + Physics Guidance
    With SpatialTransformer at bottleneck and SpectralAttention at output
    """
    def __init__(self, in_ch=3, out_ch=31, base=64, H_matrix=None):
        super().__init__()
        self.out_channels = out_ch

        # Physics guidance Load
        self.use_physics = H_matrix is not None
        if self.use_physics:
            self.physics_head = PhysicsGuidedHead(H_matrix, out_ch)
            self.physics_encoder = nn.Sequential(
                nn.Conv2d(out_ch, base, 3, padding=1, bias=False),
                gn(base), nn.SiLU(inplace=True),
                ResidualBlock(base)
            )
            self.init_fusion = nn.Sequential(
                nn.Conv2d(base * 2, base, 1, bias=False),
                gn(base), nn.SiLU(inplace=True)
            )

        # Initial convolution
        self.inc = DoubleConv(in_ch, base)

        # Encoder (downsampling)
        self.d1 = Down(base, base * 2)
        self.d2 = Down(base * 2, base * 4)
        self.d3 = Down(base * 4, base * 8)

        # Bottleneck with Self-Attention
        self.bot_conv = DoubleConv(base * 8, base * 8)
        self.bot_attn = SpatialTransformer(base * 8, num_heads=8)

        # Decoder (upsampling)
        self.u3 = Up(base * 8, base * 4, base * 4)
        self.u2 = Up(base * 4, base * 2, base * 2)
        self.u1 = Up(base * 2, base, base)
        self.ref = DoubleConv(base, base)

        # Multi-Scale Feature Fusion
        self.msff = MSFF(base * 4, base * 2, base, base, base)

        # Detail Enhancement
        self.det = DetailEnhancement(base * 2)

        # Output with SpectralAttention
        self.out = nn.Sequential(
            nn.Conv2d(base * 2, base, 3, padding=1, bias=False), gn(base), nn.SiLU(inplace=True),
            nn.Conv2d(base, out_ch, 3, padding=1),
            SpectralAttention(out_ch),
            nn.Tanh()  # Output residual in [-1, 1]
        )

        # Learnable residual scale
        self.residual_scale = nn.Parameter(torch.ones(1, out_ch, 1, 1) * 0.1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Physics guidance
        if self.use_physics:
            hsi_physics, _scale = self.physics_head(x)
            physics_feat = self.physics_encoder(hsi_physics)

        # Initial features
        x0 = self.inc(x)

        # Fuse physics and RGB features
        if self.use_physics:
            x0 = self.init_fusion(torch.cat([x0, physics_feat], dim=1))

        # Encoder
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        x3 = self.d3(x2)

        # Bottleneck with self-attention
        b = self.bot_attn(self.bot_conv(x3))

        # Decoder with skip connections
        u3 = self.u3(b, x2)
        u2 = self.u2(u3, x1)
        u1 = self.ref(self.u1(u2, x0))

        # Multi-scale fusion
        msff_out = self.msff(u3, u2, u1, u1)

        # Detail enhancement
        det_out = self.det(msff_out)

        # Output residual
        residual = self.out(det_out)

        # Final output
        if self.use_physics:
            out = hsi_physics + self.residual_scale * residual
        else:
            out = torch.sigmoid(residual)

        return torch.clamp(out, 0, 1)

    def get_physics_info(self) -> dict:
        if self.use_physics:
            return {
                "fusion_weight": float(self.physics_head.get_fusion_weight().item()),
                "residual_scale_mean": float(self.residual_scale.mean().item())
            }
        return {}


######################################
########## Loss Function #############
######################################

class MultiTaskLoss(nn.Module):
    """
    Improved uncertainty-weighted multi-task loss
    Key improvements:
    - PSNR normalization (MSE × psnr_scale) to match other loss scales
    - Tighter clamp range for more stable weights
    """
    def __init__(self, psnr_scale=20.0, clamp_lo=-1.5, clamp_hi=1.5, l2_reg=1e-4):
        super().__init__()
        self.psnr_scale = psnr_scale
        self.clamp_lo = clamp_lo
        self.clamp_hi = clamp_hi
        self.l2_reg = l2_reg

        # Learnable log-variance parameters
        self.log_vars = nn.ParameterDict({
            "sam": nn.Parameter(torch.zeros(1)),
            "psnr": nn.Parameter(torch.zeros(1)),
            "mrae": nn.Parameter(torch.zeros(1)),
            "edge": nn.Parameter(torch.zeros(1))
        })

        # Sobel filters for edge loss
        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        )

    def spectral_angle_mapper(self, pred, target):
        """SAM loss: 1 - cos(theta), normalized to [0, 1]"""
        B, C, H, W = pred.shape
        p = pred.view(B, C, -1)
        t = target.view(B, C, -1)
        p = F.normalize(p, dim=1, eps=1e-8)
        t = F.normalize(t, dim=1, eps=1e-8)
        cos = (p * t).sum(1).clamp(-0.9999, 0.9999)
        # Normalize by pi to get [0, 1] range
        ang = torch.acos(cos).mean() / math.pi
        return ang

    def mean_relative_absolute_error(self, pred, target):
        """MRAE loss, clamped and normalized"""
        mrae = (torch.abs(pred - target) / (target + 1e-8)).mean()
        return torch.clamp(mrae, 0, 10) / 10 

    def edge_loss(self, pred, target):
        """Edge preservation loss using Sobel gradients"""
        # Average across channels for edge detection
        if pred.size(1) > 1:
            pg = pred.mean(1, keepdim=True)
            tg = target.mean(1, keepdim=True)
        else:
            pg, tg = pred, target

        # Sobel gradients
        px = F.conv2d(pg, self.sobel_x, padding=1)
        py = F.conv2d(pg, self.sobel_y, padding=1)
        tx = F.conv2d(tg, self.sobel_x, padding=1)
        ty = F.conv2d(tg, self.sobel_y, padding=1)

        pe = torch.sqrt(px ** 2 + py ** 2 + 1e-8)
        te = torch.sqrt(tx ** 2 + ty ** 2 + 1e-8)

        return F.l1_loss(pe, te)

    def forward(self, pred, target):
        pred = torch.clamp(pred, 0, 1)
        target = torch.clamp(target, 0, 1)

        losses = {}

        # SAM loss (normalized to [0, 1])
        losses["sam"] = self.spectral_angle_mapper(pred, target)

        # PSNR loss (MSE × scale to match other losses)
        mse = F.mse_loss(pred, target)
        losses["psnr"] = mse * self.psnr_scale

        # MRAE loss (normalized to [0, 1])
        losses["mrae"] = self.mean_relative_absolute_error(pred, target)

        # Edge loss
        losses["edge"] = torch.clamp(self.edge_loss(pred, target), 0, 1)

        # Compute weighted total loss
        total = 0.0
        for k, v in losses.items():
            lv = torch.clamp(self.log_vars[k], self.clamp_lo, self.clamp_hi)
            precision = torch.exp(-lv)
            total = total + 0.5 * (precision * v + lv)

        # L2 regularization on log_vars
        reg = 0.0
        for p in self.log_vars.values():
            reg = reg + (p ** 2).mean()
        total = total + self.l2_reg * reg

        # Compute PSNR for logging
        with torch.no_grad():
            psnr = -10 * torch.log10(mse + 1e-10)

        # Build log dict
        log = {"loss_" + k: float(v.item()) for k, v in losses.items()}
        log.update({
            "weight_" + k: float(torch.exp(-torch.clamp(
                self.log_vars[k], self.clamp_lo, self.clamp_hi)).item())
            for k in losses.keys()
        })
        log["psnr"] = float(psnr.item())

        return total, log

    def get_weight_info(self) -> dict:
        """Get current weights for saturation detection"""
        return {
            k: float(torch.exp(-torch.clamp(
                self.log_vars[k], self.clamp_lo, self.clamp_hi)).item())
            for k in self.log_vars.keys()
        }


######################################
######### Training Strategy ##########
######################################

class EMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}

        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=(1.0 - self.decay))

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if name not in self.shadow:
                continue
            self.backup[name] = p.detach().clone()
            p.data.copy_(self.shadow[name].data)

    @torch.no_grad()
    def restore(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name not in self.backup:
                continue
            p.data.copy_(self.backup[name].data)
        self.backup = {}

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict):
        self.decay = float(state["decay"])
        shadow = state["shadow"]
        for k, v in shadow.items():
            if k in self.shadow and self.shadow[k].shape == v.shape:
                self.shadow[k].data.copy_(v.data)


# Weight Saturation Detection
def check_weight_saturation(loss_fn: MultiTaskLoss, threshold: float = 0.95) -> bool:
    """
    Check if weights are saturated at clamp boundaries
    Returns True if all weights are near the boundary
    """
    weights = loss_fn.get_weight_info()
    max_weight = math.exp(-loss_fn.clamp_lo)
    min_weight = math.exp(-loss_fn.clamp_hi)

    saturated_count = 0
    for k, w in weights.items():
        if w >= max_weight * threshold or w <= min_weight * (1 + (1 - threshold)):
            saturated_count += 1

    # Return True if more than half are saturated
    return saturated_count >= len(weights) // 2 + 1


# DDP Trainer
class DDPTrainer:
    def __init__(self, local_rank: int, global_rank: int, world_size: int, cfg: Config, args):
        self.local_rank = local_rank
        self.rank = global_rank
        self.world_size = world_size
        self.cfg = cfg
        self.args = args
        self.is_main = (self.rank == 0)

        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        import datetime as _dt
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=_dt.timedelta(minutes=60)
        )

        # Create timestamped run folder
        if self.is_main:
            date_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            run_dir = os.path.join(args.save_dir, f"run_{date_time}")
            os.makedirs(run_dir, exist_ok=True)
        else:
            run_dir = ""

        obj_list = [run_dir]
        dist.broadcast_object_list(obj_list, src=0)
        self.save_dir = obj_list[0]
        self.args.save_dir = self.save_dir

        set_seed(cfg.seed + self.rank)
        dist.barrier()

        # Datasets / Loaders
        self.train_set = ARAD1KDataset(cfg.dataset_path, "train", cfg.img_size, True, cfg.stride)
        self.val_set = ARAD1KDataset(cfg.dataset_path, "valid", cfg.img_size, False, cfg.stride)

        self.train_sampler = DistributedSampler(
            self.train_set, num_replicas=world_size, rank=self.rank, shuffle=True
        )

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=cfg.batch_size,
            sampler=self.train_sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True
        )

        self.val_loader = DataLoader(
            self.val_set,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        # Load H matrix
        H_matrix = None
        if cfg.use_physics_guidance and os.path.exists(cfg.H_matrix_path):
            H_matrix = np.load(cfg.H_matrix_path).astype(np.float32)
            if self.is_main:
                print(f"Loaded H matrix for physics guidance: {H_matrix.shape}")

        # Model
        base_model = SpectralSRNet(
            in_ch=cfg.in_channels,
            out_ch=cfg.out_channels,
            base=cfg.base_channels,
            H_matrix=H_matrix
        ).to(self.device)

        # Convert BN -> SyncBN for multi-GPU
        if world_size > 1:
            base_model = nn.SyncBatchNorm.convert_sync_batchnorm(base_model)

        # MultiTaskLoss with improved settings
        self.criterion = MultiTaskLoss(
            psnr_scale=cfg.psnr_scale,
            clamp_lo=cfg.clamp_lo,
            clamp_hi=cfg.clamp_hi,
            l2_reg=cfg.l2_reg
        ).to(self.device)

        # EMA
        self.ema = None
        if cfg.use_ema:
            self.ema = EMA(base_model, decay=cfg.ema_decay)

        # DDP wrap
        self.base = DDP(
            base_model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=True
        )

        # Optimizer (include loss parameters)
        self.opt_base = torch.optim.AdamW(
            list(self.base.parameters()) + list(self.criterion.parameters()),
            lr=cfg.base_lr,
            weight_decay=cfg.weight_decay
        )

        # Scheduler
        self.sch_base = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_base, T_max=cfg.num_epochs, eta_min=cfg.min_lr
        )

        self.scaler = GradScaler(enabled=cfg.use_amp)

        self.best_psnr = -1e9
        self.history = []

        if self.is_main:
            self._print_info()

    def _print_info(self):
        base_params = get_parameter_count(self.base.module) / 1e6
        print("\n" + "=" * 70)
        print("SpectralSRNet with Self-Attention + EMA + MultiTaskLoss")
        print("=" * 70)
        print(f"Physics Guidance: {self.cfg.use_physics_guidance}")
        print(f"EMA: {self.cfg.use_ema} (decay={self.cfg.ema_decay})")
        print(f"PSNR Scale: {self.cfg.psnr_scale}")
        print(f"Clamp Range: [{self.cfg.clamp_lo}, {self.cfg.clamp_hi}]")
        print(f"Base CNN: {base_params:.2f}M params")
        print(f"Epochs: {self.cfg.num_epochs}")
        print(f"Batch size: {self.cfg.batch_size} x {self.world_size} GPUs = {self.cfg.batch_size * self.world_size}")
        print(f"Train samples: {len(self.train_set)} | Val samples: {len(self.val_set)}")
        print(f"Save dir: {self.save_dir}")
        print("=" * 70 + "\n")

    def train_epoch(self, epoch: int) -> dict:
        self.base.train()
        self.train_sampler.set_epoch(epoch)

        total_loss = 0.0
        meter = defaultdict(float)
        n_batches = 0

        self.opt_base.zero_grad(set_to_none=True)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}") if self.is_main else self.train_loader

        for bi, (rgb, hsi) in enumerate(pbar):
            if isinstance(rgb, np.ndarray):
                rgb = torch.from_numpy(rgb).float()
            if isinstance(hsi, np.ndarray):
                hsi = torch.from_numpy(hsi).float()

            rgb = rgb.to(self.device, non_blocking=True)
            hsi = hsi.to(self.device, non_blocking=True)

            with autocast(self.cfg.use_amp):
                pred = self.base(rgb)
                loss_rec, log = self.criterion(pred, hsi)
                loss = loss_rec / max(1, self.cfg.accumulate_grad_batches)

            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                if self.is_main:
                    print(f"Warning: NaN/Inf loss at epoch {epoch}, batch {bi}. Skipping.")
                self.opt_base.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss).backward()

            do_step = ((bi + 1) % max(1, self.cfg.accumulate_grad_batches) == 0)

            if do_step:
                self.scaler.unscale_(self.opt_base)
                torch.nn.utils.clip_grad_norm_(
                    list(self.base.parameters()) + list(self.criterion.parameters()),
                    self.cfg.grad_clip
                )
                self.scaler.step(self.opt_base)
                self.scaler.update()
                self.opt_base.zero_grad(set_to_none=True)

                if self.ema is not None:
                    self.ema.update(self.base.module)

            total_loss += float(loss_rec.item())
            for k, v in log.items():
                meter[k] += float(v)
            n_batches += 1

            if self.is_main and (bi + 1) % self.cfg.log_interval == 0 and n_batches > 0:
                physics_info = self.base.module.get_physics_info()
                weights = self.criterion.get_weight_info()
                extra = f" | w_sam={weights['sam']:.2f}, w_psnr={weights['psnr']:.2f}"
                if physics_info:
                    extra += f" | rs={physics_info.get('residual_scale_mean', 0):.3f}"
                pbar.set_postfix_str(
                    f"loss={total_loss / n_batches:.4f}, psnr={meter.get('psnr', 0) / n_batches:.2f}{extra}"
                )

        log_epoch = {
            "epoch": int(epoch),
            "train_loss": total_loss / max(1, n_batches),
        }
        for k in meter:
            log_epoch[k] = meter[k] / max(1, n_batches)

        return log_epoch

    @torch.no_grad()
    def evaluate(self, epoch: int) -> dict:
        self.base.eval()

        # Use EMA weights for evaluation
        if self.ema is not None:
            self.ema.apply_shadow(self.base.module)

        psnr_list = []
        sam_list = []
        mrae_list = []

        for rgb, hsi in self.val_loader:
            if isinstance(rgb, np.ndarray):
                rgb = torch.from_numpy(rgb).float()
            if isinstance(hsi, np.ndarray):
                hsi = torch.from_numpy(hsi).float()

            rgb = rgb.to(self.device, non_blocking=True)
            hsi = hsi.to(self.device, non_blocking=True)

            pred = self.base(rgb)

            if torch.isnan(pred).any() or torch.isinf(pred).any():
                continue

            psnr_list.append(float(calc_psnr(pred, hsi).item()))
            sam_list.append(float(calc_sam_deg(pred, hsi).item()))
            mrae_list.append(float(calc_mrae(pred, hsi).item()))

        # Restore original weights
        if self.ema is not None:
            self.ema.restore(self.base.module)

        local_metrics = torch.tensor([
            np.mean(psnr_list) if psnr_list else 0.0,
            np.mean(sam_list) if sam_list else 0.0,
            np.mean(mrae_list) if mrae_list else 0.0,
            float(len(psnr_list)),
        ], device=self.device)

        dist.barrier()

        all_metrics = [torch.zeros_like(local_metrics) for _ in range(self.world_size)]
        dist.all_gather(all_metrics, local_metrics)

        total_samples = sum(m[3].item() for m in all_metrics)
        if total_samples > 0:
            psnr = sum(m[0].item() * m[3].item() for m in all_metrics) / total_samples
            sam = sum(m[1].item() * m[3].item() for m in all_metrics) / total_samples
            mrae = sum(m[2].item() * m[3].item() for m in all_metrics) / total_samples
        else:
            psnr = 0.0
            sam = 0.0
            mrae = 0.0

        log = {
            "val_psnr": float(psnr),
            "val_sam": float(sam),
            "val_mrae": float(mrae),
        }

        if log["val_psnr"] > self.best_psnr:
            self.best_psnr = log["val_psnr"]

        return log

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        if not self.is_main:
            return

        ckpt = {
            "epoch": int(epoch),
            "base_model": self.base.module.state_dict(),
            "opt_base": self.opt_base.state_dict(),
            "sch_base": self.sch_base.state_dict(),
            "criterion": self.criterion.state_dict(),
            "best_psnr": float(self.best_psnr),
            "config": vars(self.cfg),
        }
        if self.ema is not None:
            ckpt["ema"] = self.ema.state_dict()

        torch.save(ckpt, os.path.join(self.save_dir, "latest.pth"))

        if epoch % self.cfg.save_interval == 0:
            torch.save(ckpt, os.path.join(self.save_dir, f"epoch_{epoch}.pth"))

        if is_best:
            torch.save(ckpt, os.path.join(self.save_dir, "best.pth"))

        with open(os.path.join(self.save_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

    def fit(self):
        best_so_far = -1e9
        start_epoch = 1
        saturation_warning_count = 0

        # Resume
        if self.args.resume and os.path.exists(self.args.resume):
            if self.is_main:
                print(f"Resuming from checkpoint: {self.args.resume}")
            ckpt = torch.load(self.args.resume, map_location=self.device)

            self.base.module.load_state_dict(ckpt["base_model"], strict=True)
            self.opt_base.load_state_dict(ckpt["opt_base"])
            if "sch_base" in ckpt:
                self.sch_base.load_state_dict(ckpt["sch_base"])
            if "criterion" in ckpt:
                self.criterion.load_state_dict(ckpt["criterion"])
            if self.ema is not None and "ema" in ckpt:
                self.ema.load_state_dict(ckpt["ema"])

            start_epoch = int(ckpt["epoch"]) + 1
            best_so_far = float(ckpt.get("best_psnr", -1e9))
            self.best_psnr = best_so_far

            if self.is_main:
                print(f"Resumed from epoch {ckpt['epoch']}, best PSNR: {best_so_far:.2f}")

        for epoch in range(start_epoch, self.cfg.num_epochs + 1):
            tr_log = self.train_epoch(epoch)

            # Check weight saturation
            if self.is_main and epoch % 10 == 0:
                if check_weight_saturation(self.criterion):
                    saturation_warning_count += 1
                    weights = self.criterion.get_weight_info()
                    print(f"\nWeight saturation warning ({saturation_warning_count}/3)!")
                    print(f"   Weights: {weights}")
                    if saturation_warning_count >= 3:
                        print("Training stopped due to persistent weight saturation.")
                        break
                else:
                    saturation_warning_count = max(0, saturation_warning_count - 1)

            do_val = (epoch % self.cfg.val_interval == 0) or (epoch == self.cfg.num_epochs)
            if do_val:
                val_log = self.evaluate(epoch)

                if self.is_main:
                    tr_log.update(val_log)
                    physics_info = self.base.module.get_physics_info()
                    tr_log.update(physics_info)
                    tr_log.update(self.criterion.get_weight_info())

                    self.history.append(tr_log)

                    is_best = val_log["val_psnr"] > best_so_far
                    if is_best:
                        best_so_far = val_log["val_psnr"]

                    self.save_checkpoint(epoch, is_best=is_best)

                    weights = self.criterion.get_weight_info()
                    physics_str = ""
                    if physics_info:
                        physics_str = f" | rs={physics_info.get('residual_scale_mean', 0):.3f}"

                    print(
                        f"\n[Epoch {epoch}] "
                        f"Loss={tr_log['train_loss']:.4f} | "
                        f"Val-PSNR={val_log['val_psnr']:.2f} dB | "
                        f"Val-SAM={val_log['val_sam']:.2f}° | "
                        f"Weights: sam={weights['sam']:.2f}, psnr={weights['psnr']:.2f}, "
                        f"mrae={weights['mrae']:.2f}, edge={weights['edge']:.2f}"
                        f"{physics_str} | "
                        f"Best={self.best_psnr:.2f} dB\n"
                    )

                dist.barrier()

            self.sch_base.step()

        if self.is_main:
            print(f"\nTraining completed! Best PSNR: {self.best_psnr:.2f} dB")
            print(f"Checkpoints saved to: {self.save_dir}")

        dist.destroy_process_group()


######################################
############ Main Part ###############
######################################

def main():
    parser = argparse.ArgumentParser(description="SpectralSRNet + Self-Attention + EMA + MultiTaskLoss")
    parser.add_argument("--data_root", type=str, default="../../dataset/")
    parser.add_argument("--H_matrix_path", type=str, default="../../dataset/H_official_raw.npy")
    parser.add_argument("--save_dir", type=str, default="./checkpoints/")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_physics", action="store_true", help="Disable physics guidance")
    parser.add_argument("--no_ema", action="store_true", help="Disable EMA evaluation")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--psnr_scale", type=float, default=20.0)
    parser.add_argument("--test", action="store_true", help="Run quick test")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_rank = int(os.environ.get("RANK", 0))

    cfg = Config()
    cfg.dataset_path = args.data_root
    cfg.H_matrix_path = args.H_matrix_path
    cfg.batch_size = args.batch_size
    cfg.img_size = args.img_size
    cfg.stride = args.stride
    cfg.num_workers = args.num_workers
    cfg.num_epochs = args.epochs
    cfg.use_physics_guidance = not args.no_physics
    cfg.use_ema = not args.no_ema
    cfg.ema_decay = float(args.ema_decay)
    cfg.psnr_scale = args.psnr_scale

    if args.test:
        cfg.num_epochs = 2
        cfg.val_interval = 1
        cfg.save_interval = 1
        cfg.log_interval = 1

    trainer = DDPTrainer(local_rank, global_rank, world_size, cfg, args)
    trainer.fit()


if __name__ == "__main__":
    main()
