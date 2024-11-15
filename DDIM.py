"""
Implementation of Denoising Diffusion Implicit Models (DDIM)
Based on the paper "Denoising Diffusion Implicit Models" by Song et al. (2020)
https://arxiv.org/abs/2010.02502

Key improvements:
- Deterministic sampling process
- Accelerated sampling with fewer steps
- Local attention mechanism
- Position embeddings
- Classifier-free guidance

Note: Implementation assisted by AI (ChatGPT, Claude)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import torchvision
import numpy as np
from torch.nn import init
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.image.fid import FrechetInceptionDistance
import matplotlib.pyplot as plt
import os

class LocalAttention(nn.Module):
    def __init__(self, channels, window_size=8):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def window_partition(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H // self.window_size, self.window_size, 
                  W // self.window_size, self.window_size)
        windows = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        windows = windows.view(-1, C, self.window_size, self.window_size)
        return windows

    def window_reverse(self, windows, H, W):
        B = int(windows.shape[0] / (H * W / self.window_size / self.window_size))
        x = windows.view(B, H // self.window_size, W // self.window_size, 
                        self.channels, self.window_size, self.window_size)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, self.channels, H, W)
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        windows = self.window_partition(x)
        
        qkv = self.qkv(windows).chunk(3, dim=1)
        q, k, v = map(lambda t: t.view(t.shape[0], self.channels, -1), qkv)
        
        attn = (q.transpose(-2, -1) @ k) * (self.channels ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = (v @ attn.transpose(-2, -1)).view_as(windows)
        out = self.proj(out)
        
        return self.window_reverse(out, H, W)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t_emb):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t_emb)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
    
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, skip_x, t_emb):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t_emb)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class ImprovedUNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128, emb_dim=time_dim)
        self.sa1 = LocalAttention(128)
        self.down2 = Down(128, 256, emb_dim=time_dim)
        self.sa2 = LocalAttention(256)
        self.down3 = Down(256, 256, emb_dim=time_dim)
        self.sa3 = LocalAttention(256)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128, emb_dim=time_dim)
        self.sa4 = LocalAttention(128)
        self.up2 = Up(256, 64, emb_dim=time_dim)
        self.sa5 = LocalAttention(64)
        self.up3 = Up(128, 64, emb_dim=time_dim)
        self.sa6 = LocalAttention(64)
        
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)
        self.label_emb = nn.Embedding(10, time_dim)
        
        self.initialize_weights()

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x, t, y=None):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        
        if y is not None:
            t += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        
        return self.outc(x)

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=32, device="cuda", eta=0.0):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.eta = eta

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        ε = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * ε, ε

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, labels=None, cfg_scale=3, sampling_steps=50):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)

            timesteps = torch.linspace(self.noise_steps - 1, 0, sampling_steps).long()
            
            for i in range(0, len(timesteps) - 1):
                t = timesteps[i]
                t_next = timesteps[i + 1]
                t_vec = torch.ones(n, device=self.device) * t
                
                if labels is not None:
                    predicted_noise = model(x, t_vec, labels)
                    predicted_noise_uncond = model(x, t_vec, None)
                    predicted_noise = predicted_noise_uncond + cfg_scale * (predicted_noise - predicted_noise_uncond)
                else:
                    predicted_noise = model(x, t_vec)

                alpha = self.alpha_hat[t]
                alpha_next = self.alpha_hat[t_next]
                sigma = self.eta * torch.sqrt((1 - alpha_next) / (1 - alpha)) * torch.sqrt(1 - alpha / alpha_next)
                c1 = torch.sqrt(alpha_next / alpha)
                c2 = torch.sqrt(1 - alpha_next - sigma ** 2)
                current_predicted_x0 = (x - torch.sqrt(1 - alpha) * predicted_noise) / torch.sqrt(alpha)
                noise = torch.randn_like(x) if self.eta > 0 else 0
                x = c1 * current_predicted_x0 - c2 * predicted_noise + sigma * noise

        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        return x

def calculate_fid(real_images, fake_images):
    fid = FrechetInceptionDistance(normalize=True)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    return fid.compute()

def save_samples(samples, path):
    grid = utils.make_grid(samples, nrow=4, normalize=True)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.savefig(path)
    plt.close()

def train():
    device = "cuda"
    lr = 1e-4
    batch_size = 12
    num_epochs = 300
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    model = ImprovedUNet(c_in=3, c_out=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    warmup_steps = 1000
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs * len(dataloader) - warmup_steps, eta_min=1e-6)
    
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=32, device=device, eta=0.0)  # eta=0.0 for deterministic DDIM
    
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            if i + epoch * len(dataloader) < warmup_steps:
                lr = optimizer.param_groups[0]["lr"] * i / warmup_steps
                optimizer.param_groups[0]["lr"] = lr
            
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            
            if np.random.random() < 0.1:
                labels = None
                
            predicted_noise = model(x_t, t, labels)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if i + epoch * len(dataloader) >= warmup_steps:
                scheduler.step()
                
            total_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Epoch {epoch} | step {i:03d} Loss: {loss.item():.5f}")

        if epoch % 10 == 0:
            model.eval()
            samples = []
            for class_label in range(10):
                class_labels = torch.tensor([class_label] * 4).to(device)
                sample = diffusion.sample(model, n=4, labels=class_labels, cfg_scale=3, sampling_steps=50)
                samples.append(sample)
            samples = torch.cat(samples, dim=0)
            save_samples(samples, f"samples/epoch_{epoch}.png")
            
            real_batch = next(iter(dataloader))[0]
            fake_batch = diffusion.sample(model, n=batch_size, sampling_steps=50)
            fid_score = calculate_fid(real_batch, fake_batch)
            print(f"Epoch {epoch} | FID Score: {fid_score:.2f}")

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs,
    }, "checkpoints/final_model.pt")

if __name__ == "__main__":
    train()