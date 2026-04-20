from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, groups: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch, kernel_size=3, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = ConvBNAct(in_ch, in_ch, kernel_size=3, stride=stride, groups=in_ch)
        self.pointwise = ConvBNAct(in_ch, out_ch, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class SmallCNNEncoder(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(in_channels, 32, 3, 1),
            nn.MaxPool2d(2),
            ConvBNAct(32, 64, 3, 1),
            nn.MaxPool2d(2),
            ConvBNAct(64, 96, 3, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(96, 128)
        self.output_dim = 128

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        return self.proj(h)


class WideCNNEncoder(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(in_channels, 64, 3, 1),
            ConvBNAct(64, 64, 3, 1),
            nn.MaxPool2d(2),
            ConvBNAct(64, 128, 3, 1),
            ConvBNAct(128, 128, 3, 1),
            nn.MaxPool2d(2),
            ConvBNAct(128, 160, 3, 1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(160, 192)
        self.output_dim = 192

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        return self.proj(h)


class TinyResNetEncoder(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.stem = ConvBNAct(in_channels, 32, 3, 1)
        self.layer1 = ResidualBlock(32, 32, 1)
        self.layer2 = ResidualBlock(32, 64, 2)
        self.layer3 = ResidualBlock(64, 64, 1)
        self.layer4 = ResidualBlock(64, 96, 2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(96, 160)
        self.output_dim = 160

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.pool(h).flatten(1)
        return self.proj(h)


class TinyMobileNetEncoder(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(in_channels, 32, 3, 1),
            DepthwiseSeparableConv(32, 48, 1),
            DepthwiseSeparableConv(48, 64, 2),
            DepthwiseSeparableConv(64, 96, 1),
            DepthwiseSeparableConv(96, 128, 2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(128, 160)
        self.output_dim = 160

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x).flatten(1)
        return self.proj(h)


@dataclass
class ModelSpec:
    arch_name: str
    feature_dim: int
    latent_dim: int
    num_classes: int


class PALMFLModel(nn.Module):
    def __init__(
        self,
        arch_name: str,
        in_channels: int,
        latent_dim: int,
        num_classes: int,
        adapter_hidden_dim: int = 128,
        head_hidden_dim: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        encoder, feature_dim = build_encoder(arch_name, in_channels)
        self.arch_name = arch_name
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        self.adapter = nn.Sequential(
            nn.Linear(feature_dim, adapter_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(adapter_hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        if head_hidden_dim > 0:
            self.head = nn.Sequential(
                nn.Linear(latent_dim, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden_dim, num_classes),
            )
        else:
            self.head = nn.Linear(latent_dim, num_classes)

        self.spec = ModelSpec(
            arch_name=arch_name,
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            num_classes=num_classes,
        )

    def encode_backbone(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def encode_latent(self, x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        z = self.adapter(self.encode_backbone(x))
        if normalize:
            z = F.normalize(z, p=2, dim=-1)
        return z

    def classify_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)

    def forward(self, x: torch.Tensor, return_latent: bool = False, normalize_latent: bool = False):
        z = self.encode_latent(x, normalize=normalize_latent)
        logits = self.classify_latent(z)
        if return_latent:
            return logits, z
        return logits


def build_encoder(arch_name: str, in_channels: int) -> Tuple[nn.Module, int]:
    arch_name = arch_name.lower()
    if arch_name == "small_cnn":
        model = SmallCNNEncoder(in_channels)
    elif arch_name == "wide_cnn":
        model = WideCNNEncoder(in_channels)
    elif arch_name == "tiny_resnet":
        model = TinyResNetEncoder(in_channels)
    elif arch_name == "tiny_mobilenet":
        model = TinyMobileNetEncoder(in_channels)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}")
    return model, int(model.output_dim)


def available_architectures() -> Dict[str, str]:
    return {
        "small_cnn": "3-layer convolutional encoder",
        "wide_cnn": "wider convolutional encoder",
        "tiny_resnet": "small residual encoder",
        "tiny_mobilenet": "depthwise-separable mobile encoder",
    }


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError("off_diagonal expects a square matrix")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def covariance_regularizer(z: torch.Tensor) -> torch.Tensor:
    if z.ndim != 2:
        raise ValueError("Latent tensor must have shape [batch, dim]")
    batch_size = z.size(0)
    if batch_size < 2:
        return z.new_tensor(0.0)
    z = z - z.mean(dim=0, keepdim=True)
    z = z / (z.std(dim=0, unbiased=False, keepdim=True) + 1e-4)
    cov = (z.T @ z) / batch_size
    diag_loss = (torch.diagonal(cov) - 1.0).pow(2).mean()
    off_loss = off_diagonal(cov).pow(2).mean()
    return diag_loss + off_loss


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()
