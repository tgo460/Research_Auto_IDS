"""
architecture_unimodal.py — Standalone unimodal IDS models.

Phase 2: Provides two models that share the same backbone blocks as
TinyHybridStudent (architecture_improved.py) but operate on a SINGLE bus
modality. This allows fair, apples-to-apples comparison between:

  - UnimodalCANModel  : CAN-only intrusion detector (TCN-based)
  - UnimodalETHModel  : ETH-only intrusion detector (CNN-based)

and the joint multimodal TinyHybridStudent, proving (or disproving) that
fusion is strictly necessary.
"""

import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub


# ── Shared building block (identical to architecture_improved.py) ─────────────

class _DepthwiseSeparableTCN(nn.Module):
    """Depthwise-separable TCN block with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            padding=self.padding, dilation=dilation, groups=in_channels,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise(x)
        if self.padding > 0:
            out = out[..., :-self.padding]
        out = self.pointwise(out)
        out = self.bn(out)
        return self.act(out)


# ── CAN-only model ─────────────────────────────────────────────────────────────

class UnimodalCANModel(nn.Module):
    """
    TCN-based CAN-only intrusion detector.

    Architecture:
        CAN TCN backbone  →  AdaptiveAvgPool1d  →  Linear classifier head

    Input:  (B, L, input_dim)  — a batch of CAN windows
    Output: (B, num_classes)   — raw logits

    The TCN backbone is identical to the CAN branch of TinyHybridStudent so
    that multimodal vs. unimodal comparison is controlled.
    """

    def __init__(self, input_dim: int = 16, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.input_dim = input_dim

        self.backbone = nn.Sequential(
            _DepthwiseSeparableTCN(input_dim, hidden_dim, dilation=1),
            nn.BatchNorm1d(hidden_dim),
            _DepthwiseSeparableTCN(hidden_dim, hidden_dim * 2, dilation=2),
            nn.BatchNorm1d(hidden_dim * 2),
            _DepthwiseSeparableTCN(hidden_dim * 2, hidden_dim, dilation=4),
            nn.AdaptiveAvgPool1d(1),    # → (B, hidden_dim, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, input_dim)  — batch of CAN feature windows
        Returns:
            logits: (B, num_classes)
        """
        x = self.quant(x)
        # Conv1d expects (B, C, L)
        feat = self.backbone(x.permute(0, 2, 1)).flatten(1)   # (B, hidden_dim)
        out = self.classifier(feat)
        return self.dequant(out)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled feature vector before the classifier head."""
        x = self.quant(x)
        return self.backbone(x.permute(0, 2, 1)).flatten(1)


# ── ETH-only model ─────────────────────────────────────────────────────────────

class UnimodalETHModel(nn.Module):
    """
    CNN-based ETH-only intrusion detector.

    Architecture:
        Conv2d backbone  →  AdaptiveAvgPool2d  →  Linear classifier head

    Input:  (B, 1, H, W)     — a batch of ETH image windows (64×64 after Phase 1)
    Output: (B, num_classes) — raw logits

    The CNN backbone is identical to the ETH branch of TinyHybridStudent.
    Works with any image size thanks to AdaptiveAvgPool2d.
    """

    def __init__(self, num_classes: int = 2, feat_dim: int = 32):
        super().__init__()
        self.feat_dim = feat_dim

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, feat_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),    # → (B, feat_dim, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.BatchNorm1d(feat_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feat_dim * 2, num_classes),
        )

        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) or (B, T, 1, H, W) — ETH image(s)
        Returns:
            logits: (B, num_classes)
        """
        x = self.quant(x)
        if x.dim() == 5:
            # (B, T, 1, H, W) → average over time axis
            B, T, C, H, W = x.shape
            feat = self.backbone(x.view(B * T, C, H, W)).flatten(1)  # (B*T, feat_dim)
            feat = feat.view(B, T, -1).mean(dim=1)                   # (B, feat_dim)
        else:
            feat = self.backbone(x).flatten(1)    # (B, feat_dim)
        out = self.classifier(feat)
        return self.dequant(out)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled feature vector before the classifier head."""
        x = self.quant(x)
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            feat = self.backbone(x.view(B * T, C, H, W)).flatten(1)
            return feat.view(B, T, -1).mean(dim=1)
        return self.backbone(x).flatten(1)
