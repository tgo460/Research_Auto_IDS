import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub

class DepthwiseSeparableTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(DepthwiseSeparableTCN, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, 
                                   padding=self.padding, dilation=dilation, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels) # Added BatchNorm
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.depthwise(x)
        if self.padding > 0:
            out = out[..., :-self.padding]
        out = self.pointwise(out)
        out = self.bn(out)
        out = self.act(out)
        return out

class TinyHybridStudent(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, num_classes=2):
        super(TinyHybridStudent, self).__init__()
        
        # CAN Branch (TCN)
        # Improved Architecture: Increased Width [64, 128, 64] vs Original [32, 64, 32]
        self.can_branch = nn.Sequential(
            DepthwiseSeparableTCN(input_dim, 64, dilation=1), # input_dim flexible
            nn.BatchNorm1d(64),
            DepthwiseSeparableTCN(64, 128, dilation=2),
            nn.BatchNorm1d(128),
            DepthwiseSeparableTCN(128, 64, dilation=4),
            nn.AdaptiveAvgPool1d(1) # (B, 64, 1)
        )
        
        # ETH Branch (CNN)
        self.eth_branch = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1) # (B, 32, 1, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 + 32, 128), # 64 (CAN) + 32 (ETH)
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3), # Added Dropout for regularization
            nn.Linear(128, num_classes)
        )
        
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def _extract_features(self, can, eth):
        can = self.quant(can)
        eth = self.quant(eth)
        
        # CAN: (B, L, C) -> (B, C, L)
        can_out = self.can_branch(can.permute(0, 2, 1))
        can_feat = can_out.flatten(1) # (B, 64)
        
        # ETH: 
        if eth.dim() == 5:
            B, T, C, H, W = eth.size()
            eth = eth.view(B*T, C, H, W)
            eth_out = self.eth_branch(eth)
            eth_feat = eth_out.flatten(1) # (B*T, 32)
            if T > 1:
                eth_feat = eth_feat.view(B, T, -1).mean(dim=1)
        else:
            eth_out = self.eth_branch(eth)
            eth_feat = eth_out.flatten(1)

        return can_feat, eth_feat

    def forward(self, can, eth):
        can_feat, eth_feat = self._extract_features(can, eth)
        x = torch.cat((can_feat, eth_feat), dim=1) # (B, 96)
        out = self.classifier(x)
        out = self.dequant(out)
        return out
