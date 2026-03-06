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
        self.bn = nn.BatchNorm1d(out_channels) # Kept
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
        # Increased capacity (channels) and depth
        self.can_branch = nn.Sequential(
            DepthwiseSeparableTCN(input_dim, 64, dilation=1),
            nn.BatchNorm1d(64),
            DepthwiseSeparableTCN(64, 128, dilation=2),
            nn.BatchNorm1d(128),
            DepthwiseSeparableTCN(128, 64, dilation=4),
            nn.AdaptiveAvgPool1d(1) # Output (B, 64, 1) -> (B, 64)
        )
        
        # ETH Branch (CNN)
        # Input channels 1 based on L116: Conv2d(1, 16, ...)
        self.eth_branch = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1) # Output (B, 32, 1, 1) -> (B, 32)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64 + 32, 128), # 64 from CAN + 32 from ETH
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # self.can_head = nn.Linear(32, 2)
        # self.eth_head = nn.Linear(32, 2)
        
        self.quant = QuantStub()

        self.dequant = DeQuantStub()

    def _extract_features(self, can, eth):
        can = self.quant(can)
        eth = self.quant(eth)
        
        # CAN: (B, L, C) -> (B, C, L) because Conv1d expects (N, C, L)
        # However, checking disassembly L130: can.permute(0, 2, 1). 
        # If input is (B, L, C=10), then permute gives (B, 10, L).
        # can_branch output: (B, 32, 1) due to AdaptiveAvgPool.
        # flatten(1) -> (B, 32).
        can_out = self.can_branch(can.permute(0, 2, 1))
        can_feat = can_out.flatten(1)
        
        # ETH: 
        # Disassembly L133: if eth.dim() == 5
        eth_feat = None
        if eth.dim() == 5:
            # (B, T, C, H, W)
            # L134 unpacks B, T, C, H, W
            # L136 view(B*T, C, H, W)
            # L138 eth_branch(eth_4d).flatten(1) -> (B*T, 32)
            # L140 view(B, T, -1)[:, -1, :] -> (B, 32) (taking last frame)
            
            b, t, c, h, w = eth.shape
            eth_4d = eth.view(b * t, c, h, w)
            eth_feat_4d = self.eth_branch(eth_4d).flatten(1)
            eth_feat = eth_feat_4d.view(b, t, -1)[:, -1, :]
        else:
            # L143 eth_branch(eth).flatten(1)
            eth_feat = self.eth_branch(eth).flatten(1)
            
        return can_feat, eth_feat

    def forward(self, can, eth):
        can_feat, eth_feat = self._extract_features(can, eth)
        
        # L149 cat((can_feat, eth_feat), dim=1) -> (B, 64)
        # classifier -> (B, 2)
        fused_logits = self.classifier(torch.cat((can_feat, eth_feat), dim=1))
        
        return self.dequant(fused_logits)

    def forward_diagnostics(self, can, eth):
        can_feat, eth_feat = self._extract_features(can, eth)
        
        fused_logits = self.classifier(torch.cat((can_feat, eth_feat), dim=1))
        can_logits = self.can_head(can_feat)
        eth_logits = self.eth_head(eth_feat)
        
        return {
            'fused_logits': self.dequant(fused_logits),
            'can_logits': self.dequant(can_logits),
            'eth_logits': self.dequant(eth_logits),
            'can_feat': self.dequant(can_feat),
            'eth_feat': self.dequant(eth_feat)
        }

