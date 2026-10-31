"""Neural model: a bidirectional GRU that reads the hand-skeleton sequence frame by frame.
Input: (B, T, V=22, C=9) channel tensors built by `train_dl.make_channels`."""

import torch
import torch.nn as nn

from .constants import NUM_CLASSES, NUM_JOINTS


class GRUNet(nn.Module):
    """Sequence baseline that ignores the graph structure (flattens joints per frame)."""

    def __init__(self, in_ch=9, num_nodes=NUM_JOINTS, hidden=128, layers=2,
                 n_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        d = in_ch * num_nodes
        self.bn = nn.BatchNorm1d(d)
        self.inp = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * hidden, n_classes))

    def forward(self, x):                                  # (B, T, V, C)
        B, T, V, C = x.shape
        x = x.reshape(B, T, V * C)
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(self.inp(x))
        return self.head(torch.cat([h.mean(1), h.max(1).values], dim=1))


def build_model(kind: str) -> nn.Module:
    if kind == "gru":
        return GRUNet()
    raise ValueError(f"unknown model {kind}")
