import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights

import torch
import torch.nn as nn

class MaxHead(nn.Module):
    def __init__(self, dim, num_labels):
        super().__init__()
        self.fc = nn.Linear(dim, num_labels)

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        max_val, max_indices = torch.max(x, dim=1)
        logits = self.fc(max_val)
        
        mask = torch.zeros_like(x, device=x.device)
        b_idx = torch.arange(batch_size, device=x.device).view(-1, 1).expand(-1, dim)
        d_idx = torch.arange(dim, device=x.device).view(1, -1).expand(batch_size, -1)
        
        mask[b_idx, max_indices, d_idx] = 1.0
        token_importance = mask.sum(dim=2) / dim
        
        return {
            "l3": logits,
            "attention": token_importance
        }

class GlobalAttention(nn.Module):
    def __init__(self, dim, num_labels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.fc = nn.Linear(dim, num_labels)

    def forward(self, x):
        e = self.proj(x)
        w = F.softmax(e.squeeze(-1), dim=1)
        c = (x * w.unsqueeze(-1)).sum(dim=1)
        return self.fc(c)

class LabelWiseAttention(nn.Module):
    def __init__(self, dim, num_labels):
        super().__init__()
        self.q = nn.Parameter(torch.randn(num_labels, dim))
        self.fc = nn.Linear(dim, 1)

    def forward(self, x):
        q = self.q.to(x.dtype)

        att = torch.matmul(x, q.transpose(0, 1))
        att = F.softmax(att, dim=1)
        out = torch.matmul(att.transpose(1, 2), x)
        return self.fc(out).squeeze(-1)