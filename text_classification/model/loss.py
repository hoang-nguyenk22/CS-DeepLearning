import torch.nn.functional as F
import torch
from torch import nn


def get_cb_weights(df, mlb, beta=0.999):
    """
    A simple weight assigning formula: W = (1 - beta)/(1 - beta^n)
    Satisfy: f'(x1) < f'(x2)
    """
    df['tags'] = df['tags'].apply(clean_tags)
    tag_counts = df['tags'].explode().value_counts()
    
    
    samples_per_class = np.array([tag_counts.get(cls, 0) for cls in mlb.classes_])
    
    e = 1.0 - np.power(beta, samples_per_class)
    
    weights = (1.0 - beta) / np.where(e == 0, 0.01, e)
    
    #weights = weights / np.sum(weights) * len(mlb.classes_)
    #weights = np.power(weights, 0.67)
    weights = weights / np.min(weights) # Normalize: min_weight = min_weight/min_weight = 1
    
    print(f"--> Min Weight (Head): {np.min(weights):.4f}")
    print(f"--> Max Weight (Tail): {np.max(weights):.4f}")
    print(f"--> Average Weight: {np.average(weights):.4f}")
    return torch.tensor(weights, dtype=torch.float)

class ASLCB(nn.Module):
    def __init__(self, gamma_neg=2, gamma_pos=0.0, clip=0.05, cb_weights=None):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.cb_weights = cb_weights # Tensor (662,)

    def forward(self, x, y):
        # x: logits, y: targets
        xs_pos = torch.sigmoid(x)
        xs_neg = 1 - xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic BCE
        loss_pos = y * torch.log(xs_pos.clamp(min=1e-7))
        loss_neg = (1 - y) * torch.log(xs_neg.clamp(min=1e-7))
        loss = - (loss_pos + loss_neg)

        # Asymmetric Focusing
        final_weight = torch.where(y > 0.5, 
                                   torch.pow(1 - xs_pos, self.gamma_pos), 
                                   torch.pow(1 - xs_neg, self.gamma_neg))
        loss *= final_weight

        # Apply Class-Balanced weights
        if self.cb_weights is not None:
            loss *= self.cb_weights.to(x.device)

        return loss.mean()