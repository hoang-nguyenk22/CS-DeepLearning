import timm
import torch
import torch.nn as nn
from typing import Optional

def get_model(model_name, num_classes=100, pretrained=True, freeze_backbone=False):
    """
    Returns a PyTorch model with the specified architecture and modified classification head.
    Available model_names: 'resnet50', 'efficientnet_b0', 'vit_b_16', 'deit_s'
    """
    
    model_mapping = {
        'resnet50': 'resnet50.tv_in1k', 
        'efficientnet_b0': 'efficientnet_b0.ra_in1k',
        'vit_b_16': 'vit_base_patch16_224.augreg_in21k_ft_in1k',
        'deit_s': 'deit_small_patch16_224.fb_in1k'
    }
    
    if model_name not in model_mapping:
        raise ValueError(f"Unknown model_name: {model_name}. Choose from {list(model_mapping.keys())}")
        
    timm_name = model_mapping[model_name]
    
    print(f"Loading {timm_name}: Adaptation to {num_classes} classes")
    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
    
    if freeze_backbone:
        print("Freezing backbone for linear probing.")
        for param in model.parameters():
            param.requires_grad = False
            
        for param in model.get_classifier().parameters():
            param.requires_grad = True
    
    return model


def get_layer_wise_optimizer(model, model_name: str, base_lr: float = 1e-4, decay: float = 0.65):
    """
    Builds layer-wise learning rate decay (LLRD) parameter groups.
    ViT: Each transformer block gets LR scaled by decay.
    CNN: Simple 2-group split (backbone vs head).
    """
    param_groups = []
    assigned_ids: set = set()

    def _add_group(params, lr, name):
        unique = [p for p in params if p.requires_grad and id(p) not in assigned_ids]
        if unique:
            assigned_ids.update(id(p) for p in unique)
            param_groups.append({'params': unique, 'lr': lr, 'name': name})

    if model_name in ('vit_b_16', 'deit_s'):
        blocks = list(model.blocks)

        # 1. Head
        _add_group(list(model.get_classifier().parameters()), base_lr, 'head')

        # 2. Transformer blocks
        for i, block in enumerate(reversed(blocks)):
            lr = base_lr * (decay ** (i + 1))
            _add_group(list(block.parameters()), lr, f'block_{len(blocks)-1-i}')

        # 3. Remaining params (patch_embed, pos_embed, cls_token, head norm, etc.)
        #    Everything not yet assigned gets the smallest LR automatically.
        embed_lr = base_lr * (decay ** (len(blocks) + 1))
        remaining = [p for name, p in model.named_parameters()
                     if p.requires_grad and id(p) not in assigned_ids]
        if remaining:
            assigned_ids.update(id(p) for p in remaining)
            param_groups.append({'params': remaining, 'lr': embed_lr, 'name': 'embed'})

    else:
        # CNN fallback: head at base_lr, rest at 0.1× base_lr
        _add_group(list(model.get_classifier().parameters()), base_lr, 'head')
        remaining = [p for p in model.parameters()
                     if p.requires_grad and id(p) not in assigned_ids]
        if remaining:
            assigned_ids.update(id(p) for p in remaining)
            param_groups.append({'params': remaining, 'lr': base_lr * 0.1, 'name': 'backbone'})

    return param_groups

if __name__ == "__main__":
    # Test model creation and forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    num_classes = 100
    
    models_to_test = ['resnet50', 'efficientnet_b0', 'vit_b_16', 'deit_s']
    
    for m_name in models_to_test:
        print(f"\n--- Testing {m_name} ---")
        model = get_model(m_name, num_classes=num_classes)
        
        # Test shape
        output = model(dummy_input)
        print(f"Output shape: {output.shape} (Expected: 2, {num_classes})")
        assert output.shape == (2, num_classes), f"Shape mismatch for {m_name}!"
        
        # Total parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total Parameters: {total_params / 1e6:.2f} M")
