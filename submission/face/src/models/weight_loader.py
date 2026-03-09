"""
Weight Loader for CryptoFace Pretrained Checkpoints

This module loads CryptoFace PyTorch checkpoints and maps them to Orion's
CryptoFacePCNN model structure.

Key transformations:
1. Load backbone weights (conv, HerPN, pooling) for each patch network
2. Chunk concatenated linear.weight [256, 1024] → 4×[256, 256] per-patch weights
3. Fuse final BatchNorm statistics into per-patch linear layers
4. Handle normalization (ChannelSquare replacement for BatchNorm1d)
"""

import torch
from pathlib import Path
from typing import Union, Dict


def load_cryptoface_checkpoint(model, checkpoint_path: Union[str, Path]) -> None:
    """
    Load CryptoFace pretrained weights into Orion CryptoFacePCNN model.

    Args:
        model: Orion CryptoFacePCNN model instance
        checkpoint_path: Path to .ckpt file
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'backbone' not in ckpt:
        raise ValueError("Checkpoint must contain 'backbone' key")

    state_dict = ckpt['backbone']

    N = model.N
    embedding_dim = model.embedding_dim
    backbone_dim = model.backbone_dim

    print(f"[weight_loader] Loading checkpoint: {checkpoint_path.name}  "
          f"({len(state_dict)} params, {N} patches)", flush=True)

    # Step 1: Load backbone weights for each patch network
    for i in range(N):
        _load_backbone_weights(model.nets[i], state_dict, i)

    # Step 2: Fuse BatchNorm into Linear layer (following CryptoFace pcnn.py:78-83),
    # then chunk the fused weight into N per-patch pieces.
    if 'linear.weight' not in state_dict or 'linear.bias' not in state_dict:
        raise ValueError("Checkpoint missing 'linear.weight' or 'linear.bias'")
    if 'bn.running_mean' not in state_dict or 'bn.running_var' not in state_dict:
        raise ValueError("Checkpoint missing BatchNorm statistics")

    full_weight = state_dict['linear.weight']  # [embedding_dim, N*backbone_dim]
    full_bias = state_dict['linear.bias']      # [embedding_dim]
    bn_mean = state_dict['bn.running_mean']    # [embedding_dim]
    bn_var = state_dict['bn.running_var']      # [embedding_dim]
    bn_eps = model.normalization.eps

    if full_weight.shape != (embedding_dim, N * backbone_dim):
        raise ValueError(
            f"Linear weight shape mismatch: expected [{embedding_dim}, {N*backbone_dim}], "
            f"got {full_weight.shape}"
        )

    # Normalize weight: weight_fused = weight / sqrt(var + eps)
    weight_fused = torch.divide(full_weight.T, torch.sqrt(bn_var + bn_eps))
    weight_fused = weight_fused.T  # [embedding_dim, N*backbone_dim]

    # Normalize bias: bias_fused = (bias - mean) / sqrt(var + eps)
    bias_fused = torch.divide(full_bias - bn_mean, torch.sqrt(bn_var + bn_eps))

    # Divide bias by number of patches (following CryptoFace pcnn.py:85)
    bias_fused = bias_fused / N

    # Chunk fused weight into N pieces along dim=1
    chunked_weights = torch.chunk(weight_fused, N, dim=1)  # N × [embedding_dim, backbone_dim]

    for i in range(N):
        model.linear[i].weight.data = chunked_weights[i].clone()
        model.linear[i].bias.data = bias_fused.clone()  # All patches get same fused bias

    # Step 3: Set BatchNorm to identity (mean=0, var=1) since it's fused into linear layers
    model.normalization.running_mean.data = torch.zeros_like(bn_mean)
    model.normalization.running_var.data = torch.ones_like(bn_var)

    model.eval()
    print(f"[weight_loader] Checkpoint loaded.", flush=True)


def _load_backbone_weights(
    backbone_module,
    state_dict: Dict[str, torch.Tensor],
    patch_idx: int,
) -> None:
    """
    Load weights for a single Backbone network from CryptoFace checkpoint.

    Remaps CryptoFace naming convention to Orion naming:
    - CryptoFace: nets.{i}.layers.{j}.herpn{k}.bn{n}.{param}
    - Orion:      nets.{i}.layer{j+1}.bn{n}_{k}.{param}

    Example mappings:
    - nets.0.layers.0.herpn1.bn0.running_mean → nets.0.layer1.bn0_1.running_mean
    - nets.0.layers.0.herpn2.bn1.running_var → nets.0.layer1.bn1_2.running_var
    - nets.0.herpnpool.herpn.bn0.running_mean → nets.0.herpnpool.bn0.running_mean

    Also extracts herpn.weight and herpn.bias and stores them as attributes on
    the layer modules for use in init_orion_params().
    """
    prefix = f'nets.{patch_idx}.'
    patch_keys = {k: v for k, v in state_dict.items() if k.startswith(prefix)}
    backbone_state = {}
    herpn_params = {}

    for key, value in patch_keys.items():
        key_no_prefix = key[len(prefix):]

        # Remap CryptoFace keys to Orion keys
        # CryptoFace: layers.0.herpn1.bn0 → Orion: layer1.bn0_1
        # CryptoFace: layers.0.herpn2.bn0 → Orion: layer1.bn0_2
        if key_no_prefix.startswith('layers.'):
            parts = key_no_prefix.split('.')
            layer_num = int(parts[1])  # e.g., 0, 1, 2, 3, 4
            rest = '.'.join(parts[2:])  # e.g., "herpn1.bn0.running_mean"

            # CryptoFace uses layers.0-4 → Orion uses layer1-5
            orion_layer_name = f'layer{layer_num + 1}'

            # Handle HerPN weight and bias (store separately)
            if rest == 'herpn1.weight':
                herpn_params[f'{orion_layer_name}.herpn1_weight'] = value
                continue
            elif rest == 'herpn1.bias':
                herpn_params[f'{orion_layer_name}.herpn1_bias'] = value
                continue
            elif rest == 'herpn2.weight':
                herpn_params[f'{orion_layer_name}.herpn2_weight'] = value
                continue
            elif rest == 'herpn2.bias':
                herpn_params[f'{orion_layer_name}.herpn2_bias'] = value
                continue
            # Handle shortcut remapping (CryptoFace: shortcut.0/shortcut.1 → Orion: shortcut_conv/shortcut_bn)
            elif 'shortcut.0' in rest:
                rest = rest.replace('shortcut.0', 'shortcut_conv')
            elif 'shortcut.1' in rest:
                rest = rest.replace('shortcut.1', 'shortcut_bn')
            # Handle HerPN BatchNorm remapping
            elif 'herpn1' in rest:
                # herpn1.bn0 → bn0_1, herpn1.bn1 → bn1_1, herpn1.bn2 → bn2_1
                rest = rest.replace('herpn1.bn0', 'bn0_1')
                rest = rest.replace('herpn1.bn1', 'bn1_1')
                rest = rest.replace('herpn1.bn2', 'bn2_1')
            elif 'herpn2' in rest:
                # herpn2.bn0 → bn0_2, herpn2.bn1 → bn1_2, herpn2.bn2 → bn2_2
                rest = rest.replace('herpn2.bn0', 'bn0_2')
                rest = rest.replace('herpn2.bn1', 'bn1_2')
                rest = rest.replace('herpn2.bn2', 'bn2_2')

            new_key = f'{orion_layer_name}.{rest}'
        else:
            # Handle herpnpool weight and bias
            if key_no_prefix == 'herpnpool.herpn.weight':
                herpn_params['herpnpool.herpn_weight'] = value
                continue
            elif key_no_prefix == 'herpnpool.herpn.bias':
                herpn_params['herpnpool.herpn_bias'] = value
                continue
            # Handle herpnpool remapping
            elif 'herpnpool.herpn.bn0' in key_no_prefix:
                new_key = key_no_prefix.replace('herpnpool.herpn.bn0', 'herpnpool.bn0')
            elif 'herpnpool.herpn.bn1' in key_no_prefix:
                new_key = key_no_prefix.replace('herpnpool.herpn.bn1', 'herpnpool.bn1')
            elif 'herpnpool.herpn.bn2' in key_no_prefix:
                new_key = key_no_prefix.replace('herpnpool.herpn.bn2', 'herpnpool.bn2')
            else:
                new_key = key_no_prefix

        backbone_state[new_key] = value

    # Load into backbone (strict=False to allow missing keys like jigsaw)
    missing_keys, unexpected_keys = backbone_module.load_state_dict(
        backbone_state, strict=False
    )

    # Store herpn.weight and herpn.bias as attributes on the layer modules
    # These will be used by init_orion_params() instead of ones/zeros
    for key, value in herpn_params.items():
        parts = key.split('.')
        if len(parts) == 2:
            layer_name, param_name = parts
            layer_module = getattr(backbone_module, layer_name)
            try:
                device = next(layer_module.parameters()).device
                value = value.to(device)
            except StopIteration:
                pass
            setattr(layer_module, param_name, value)

    if missing_keys or unexpected_keys:
        missing_no_jigsaw = [k for k in missing_keys if 'jigsaw' not in k]
        if missing_no_jigsaw:
            print(f"    [weight_loader] Missing keys (patch {patch_idx}): {missing_no_jigsaw[:3]}")
        if unexpected_keys:
            print(f"    [weight_loader] Unexpected keys (patch {patch_idx}): {unexpected_keys[:3]}")
