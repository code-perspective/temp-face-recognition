"""
CryptoFace-compatible PCNN for Orion FHE.

This module implements a Patch-based CNN that matches CryptoFace's architecture
for encrypted face recognition inference.

Architecture:
- Configurable number of patches (4, 9, or 16)
- N backbones (one per patch) with HerPN activations
- N linear layers (256→256 each)
- Aggregation via summation
- Final L2 normalization via polynomial approximation (L2NormPoly)

Reference: CryptoFace CVPR 2025
"""
import torch.nn as nn
import orion.nn as on
from orion.models.pcnn import Backbone, L2NormPoly

__all__ = ['CryptoFacePCNN', 'CryptoFaceNet']


class CryptoFacePCNN(on.Module):
    """
    CryptoFace-compatible Patch-based CNN for encrypted face recognition.

    Matches the CryptoFace inference architecture:
    - Per-patch processing through identical backbones
    - Per-patch linear transformations (256→256)
    - Feature aggregation via summation
    - Final normalization

    Args:
        input_size (int): Input image size (e.g., 64, 96, 128)
        patch_size (int): Patch size (e.g., 32)
        embedding_dim (int): Output embedding dimension (default: 256)

    Examples:
        >>> model = CryptoFacePCNN(input_size=64, patch_size=32)   # 4 patches
        >>> model = CryptoFacePCNN(input_size=96, patch_size=32)   # 9 patches
        >>> model = CryptoFacePCNN(input_size=128, patch_size=32)  # 16 patches
    """

    def __init__(
        self,
        input_size=64,
        patch_size=32,
        embedding_dim=256,
        l2_norm_coeffs=None,
    ):
        super().__init__()

        self.input_size = input_size
        self.patch_size = patch_size
        self.H = self.W = input_size // patch_size  # Grid size
        self.N = self.H * self.W  # Number of patches

        # Backbone output dimension (always 64*2*2 = 256)
        self.backbone_dim = 64 * 2 * 2

        self.embedding_dim = embedding_dim

        print(f"[CryptoFacePCNN] input={input_size}×{input_size}  "
              f"patches={self.N} ({self.H}×{self.W})  embed_dim={embedding_dim}")

        self.nets = nn.ModuleList([
            Backbone(
                output_size=(2, 2),  # 64 channels × 2×2 = 256 features
                input_size=patch_size
            )
            for _ in range(self.N)
        ])

        self.linear = nn.ModuleList([
            on.Linear(self.backbone_dim, embedding_dim)
            for _ in range(self.N)
        ])

        if l2_norm_coeffs is not None:
            a, b, c = l2_norm_coeffs
        else:
            # Default coefficients fitted on face_dataset.npy (CelebA, 100 pairs)
            # SOS range: min=108, max=505, mean=267, std=76
            # Fitted via 3-point method: x3=192, x1=264, x2=337
            a, b, c = 3.497142e-07, -3.070292e-04, 1.182291e-01
        self.normalization = L2NormPoly(a, b, c, embedding_dim)

    def init_orion_params(self):
        """
        Initialize HerPN parameters and fuse backbone BN into the following
        Linear layer (saves 1 level of FHE depth). Must be called before orion.fit().
        """
        for i, net in enumerate(self.nets):
            net.init_orion_params()
            net.fuse_bn_into_linear(self.linear[i])


def CryptoFaceNet(input_size, embedding_dim=256, l2_norm_coeffs=None):
    """
    Create a CryptoFace PCNN model for the given input size.

    Supported input sizes and their patch configurations:
    - 64:  2×2 grid → 4  patches of 32×32
    - 96:  3×3 grid → 9  patches of 32×32
    - 128: 4×4 grid → 16 patches of 32×32

    Args:
        input_size (int): Input image size. Must be a multiple of 32.
        embedding_dim (int): Output embedding dimension (default: 256)
        l2_norm_coeffs (tuple): L2 norm polynomial coefficients (a, b, c)

    Returns:
        CryptoFacePCNN instance
    """
    if input_size % 32 != 0:
        raise ValueError(f"input_size must be a multiple of 32, got {input_size}")
    return CryptoFacePCNN(
        input_size=input_size,
        patch_size=32,
        embedding_dim=embedding_dim,
        l2_norm_coeffs=l2_norm_coeffs,
    )
