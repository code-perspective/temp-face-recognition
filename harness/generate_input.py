#!/usr/bin/env python3
"""
generate_input.py - Sample face pairs for one benchmark run.

Reads the master face dataset (face_dataset.npy) and samples batch_size
pairs into the size-specific intermediate directory. The submission's
client_preprocess_input reads from there and applies its own preprocessing
(resize, normalize, patch extraction, etc.).

Output format: test_pairs.npz with keys pair_NNNNN_img0 / pair_NNNNN_img1,
each a (3, H, W) uint8 numpy array in RGB channel order. Images may have
different spatial dimensions across pairs.
"""
# Copyright 2025 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import sys
import numpy as np
from pathlib import Path
from PIL import Image
from utils import parse_submission_arguments


def _to_chw_uint8(elem):
    """Return a (3, H, W) uint8 RGB array from either raw ndarray or encoded
    image bytes. The master dataset stores original JPEG file bytes to keep the
    committed file small; per-run test_pairs.npz stays decoded uint8 so the
    submission contract is unchanged."""
    if isinstance(elem, (bytes, bytearray, np.bytes_)):
        img = Image.open(io.BytesIO(bytes(elem))).convert("RGB")
        return np.asarray(img, dtype=np.uint8).transpose(2, 0, 1)
    return np.asarray(elem)


def main():
    size, params, seed, _, _ = parse_submission_arguments('Generate face pairs for FHE benchmark.')

    master_npy    = params.rootdir / "datasets" / "face_dataset.npy"
    master_labels = params.rootdir / "datasets" / "face_dataset_labels.txt"

    if not master_npy.exists():
        sys.exit(f"[harness] Error: master dataset not found: {master_npy}")
    if not master_labels.exists():
        sys.exit(f"[harness] Error: master labels file not found: {master_labels}")

    data   = np.load(master_npy, allow_pickle=True)  # object array of (2, C, H, W) pairs
    labels = [int(l.strip()) for l in master_labels.read_text().strip().splitlines() if l.strip()]
    if len(labels) != len(data):
        sys.exit(f"[harness] Error: labels count ({len(labels)}) does not match dataset size ({len(data)})")
    n_total    = len(data)
    batch_size = params.get_batch_size()

    if batch_size > n_total:
        sys.exit(f"[harness] Error: batch_size={batch_size} exceeds dataset size={n_total}")

    rng     = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=batch_size, replace=False)

    out_dir = params.dataset_intermediate_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for i, idx in enumerate(indices):
        pair = data[idx]
        arrays[f'pair_{i:05d}_img0'] = _to_chw_uint8(pair[0])
        arrays[f'pair_{i:05d}_img1'] = _to_chw_uint8(pair[1])
    np.savez(out_dir / "test_pairs.npz", **arrays)
    (out_dir / "test_labels.txt").write_text(
        "\n".join(str(labels[i]) for i in indices) + "\n"
    )

    seed_str = str(seed) if seed is not None else "random"
    print(f"[harness] Sampled {batch_size} face pairs (seed={seed_str}) → {out_dir}")


if __name__ == "__main__":
    main()
