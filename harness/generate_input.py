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

import sys
import numpy as np
from pathlib import Path
from utils import parse_submission_arguments


def main():
    size, params, seed, _, _ = parse_submission_arguments('Generate face pairs for FHE benchmark.')

    master_npy    = params.rootdir / "datasets" / "face_dataset.npy"
    master_labels = params.rootdir / "datasets" / "face_dataset_labels.txt"

    if not master_npy.exists():
        sys.exit(f"[harness] Error: master dataset not found: {master_npy}")

    data   = np.load(master_npy, allow_pickle=True)
    labels = [int(l.strip()) for l in master_labels.read_text().strip().splitlines() if l.strip()]
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
        arrays[f'pair_{i:05d}_img0'] = pair[0]
        arrays[f'pair_{i:05d}_img1'] = pair[1]
    np.savez(out_dir / "test_pairs.npz", **arrays)
    (out_dir / "test_labels.txt").write_text(
        "\n".join(str(labels[i]) for i in indices)
    )

    print(f"[harness] Sampled {batch_size} face pairs (seed={seed}) → {out_dir}")


if __name__ == "__main__":
    main()
