#!/usr/bin/env python3
"""
generate_dataset.py - Provision and validate the face pair dataset.

The benchmark dataset (face_dataset.npy + face_dataset_labels.txt) is hosted on
Hugging Face (halmsu/celeba-1024-pairs). If the files are not already present at
the path passed as argument, they are downloaded from there and cached locally.
This script then validates the files and prints basic statistics.

To run fully offline, place the two files next to the given path beforehand.

Usage:  python3 generate_dataset.py <dataset_npy_path>
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
import shutil
import numpy as np
from pathlib import Path

# Hugging Face dataset repo hosting the benchmark face pairs.
HF_DATASET_REPO = "halmsu/celeba-1024-pairs"
DATASET_NPY     = "face_dataset.npy"
DATASET_LABELS  = "face_dataset_labels.txt"


def _download_from_hf(dest_dir: Path):
    """Download face_dataset.npy + labels from Hugging Face into dest_dir."""
    from huggingface_hub import hf_hub_download
    dest_dir.mkdir(parents=True, exist_ok=True)
    for fname in (DATASET_NPY, DATASET_LABELS):
        target = dest_dir / fname
        if target.exists():
            continue
        print(f"[harness] Downloading {fname} from HF dataset {HF_DATASET_REPO} ...",
              flush=True)
        cached = hf_hub_download(repo_id=HF_DATASET_REPO, filename=fname,
                                 repo_type="dataset")
        shutil.copyfile(cached, target)


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: generate_dataset.py <dataset_npy_path>")

    npy_path    = Path(sys.argv[1])
    labels_path = npy_path.parent / DATASET_LABELS

    # Provision from Hugging Face if either file is missing.
    if not npy_path.exists() or not labels_path.exists():
        try:
            _download_from_hf(npy_path.parent)
        except Exception as e:
            sys.exit(f"[harness] Error: dataset not found locally and Hugging Face "
                     f"download failed: {e}")

    if not npy_path.exists():
        sys.exit(f"[harness] Error: dataset not found: {npy_path}")
    if not labels_path.exists():
        sys.exit(f"[harness] Error: labels not found: {labels_path}")

    data   = np.load(npy_path, allow_pickle=True)  # object array; each pair is [img0, img1]
    labels = [int(l.strip()) for l in labels_path.read_text().strip().splitlines() if l.strip()]

    if len(data) != len(labels):
        sys.exit(f"[harness] Error: pair count mismatch — "
                 f"npy has {len(data)} pairs, labels has {len(labels)}")
    if len(data) == 0:
        sys.exit("[harness] Error: dataset is empty (0 pairs found)")

    # Images are stored as original JPEG bytes (compact) or raw (3, H, W) arrays.
    example = data[0][0]
    if isinstance(example, (bytes, bytearray, np.bytes_)):
        import io
        from PIL import Image
        example_shape = np.asarray(Image.open(io.BytesIO(bytes(example))).convert("RGB")).shape
    else:
        example_shape = np.asarray(example).shape

    n_same = sum(labels)
    n_diff = len(labels) - n_same
    print(f"[harness] Face dataset: {len(data)} pairs  example_img_shape={example_shape}  "
          f"same={n_same}  diff={n_diff}")


if __name__ == "__main__":
    main()
