#!/usr/bin/env python3
"""
generate_dataset.py - Validate the pre-provided face pair dataset.

The benchmark dataset (face_dataset.npy + face_dataset_labels.txt) must be
placed by the user at the path passed as argument before running the harness.
This script validates the files exist and prints basic statistics.

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
import numpy as np
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: generate_dataset.py <dataset_npy_path>")

    npy_path    = Path(sys.argv[1])
    labels_path = npy_path.parent / "face_dataset_labels.txt"

    if not npy_path.exists():
        sys.exit(f"[harness] Error: dataset not found: {npy_path}\n"
                 f"         Place the pre-provided face_dataset.npy there before running.")
    if not labels_path.exists():
        sys.exit(f"[harness] Error: labels not found: {labels_path}")

    data   = np.load(npy_path, allow_pickle=True)  # object array of (2, C, H, W) pairs
    labels = [int(l.strip()) for l in labels_path.read_text().strip().splitlines() if l.strip()]

    if len(data) != len(labels):
        sys.exit(f"[harness] Error: pair count mismatch — "
                 f"npy has {len(data)} pairs, labels has {len(labels)}")
    if len(data) == 0:
        sys.exit("[harness] Error: dataset is empty (0 pairs found)")

    n_same = sum(labels)
    n_diff = len(labels) - n_same
    print(f"[harness] Face dataset: {len(data)} pairs  example_img_shape={data[0][0].shape}  "
          f"same={n_same}  diff={n_diff}")


if __name__ == "__main__":
    main()
