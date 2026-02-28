#!/usr/bin/env python3
"""
metrics.py - Face verification quality metrics.

calculate_face_metrics() reads cosine similarity scores and ground-truth labels
from files, sweeps similarity thresholds over the full dataset, and returns
EER and TAR@FAR=1%/0.1%.
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

import numpy as np
from pathlib import Path

def calculate_face_metrics(gt_labels_file: Path, scores_file: Path, tag: str) -> dict:
    """
    Compute EER and TAR@FAR from cosine similarity scores and ground-truth labels.

    Uses a global threshold sweep over the full dataset (no cross-validation),
    which is appropriate for small datasets where KFold calibration is noisy.

    Args:
        gt_labels_file: path to test_labels.txt (one int per line, 0 or 1)
        scores_file:    path to similarity scores file (one float per line)
        tag:            label for printed output

    Returns:
        dict with keys: eer, tar_far_1_percent, tar_far_01_percent
        Returns empty dict if fewer than 2 pairs.
    """
    labels = [int(l.strip()) for l in Path(gt_labels_file).read_text().strip().splitlines() if l.strip()]
    scores = [float(s.strip()) for s in Path(scores_file).read_text().strip().splitlines() if s.strip()]

    n = min(len(labels), len(scores))
    if n < 2:
        print(f"[harness] {tag}: score={scores[0]:.6f}  label={labels[0]}")
        return {}

    labels = np.array(labels[:n], dtype=bool)
    scores = np.array(scores[:n], dtype=float)

    # Sweep similarity thresholds: predict same-person when score >= threshold
    thresholds = np.linspace(scores.min() - 1e-6, scores.max() + 1e-6, 2000)
    tprs, fprs = [], []
    for t in thresholds:
        pred_same = scores >= t
        tp = np.sum(pred_same &  labels)
        fp = np.sum(pred_same & ~labels)
        fn = np.sum(~pred_same &  labels)
        tn = np.sum(~pred_same & ~labels)
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)

    tprs = np.array(tprs)
    fprs = np.array(fprs)
    fnrs = 1.0 - tprs

    eer_idx = np.argmin(np.abs(fprs - fnrs))
    eer = float((fprs[eer_idx] + fnrs[eer_idx]) / 2)

    # TAR@FAR=1% and TAR@FAR=0.1%: largest TAR where FPR <= target
    tar_1pct  = float(tprs[np.searchsorted(-fprs, -0.01,  side='left')])
    tar_01pct = float(tprs[np.searchsorted(-fprs, -0.001, side='left')])

    result = {
        "eer":              eer,
        "tar_far_1_percent":  tar_1pct,
        "tar_far_01_percent": tar_01pct,
    }
    print(f"[harness] {tag}: "
          f"EER={eer:.4f}  "
          f"TAR@FAR=1%={tar_1pct:.4f}  "
          f"TAR@FAR=0.1%={tar_01pct:.4f}")
    return result
