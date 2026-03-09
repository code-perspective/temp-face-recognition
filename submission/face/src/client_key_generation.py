#!/usr/bin/env python3
"""
client_key_generation.py — Stage 2: FHE key generation.

Called by harness as: ./client_key_generation <size>

Generates CKKS keys via orion.init_scheme(io_mode=save).
Does NOT call fit or compile — stage 7 (server_encrypted_compute) handles that.
Also saves a fit_sample.npy for stage 7 to use.
"""
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    load_submission_config, get_face_params,
    load_detector, preprocess_one_image, init_orion_scheme
)


def main():
    if len(sys.argv) < 2:
        print("Usage: client_key_generation <size>", flush=True)
        sys.exit(1)

    size = int(sys.argv[1])
    t0 = time.time()

    cfg = load_submission_config()
    params = get_face_params(size)

    keys_dir = params.iodir() / "public_keys"
    keys_dir.mkdir(parents=True, exist_ok=True)

    # Write current_size.txt for server_preprocess_model (which has no args)
    repo_root = Path(__file__).resolve().parents[3]
    (repo_root / "io").mkdir(parents=True, exist_ok=True)
    (repo_root / "io" / "current_size.txt").write_text(str(size))

    dataset_path = repo_root / "datasets" / "face_dataset.npy"
    print(f"[client_key_generation] Loading master dataset for fit sample...", flush=True)
    dataset = np.load(dataset_path, allow_pickle=True)
    img0 = dataset[0][0]  # (3, H, W) uint8 RGB — first image of first pair

    print(f"[client_key_generation] Detecting + aligning face for fit sample...", flush=True)
    detector = load_detector()
    patches = preprocess_one_image(detector, img0, cfg["input_size"])

    fit_arr = np.stack([p.numpy() for p in patches], axis=0)  # (N, 1, 3, 32, 32)
    fit_path = keys_dir / "fit_sample.npy"
    np.save(fit_path, fit_arr)
    print(f"[client_key_generation] fit_sample.npy saved ({len(patches)} patches) → {fit_path}", flush=True)

    t_keygen = time.time()
    init_orion_scheme(cfg, params, "save")
    elapsed_keygen = time.time() - t_keygen

    elapsed = time.time() - t0
    print(f"[client_key_generation] CKKS keys saved in {elapsed_keygen:.1f}s  "
          f"total={elapsed:.1f}s", flush=True)
    print(f"[client_key_generation] Keys → {keys_dir / 'keys.h5'}", flush=True)


if __name__ == "__main__":
    main()
