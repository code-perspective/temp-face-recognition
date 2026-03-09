#!/usr/bin/env python3
"""
client_encode_encrypt_input.py — Stage 6: Encode and encrypt patches.

Called by harness as: ./client_encode_encrypt_input <size>

Initializes orion with io_mode=load (loads PK from keys.h5 for encryption),
reads input_level from public_keys/input_level.txt, then encodes and encrypts
each patch .npy file. Ciphertexts are pickled to .bin files.

Does NOT need fit/compile/preload_all — encryption only requires the public key.
"""
import sys
import pickle
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    load_submission_config, get_face_params,
    init_orion_scheme
)


def main():
    if len(sys.argv) < 2:
        print("Usage: client_encode_encrypt_input <size>", flush=True)
        sys.exit(1)

    size = int(sys.argv[1])
    cfg = load_submission_config()
    params = get_face_params(size)

    import orion

    # Lightweight init: load scheme (PK needed for encryption) + read input_level.
    init_orion_scheme(cfg, params, "load")
    input_level = int((params.iodir() / "public_keys" / "input_level.txt").read_text().strip())

    inter_dir = params.io_intermediate_dir()  # io/<size>/intermediate/
    upload_dir = params.iodir() / "ciphertexts_upload"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Find all patch files, sorted for deterministic ordering
    patch_files = sorted(inter_dir.glob("p????_i?_b*.npy"))

    count = 0
    for patch_path in patch_files:
        arr = np.load(patch_path)       # (1, 3, 32, 32) float32
        tensor = torch.from_numpy(arr)
        ctxt = orion.encrypt(orion.encode(tensor, input_level))
        out_path = upload_dir / (patch_path.stem + ".bin")
        out_path.write_bytes(pickle.dumps(ctxt.serialize()))
        count += 1

    print(f"[client_encode_encrypt_input] Encrypted {count} patches → {upload_dir}", flush=True)


if __name__ == "__main__":
    main()
