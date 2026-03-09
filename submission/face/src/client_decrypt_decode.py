#!/usr/bin/env python3
"""
client_decrypt_decode.py — Stage 8: Decrypt score ciphertexts.

Called by harness as: ./client_decrypt_decode <size>

Initializes orion with io_mode=load (loads SK from keys.h5), then decrypts
each p{i:04d}_score.bin and writes one similarity float per line to
encrypted_model_predictions.txt.
"""
import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    load_submission_config, get_face_params,
    init_orion_scheme
)


def main():
    if len(sys.argv) < 2:
        print("Usage: client_decrypt_decode <size>", flush=True)
        sys.exit(1)

    size = int(sys.argv[1])
    cfg = load_submission_config()
    params = get_face_params(size)

    # Initialize orion with SK (io_mode=load) — only SK is needed for decryption.
    from orion.core import scheme as _scheme
    from orion.backend.python.tensors import CipherTensor

    print(f"[client_decrypt_decode] init_scheme(io_mode=load)...", flush=True)
    init_orion_scheme(cfg, params, "load")

    download_dir = params.iodir() / "ciphertexts_download"
    score_files  = sorted(download_dir.glob("p????_score.bin"))

    if not score_files:
        print(f"[client_decrypt_decode] ERROR: no score files in {download_dir}", flush=True)
        sys.exit(1)

    out_path = params.get_encrypted_model_predictions_file()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        for score_path in score_files:
            ctxt = CipherTensor.deserialize(_scheme, pickle.loads(score_path.read_bytes()))
            sim  = float(ctxt.decrypt().decode().flatten()[0])
            f.write(f"{sim:.6f}\n")

    print(f"[client_decrypt_decode] Decrypted {len(score_files)} scores → {out_path}", flush=True)


if __name__ == "__main__":
    main()
