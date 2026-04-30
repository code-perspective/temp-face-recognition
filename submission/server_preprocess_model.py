#!/usr/bin/env python3
"""
server_preprocess_model.py — stub server preprocessing step.

server_encrypted_compute compiles the pipeline in memory (io_mode=none),
so the expensive fit+compile+save step is not needed here. The only downstream
dependency is input_level.txt (read by client_encode_encrypt_input to set the
encryption level), which equals len(LogQ) - 1 and can be read directly from
the CKKS config.

client_key_generation's keys.h5 (SK only) is sufficient for all downstream
stages: server_encrypted_compute loads the SK and regenerates rotation keys
and plaintext diagonals entirely in Go memory during its own compile step.
"""
import sys
import time
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import load_submission_config, get_face_params, get_repo_root


def main():
    t0 = time.time()
    cfg = load_submission_config()

    size_file = get_repo_root() / "io" / "current_size.txt"
    if not size_file.exists():
        print(f"[server_preprocess_model] ERROR: {size_file} not found. "
              f"Run client_key_generation first.", flush=True)
        sys.exit(1)
    size = int(size_file.read_text().strip())
    params = get_face_params(size)
    keys_dir = params.iodir() / "public_keys"

    # Derive input_level from the CKKS config: len(LogQ) - 1.
    with open(cfg["orion_config"]) as f:
        orion_cfg = yaml.safe_load(f)
    try:
        logq = orion_cfg["ckks_params"]["LogQ"]
    except KeyError as e:
        print(f"[server_preprocess_model] ERROR: missing key {e} in {cfg['orion_config']}", flush=True)
        sys.exit(1)
    input_level = len(logq) - 1

    (keys_dir / "input_level.txt").write_text(str(input_level))

    elapsed = time.time() - t0
    print(f"[server_preprocess_model] stub: input_level={input_level}  "
          f"total={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
