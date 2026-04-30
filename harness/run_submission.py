#!/usr/bin/env python3
"""
run_submission.py - Run the Face Verification FHE benchmark end-to-end.
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

import subprocess
import sys
import numpy as np
import utils
from params import instance_name
from metrics import calculate_face_metrics


def main():
    # 0. Parse arguments and initialise
    size, params, seed, num_runs, clrtxt = utils.parse_submission_arguments(
        'Run Face Verification FHE benchmark.'
    )
    print(f"\n[harness] Running face verification for {instance_name(size)}")

    utils.ensure_directories(params.rootdir)

    harness_dir = params.rootdir / "harness"
    exec_dir    = params.rootdir / "submission"

    # Remove and re-create IO directory
    io_dir = params.iodir()
    if io_dir.exists():
        subprocess.run(["rm", "-rf", str(io_dir)], check=True)
    io_dir.mkdir(parents=True)
    utils.log_step(0, "Init", True)

    # 1. Validate pre-provided face dataset
    dataset_npy = params.rootdir / "datasets" / "face_dataset.npy"
    subprocess.run(
        ["python3", harness_dir / "generate_dataset.py", str(dataset_npy)],
        check=True
    )
    utils.log_step(1, "Harness: Face dataset validation")

    # 2. Client: key generation
    utils.run_exe_or_python(exec_dir, "client_key_generation", str(size))
    utils.log_step(2, "Client: Key generation")
    utils.log_size(io_dir / "public_keys", "Client: Public and evaluation keys")

    # 3. Server: model preprocessing
    utils.run_exe_or_python(exec_dir, "server_preprocess_model")
    utils.log_step(3, "Server: Model preprocessing")

    # One RNG seeded once — each run draws a different per-run seed from it.
    rng = np.random.default_rng(seed)

    # Run stages 4-10 once per requested run
    for run in range(num_runs):
        run_path = params.measuredir() / f"results-{run+1}.json"
        utils.reset_run_state()
        if num_runs > 1:
            print(f"\n         [harness] Run {run+1} of {num_runs}")

        # 4. Sample input pairs from master dataset
        cmd = ["python3", harness_dir / "generate_input.py", str(size)]
        if seed is not None:
            genqry_seed = int(rng.integers(0, 0x7fffffff))
            cmd.extend(["--seed", str(genqry_seed)])
        subprocess.run(cmd, check=True)
        utils.log_step(4, "Harness: Face pair input generation")

        # 5. Client: input preprocessing (submission-specific)
        utils.run_exe_or_python(exec_dir, "client_preprocess_input", str(size))
        utils.log_step(5, "Client: Input preprocessing")

        # 6. Client: encode and encrypt
        utils.run_exe_or_python(exec_dir, "client_encode_encrypt_input", str(size))
        utils.log_step(6, "Client: Input encryption")
        utils.log_size(io_dir / "ciphertexts_upload", "Client: Encrypted input")

        # 7. Server: encrypted face verification
        utils.run_exe_or_python(exec_dir, "server_encrypted_compute", str(size))
        utils.log_step(7, "Server: Encrypted face verification")
        utils.log_size(io_dir / "ciphertexts_download", "Server: Encrypted results")

        # 8. Client: decrypt
        utils.run_exe_or_python(exec_dir, "client_decrypt_decode", str(size))
        utils.log_step(8, "Client: Result decryption")

        # 9. Client: postprocess
        utils.run_exe_or_python(exec_dir, "client_postprocess", str(size))
        utils.log_step(9, "Client: Result postprocessing")

        # 10. Quality check
        gt_labels       = params.get_ground_truth_labels_file()
        encrypted_scores = params.get_encrypted_model_predictions_file()
        harness_scores  = params.get_harness_model_predictions_file()
        test_pairs      = params.get_test_input_file()

        if not encrypted_scores.exists():
            print(f"[harness] Error: result file not found: {encrypted_scores}")
            sys.exit(1)

        # 10.1: ArcFace cleartext reference
        # Skip rerun only for single-run benchmarks where scores are already cached.
        # For num_runs > 1, pairs change each run so cleartext must be recomputed.
        if clrtxt == 1 or not harness_scores.exists() or num_runs > 1:
            subprocess.run(
                ["python3", harness_dir / "cleartext_impl.py",
                 str(test_pairs), str(harness_scores)],
                check=True
            )
        else:
            print(f"[harness] Skipping cleartext rerun (cached: {harness_scores})")
        utils.log_step(10.1, "Harness: ArcFace cleartext reference")

        # 10.2: Metrics for encrypted model
        metrics_enc = calculate_face_metrics(gt_labels, encrypted_scores, "Encrypted model")
        utils.log_quality(metrics_enc, "Encrypted model")

        # 10.3: Metrics for ArcFace reference
        metrics_clr = calculate_face_metrics(gt_labels, harness_scores, "ArcFace reference")
        utils.log_quality(metrics_clr, "ArcFace reference")

        utils.log_step(10.2, "Harness: Quality check")

        # Store measurements
        run_path.parent.mkdir(parents=True, exist_ok=True)
        utils.save_run(run_path, size)

    print(f"\nAll steps completed for face verification ({instance_name(size)})!")


if __name__ == "__main__":
    main()
