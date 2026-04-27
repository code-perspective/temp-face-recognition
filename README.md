# FHE Benchmarking Suite — Face Verification

This repository contains the harness for the face verification workload of the FHE benchmarking suite of [HomomorphicEncryption.org].

The `face-verification-harness` branch contains a reference submission under the `submission/` subdirectory (CryptoFace, based on Orion/CKKS).

Submitters clone this repository and replace the contents of `submission/` with their own implementation. The stage scripts must accept a single positional argument (instance size 0–3) and follow the file I/O contract described below.

## Prerequisites

### System dependencies

```console
sudo apt update
sudo apt install build-essential
sudo apt install python3-dev
sudo apt install golang-go
sudo apt install libgraphviz-dev
sudo apt install -y libgl1-mesa-glx
sudo apt install -y libglib2.0-0 libsm6 libxext6 libxrender-dev
```

### Python dependencies

```console
python3 -m venv bmenv
source bmenv/bin/activate
pip install -r requirements.txt
```

### Dataset (git-lfs)

The benchmark dataset (`datasets/face_dataset.npy`) is stored using git-lfs. Pull it after cloning:

```console
sudo apt-get install git-lfs
git lfs install
git lfs pull
```

### Submission dependencies (reference submission only)

Install Orion and other dependencies required by the CryptoFace reference submission:

```console
bash scripts/build_face_recognition.sh
pip install torch insightface opencv-python pyyaml
```

Set the checkpoint path in `submission/config.yml` before running.

## Running the benchmark

```console
python3 harness/run_submission.py -h
```

```
usage: run_submission.py [-h] [--num_runs NUM_RUNS] [--seed SEED]
                         [--clrtxt CLRTXT] [--batch_size BATCH_SIZE]
                         {0,1,2,3}

Run Face Verification FHE benchmark.

positional arguments:
  {0,1,2,3}            Instance size (0-single/1-small/2-medium/3-large)

options:
  --num_runs NUM_RUNS  Number of times to run stages 4-10 (default: 1)
  --seed SEED          Random seed for reproducible pair sampling
  --clrtxt CLRTXT      Set to 1 to force rerun of cleartext reference
  --batch_size INT     Override default batch size for the chosen instance
```

### Example: single-pair smoke test

```console
python3 harness/run_submission.py 0 --seed 42
```

### Example: small size, two runs

```console
python3 harness/run_submission.py 1 --seed 3 --num_runs 2
```

Results are written to `measurements/` as JSON files (`results-1.json`, `results-2.json`, …).

## Pipeline stages

The harness drives the following sequence. Stages 2, 3, and 5–9 invoke the submission's scripts via `utils.run_exe_or_python()`, which runs `submission/<stage>.py` if present, otherwise `submission/build/<stage>`.

| Stage | Script | Description |
|-------|--------|-------------|
| 0 | harness | Remove and re-create `io/<size>/` |
| 1 | harness | Validate `datasets/face_dataset.npy` |
| 2 | submission | `client_key_generation` — generate CKKS keys |
| 3 | submission | `server_preprocess_model` — server preprocessing stub |
| 4 | harness | `generate_input.py` — sample face pairs into `datasets/<size>/intermediate/` |
| 5 | submission | `client_preprocess_input` — face alignment and patch extraction |
| 6 | submission | `client_encode_encrypt_input` — encode and encrypt patches |
| 7 | submission | `server_encrypted_compute` — encrypted face verification |
| 8 | submission | `client_decrypt_decode` — decrypt similarity scores |
| 9 | submission | `client_postprocess` — optional postprocessing |
| 10 | harness | ArcFace cleartext reference + EER/TAR@FAR metrics |

Stages 4–10 repeat for each `--num_runs` iteration.

## File I/O contract

| Path | Written by | Read by |
|------|-----------|---------|
| `datasets/face_dataset.npy` | pre-provided | harness stage 1, 4 |
| `datasets/<size>/intermediate/test_pairs.npz` | harness stage 4 | submission stage 5 |
| `datasets/<size>/intermediate/test_labels.txt` | harness stage 4 | harness stage 10 |
| `io/<size>/public_keys/keys.h5` | submission stage 2 | submission stages 6, 7, 8 |
| `io/<size>/public_keys/fit_sample.npy` | submission stage 2 | submission stage 7 |
| `io/<size>/public_keys/input_level.txt` | submission stage 3 | submission stage 6 |
| `io/<size>/intermediate/*.npy` | submission stage 5 | submission stage 6 |
| `io/<size>/ciphertexts_upload/*.bin` | submission stage 6 | submission stage 7 |
| `io/<size>/ciphertexts_download/*.bin` | submission stage 7 | submission stage 8 |
| `io/<size>/encrypted_model_predictions.txt` | submission stage 8 | harness stage 10 |
| `io/<size>/harness_model_predictions.txt` | harness stage 10 | harness stage 10 |

## Directory structure

```
├── README.md
├── LICENSE.md
├── harness/
│   ├── run_submission.py       # Main harness orchestrator
│   ├── params.py               # InstanceParams and batch sizes
│   ├── utils.py                # Logging, timing, run_exe_or_python
│   ├── metrics.py              # EER and TAR@FAR calculation
│   ├── generate_dataset.py     # Validate pre-provided dataset
│   ├── generate_input.py       # Sample face pairs per run
│   ├── cleartext_impl.py       # ArcFace plaintext reference
│   └── verify_result.py        # Standalone metric verification
├── datasets/
│   ├── face_dataset.npy        # Pre-provided benchmark dataset (git-lfs)
│   └── face_dataset_labels.txt # Ground-truth labels (0=different, 1=same)
├── submission/                 # Reference submission (CryptoFace)
│   ├── config.yml
│   ├── common.py
│   ├── client_key_generation.py
│   ├── server_preprocess_model.py
│   ├── client_preprocess_input.py
│   ├── client_encode_encrypt_input.py
│   ├── server_encrypted_compute.py
│   ├── client_decrypt_decode.py
│   ├── client_postprocess.py
│   ├── models/
│   ├── utils/
│   ├── checkpoints/            # Place backbone-64x64.ckpt here
│   └── orion_configs/
├── scripts/
│   └── build_face_recognition.sh   # Install Orion dependency
├── io/                         # Client↔server communication (generated)
└── measurements/               # Per-run JSON results (generated)
```
