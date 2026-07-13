# Measurements

`harness/run_submission.py` writes one JSON measurement file per run into a
sub-directory named for the instance size: `single`, `small`, `medium`, or
`large`.

Running with `--num_runs <n>` produces `results-1.json` … `results-<n>.json` in
the corresponding sub-directory.

## Submitting

Before submitting, run each variant you intend to submit with `--num_runs 3` and
commit the resulting files to your fork:

```console
python3 harness/run_submission.py 0 --num_runs 3   # single
python3 harness/run_submission.py 1 --num_runs 3   # small
python3 harness/run_submission.py 2 --num_runs 3   # medium
python3 harness/run_submission.py 3 --num_runs 3   # large
```

The average of the three runs is the number reported for your submission.

## File schema

Each `results-*.json` follows the FHE-benchmarking measurement schema:

- **`Timing`** — wall-clock latency per stage (e.g. `Key Generation`,
  `Encrypted model preprocessing`, `Input encryption`, `Encrypted computation`,
  `Result decryption`, …), plus `Total`.
- **`Bandwidth`** — sizes of `Public and evaluation keys`, `Encrypted input`,
  `Encrypted results`.
- **`Quality`** (sizes > single) — `Encrypted model quality` and
  `Harness model quality`, each reporting `eer`, `tar_at_far_1pct`,
  `tar_at_far_01pct`.
- **`Acceptance`** (sizes > single) — whether the encrypted model's
  `TAR@FAR=0.1%` meets the acceptance threshold.
- **`Server Reported`** — the server's own timing, isolating the pure
  `Encrypted computation` from pipeline/key-loading setup.
- **`additional_measurements`** — fine-grained server breakdown
  (e.g. `Pipeline load and key setup`, `Encrypted compute (wall clock)`).
