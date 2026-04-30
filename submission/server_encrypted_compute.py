#!/usr/bin/env python3
import os
os.environ.setdefault("GOGC", "off")           # workers exit after 1 pair — GC wasteful and causes CoW faults
os.environ.setdefault("GODEBUG", "madvdontneed=1")  # return freed Go pages to OS immediately

import sys
import gc
import pickle
import multiprocessing
import random
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    parse_stage_args,
    build_pipeline_load
)

def _auto_n_workers(gb_per_worker: int):
    """
    Return the number of parallel workers that fit in currently available RAM.
    Reads MemAvailable from /proc/meminfo (sampled after the pipeline is loaded,
    so the pipeline's footprint is already subtracted).
    Returns None if /proc/meminfo is unavailable.
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    available_gb = int(line.split()[1]) / (1024 ** 2)  # kB → GB
                    return max(1, int(available_gb // gb_per_worker))
    except Exception:
        return None


@dataclass
class _WorkerEnv:
    """State set in the main process and inherited by forked workers. Not modified after fork."""
    pipeline:      object   # compiled Orion pipeline
    embedding_dim: int
    n_patches:     int
    n_pairs:       int
    n_workers:     int
    upload_dir:    Path

# Set in main() before forking; inherited by workers via fork()
_env: _WorkerEnv = None


def _process_pair(pair_idx):
    """
    Process one complete verification pair: both images → inner product → score.
    Runs in a forked worker; inherits compiled pipeline via _env.
    Each worker exits after 1 pair (maxtasksperchild=1), releasing all Go memory.

    Returns (pair_idx, serialized_score_bytes)
    """
    from orion.backend.python.tensors import CipherTensor
    from orion.core import scheme as _scheme
    from utils.he_operations import tree_reduce_add, compute_inner_product_encrypted

    # Stagger the first wave of workers to spread peak memory load.
    # Worker k sleeps k*[10,20]s; subsequent waves are naturally offset
    # because predecessors complete at different times.
    stagger = (pair_idx % _env.n_workers) * random.uniform(10, 20)
    if stagger > 0:
        print(f"[server] pair {pair_idx+1}/{_env.n_pairs}: staggering {stagger:.0f}s...", flush=True)
        time.sleep(stagger)

    t0 = time.time()
    print(f"[server] pair {pair_idx+1}/{_env.n_pairs}: starting...", flush=True)

    # Load and deserialize ciphertexts for both images (all branches)
    ctxts = [[], []]
    for j in range(2):
        for k in range(_env.n_patches):
            ctxt_dict = pickle.loads(
                (_env.upload_dir / f"p{pair_idx:04d}_i{j}_b{k}.bin").read_bytes()
            )
            ctxts[j].append(CipherTensor.deserialize(_scheme, ctxt_dict))

    # Backbone + linear for all branches of both images
    def _run_branches(ciphers):
        return [
            getattr(_env.pipeline, f"linear{k}")(getattr(_env.pipeline, f"backbone{k}")(c))
            for k, c in enumerate(ciphers)
        ]

    feats1 = _run_branches(ctxts[0])
    feats2 = _run_branches(ctxts[1])

    # Aggregate + normalize + inner product → scalar
    emb1  = _env.pipeline.normalization(tree_reduce_add(feats1))
    emb2  = _env.pipeline.normalization(tree_reduce_add(feats2))
    score = compute_inner_product_encrypted(emb1, emb2, _env.embedding_dim)

    result = pickle.dumps(score.serialize())

    del ctxts, feats1, feats2, emb1, emb2, score
    gc.collect()

    elapsed = time.time() - t0
    print(f"[server] pair {pair_idx+1}/{_env.n_pairs}: {elapsed:.1f}s", flush=True)

    return pair_idx, result


def main():
    global _env
    size, cfg, params = parse_stage_args()

    # Load SK and compile pipeline in memory — safe to fork after this
    pipeline, _, embedding_dim, n_patches = build_pipeline_load(cfg, params)  # ~25 min compile

    upload_dir   = params.iodir() / "ciphertexts_upload"
    download_dir = params.iodir() / "ciphertexts_download"
    download_dir.mkdir(parents=True, exist_ok=True)

    # Count pairs; skip any already completed
    n_pairs = sum(1 for _ in upload_dir.glob("p????_i0_b0.bin"))
    pending = [i for i in range(n_pairs)
               if not (download_dir / f"p{i:04d}_score.bin").exists()]

    gb_per_worker = cfg["gb_per_worker"]
    auto_workers = _auto_n_workers(gb_per_worker)
    if auto_workers is not None:
        n_workers = min(auto_workers, len(pending))
        print(f"[server_encrypted_compute] RAM-based worker count: "
              f"{auto_workers} (at {gb_per_worker} GB each), "
              f"capped to {n_workers} by pending pairs", flush=True)
    else:
        n_workers = min(cfg.get("n_workers", 1), len(pending))
        print(f"[server_encrypted_compute] Could not read /proc/meminfo; "
              f"using n_workers={n_workers} from config", flush=True)

    _env = _WorkerEnv(
        pipeline=pipeline,
        embedding_dim=embedding_dim,
        n_patches=n_patches,
        n_pairs=n_pairs,
        n_workers=n_workers,
        upload_dir=upload_dir,
    )

    print(f"[server_encrypted_compute] {n_pairs} pairs total, "
          f"{len(pending)} pending, {n_pairs - len(pending)} already done, "
          f"forking {n_workers} workers (pair-level, maxtasksperchild=1)...", flush=True)

    if not pending:
        print("[server_encrypted_compute] Nothing to do.", flush=True)
        return

    # Fork N workers — maxtasksperchild=1 ensures full memory release after each pair
    ctx  = multiprocessing.get_context("fork")
    pool = ctx.Pool(processes=n_workers, maxtasksperchild=1)

    n_done = 0
    try:
        for pair_idx, result_bytes in pool.imap_unordered(
            _process_pair, pending, chunksize=1
        ):
            out_path = download_dir / f"p{pair_idx:04d}_score.bin"
            out_path.write_bytes(result_bytes)
            n_done += 1
            print(f"[server] {n_done}/{len(pending)} pending pairs saved", flush=True)
        pool.close()
    except Exception:
        pool.terminate()
        raise
    finally:
        pool.join()

    gc.collect()
    print(f"[server_encrypted_compute] Done. {_env.n_pairs} scores → {download_dir}", flush=True)


if __name__ == "__main__":
    main()
