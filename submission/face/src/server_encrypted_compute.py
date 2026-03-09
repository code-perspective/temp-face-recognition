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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    load_submission_config, get_face_params,
    build_pipeline_load
)

# Worker globals — set in main process, inherited by forked workers
_pipeline      = None
_embedding_dim = None
_n_patches     = None
_n_pairs       = None
_n_workers     = None
_upload_dir    = None


def _process_pair(pair_idx):
    """
    Process one complete verification pair: both images → inner product → score.
    Runs in a forked worker; inherits compiled _pipeline.
    Each worker exits after 1 pair (maxtasksperchild=1), releasing all Go memory.

    Returns (pair_idx, serialized_score_bytes)
    """
    from orion.backend.python.tensors import CipherTensor
    from orion.core import scheme as _scheme
    from utils.he_operations import tree_reduce_add, compute_inner_product_encrypted

    # Stagger the first wave of workers to spread peak memory load.
    # Worker k sleeps k*[10,20]s; subsequent waves are naturally offset
    # because predecessors complete at different times.
    stagger = (pair_idx % _n_workers) * random.uniform(10, 20)
    if stagger > 0:
        print(f"[server] pair {pair_idx+1}/{_n_pairs}: staggering {stagger:.0f}s...", flush=True)
        time.sleep(stagger)

    t0 = time.time()
    print(f"[server] pair {pair_idx+1}/{_n_pairs}: starting...", flush=True)

    # Load and deserialize ciphertexts for both images (all branches)
    ctxts = [[], []]
    for j in range(2):
        for k in range(_n_patches):
            ctxt_dict = pickle.loads(
                (_upload_dir / f"p{pair_idx:04d}_i{j}_b{k}.bin").read_bytes()
            )
            ctxts[j].append(CipherTensor.deserialize(_scheme, ctxt_dict))

    # Backbone + linear for all branches of both images
    def _run_branches(ciphers):
        return [
            getattr(_pipeline, f"linear{k}")(getattr(_pipeline, f"backbone{k}")(c))
            for k, c in enumerate(ciphers)
        ]

    feats1 = _run_branches(ctxts[0])
    feats2 = _run_branches(ctxts[1])

    # Aggregate + normalize + inner product → scalar
    emb1  = _pipeline.normalization(tree_reduce_add(feats1))
    emb2  = _pipeline.normalization(tree_reduce_add(feats2))
    score = compute_inner_product_encrypted(emb1, emb2, _embedding_dim)

    result = pickle.dumps(score.serialize())

    del ctxts, feats1, feats2, emb1, emb2, score
    gc.collect()

    elapsed = time.time() - t0
    print(f"[server] pair {pair_idx+1}/{_n_pairs}: {elapsed:.1f}s", flush=True)

    return pair_idx, result


def main():
    if len(sys.argv) < 2:
        print("Usage: server_encrypted_compute <size>", flush=True)
        sys.exit(1)

    size = int(sys.argv[1])
    cfg    = load_submission_config()
    params = get_face_params(size)

    # Load compiled pipeline — safe to fork after this
    global _pipeline, _embedding_dim, _n_patches, _n_pairs, _n_workers, _upload_dir
    pipeline, _, embedding_dim, n_patches = build_pipeline_load(cfg, params)  # ~25 min compile
    _pipeline      = pipeline
    _embedding_dim = embedding_dim
    _n_patches     = n_patches
    _upload_dir    = params.iodir() / "ciphertexts_upload"

    download_dir = params.iodir() / "ciphertexts_download"
    download_dir.mkdir(parents=True, exist_ok=True)

    # Count pairs; skip any already completed
    n_pairs   = len(list(_upload_dir.glob("p????_i0_b0.bin")))
    _n_pairs  = n_pairs
    pending   = [i for i in range(n_pairs)
                 if not (download_dir / f"p{i:04d}_score.bin").exists()]
    n_workers = min(cfg.get("n_workers", 4), len(pending))
    _n_workers = n_workers

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
    finally:
        pool.terminate()
        pool.join()

    gc.collect()
    print(f"[server_encrypted_compute] Done. {n_pairs} scores → {download_dir}", flush=True)


if __name__ == "__main__":
    main()
