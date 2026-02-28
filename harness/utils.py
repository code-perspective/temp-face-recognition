#!/usr/bin/env python3
"""
utils.py - Harness utilities for argument parsing, logging, and results saving.

Provides:
  - parse_submission_arguments(): CLI argument parsing
  - ensure_directories():         validate required repo subdirectories
  - build_submission():           build submission via its build_task.sh
  - log_step():                   print per-stage elapsed time
  - log_size():                   print and record directory sizes
  - log_quality():                record quality metrics (EER, TAR@FAR)
  - save_run():                   write per-run JSON results to measurements/
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

import sys
import subprocess
import argparse
import json
from datetime import datetime
from pathlib import Path
from params import InstanceParams, SINGLE, LARGE
from typing import Tuple

# Global variable to track the last timestamp
_last_timestamp: datetime = None
# Global variable to store measured times
_timestamps = {}
_timestampsStr = {}
# Global variable to store measured sizes
_bandwidth = {}
# Global variable to store model quality metrics
_model_quality = {}

def parse_submission_arguments(workload: str) -> Tuple[int, InstanceParams, int, int, int]:
    """
    Get the arguments of the submission. Populate arguments as needed for the workload.
    """
    # Parse arguments using argparse
    parser = argparse.ArgumentParser(description=workload)
    parser.add_argument('size', type=int, choices=range(SINGLE, LARGE+1),
                        help='Instance size (0-single/1-small/2-medium/3-large)')
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of times to run steps 4-9 (default: 1)')
    parser.add_argument('--seed', type=int,
                        help='Random seed for dataset and query generation')
    parser.add_argument('--clrtxt', type=int,
                        help='Specify with 1 if to rerun the cleartext computation')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Number of face pairs to sample (default: size-dependent)')

    args = parser.parse_args()
    size = args.size
    seed = args.seed
    num_runs = args.num_runs
    clrtxt = args.clrtxt

    # Use params.py to get instance parameters
    params = InstanceParams(size, batch_size=args.batch_size)
    return size, params, seed, num_runs, clrtxt

def ensure_directories(rootdir: Path):
    """ Check that the current directory has sub-directories
    'harness', 'scripts', and 'submission' """
    required_dirs = ['harness', 'scripts', 'submission']
    for dir_name in required_dirs:
        if not (rootdir / dir_name).exists():
            print(f"Error: Required directory '{dir_name}'",
                  f"not found in {rootdir}")
            sys.exit(1)

def build_submission(script_dir: Path):
    """
    Build the submission. Fetching dependencies and compiling is 
    delegated entirely to the submission's build_task.sh.
    """
    subprocess.run([script_dir / "build_task.sh", "./submission"], check=True)

def log_step(step_num: int, step_name: str, start: bool = False):
    """
    Print a timestamped completion message and record elapsed time for a stage.
    If start=True, records the start time without printing (used for step 0).
    """
    global _last_timestamp
    global _timestamps
    global _timestampsStr
    now = datetime.now()
    timestamp = now.strftime("%H:%M:%S")

    # Calculate elapsed time if this isn't the first call
    elapsed_str = ""
    elapsed_seconds = 0
    if _last_timestamp is not None:
        elapsed_seconds = (now - _last_timestamp).total_seconds()
        elapsed_str = f" (elapsed: {round(elapsed_seconds, 4)}s)"

    # Update the last timestamp for the next call
    _last_timestamp = now

    if (not start):
        print(f"{timestamp} [harness] {step_num}: {step_name} completed{elapsed_str}")
        _timestampsStr[step_name] = f"{round(elapsed_seconds, 4)}s"
        _timestamps[step_name] = elapsed_seconds

def log_size(path: Path, object_name: str, flag: bool = False, previous: int = 0):
    global _bandwidth
    
    # Check if the path exists before trying to calculate size
    if not path.exists():
        print(f"         [harness] Warning: {object_name} path does not exist: {path}")
        _bandwidth[object_name] = "0B"
        return 0
    
    size = int(subprocess.run(["du", "-sb", path], check=True,
                           capture_output=True, text=True).stdout.split()[0])
    if(flag):
        size -= previous
    
    print("         [harness]", object_name, "size:", human_readable_size(size))

    _bandwidth[object_name] = human_readable_size(size)
    return size

def human_readable_size(n: int) -> str:
    """Convert a byte count to a human-readable string (e.g. 1.4G, 358.8K)."""
    for unit in ["B","K","M","G","T"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}P"

def save_run(path: Path, size: int = 0):
    """
    Write per-run timing, bandwidth, and (for batch sizes > 0) quality metrics
    to a JSON file at the given path.
    """
    global _timestamps
    global _timestampsStr
    global _bandwidth
    global _model_quality

    if size == 0:
        json.dump({
            "total_latency_ms": round(sum(_timestamps.values()), 4),
            "per_stage": _timestampsStr,
            "bandwidth": _bandwidth,
        }, open(path,"w"), indent=2)
    else:
        json.dump({
            "total_latency_ms": round(sum(_timestamps.values()), 4),
            "per_stage": _timestampsStr,
            "bandwidth": _bandwidth,
            "model_quality" : _model_quality,
        }, open(path,"w"), indent=2)

    print("[total latency]", f"{round(sum(_timestamps.values()), 4)}s")

def log_quality(metrics: dict, tag: str):
    """Store quality metrics returned by calculate_face_metrics() in the global quality dict."""
    global _model_quality
    if not metrics:
        return
    _model_quality[tag] = {
        "eer":              metrics["eer"],
        "tar_at_far_1pct":  metrics["tar_far_1_percent"],
        "tar_at_far_01pct": metrics["tar_far_01_percent"],
    }
