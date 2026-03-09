import sys
import yaml
import time
import numpy as np
import torch
from pathlib import Path

from models.cryptoface_pcnn import CryptoFaceNet
from models.weight_loader import load_cryptoface_checkpoint
from models.pipeline import PerImagePipeline
from utils.preprocessing import extract_patches


def load_submission_config() -> dict:
    """
    Reads submission/face/config.yml relative to repo root.
    Repo root is Path(__file__).parents[3] (src/ → face/ → submission/ → repo_root/).
    Returns the 'cryptoface' sub-dict from the yaml.
    Relative paths (ckpt_path, orion_config) are resolved against repo_root.
    """
    repo_root = Path(__file__).resolve().parents[3]
    config_path = repo_root / "submission" / "face" / "config.yml"
    with open(config_path) as f:
        full_cfg = yaml.safe_load(f)
    cfg = full_cfg["cryptoface"]
    for key in ("ckpt_path", "orion_config"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str((repo_root / cfg[key]).resolve())
    return cfg


def get_face_params(size: int):
    """
    Returns InstanceParams(size, rootdir=repo_root).
    Adds <repo_root>/harness/ to sys.path so params.py is importable.
    """
    repo_root = Path(__file__).resolve().parents[3]
    harness_dir = str(repo_root / "harness")
    if harness_dir not in sys.path:
        sys.path.insert(0, harness_dir)
    from params import InstanceParams
    return InstanceParams(size, rootdir=repo_root)


def align_face(detector, img_chw_rgb: np.ndarray, output_size: int) -> np.ndarray | None:
    """
    Detect and align one face using InsightFace norm_crop (always at 112),
    then resize to output_size x output_size.

    Args:
        detector: InsightFace FaceAnalysis app (already prepare()d)
        img_chw_rgb: (3, H, W) uint8 RGB
        output_size: target crop size in pixels (e.g., 64)

    Returns:
        (output_size, output_size, 3) uint8 RGB, or None if no face detected
    """
    import cv2
    from insightface.utils.face_align import norm_crop
    img_hwc_bgr = img_chw_rgb.transpose(1, 2, 0)[:, :, ::-1]
    faces = detector.get(img_hwc_bgr)
    if not faces:
        return None
    if output_size > 112:
        aligned_bgr = norm_crop(img_hwc_bgr, faces[0].kps, image_size=output_size)
    else:
        aligned_bgr = norm_crop(img_hwc_bgr, faces[0].kps, image_size=112)
        if output_size != 112:
            aligned_bgr = cv2.resize(aligned_bgr, (output_size, output_size),
                                     interpolation=cv2.INTER_LINEAR)
    return aligned_bgr[:, :, ::-1].copy()  # return RGB


def _center_crop_resize(img_chw_rgb: np.ndarray, output_size: int) -> np.ndarray:
    """Fallback: center crop to square then resize to output_size x output_size."""
    import cv2
    hwc = img_chw_rgb.transpose(1, 2, 0)
    h, w = hwc.shape[:2]
    side = min(h, w)
    top  = (h - side) // 2
    left = (w - side) // 2
    cropped = hwc[top:top+side, left:left+side]
    return cv2.resize(cropped, (output_size, output_size), interpolation=cv2.INTER_LINEAR)


def to_tensor(img_hwc_rgb: np.ndarray) -> torch.Tensor:
    """uint8 HWC RGB -> float32 (1, 3, H, W) normalized to [-1, 1]."""
    t = torch.from_numpy(img_hwc_rgb).permute(2, 0, 1).float()
    t = t / 255.0
    t = (t - 0.5) / 0.5
    return t.unsqueeze(0)


def preprocess_one_image(detector, img_chw_uint8: np.ndarray, input_size: int) -> list:
    """
    Preprocess one image into a list of patch tensors for Orion/CryptoFace inference.

    Args:
        detector: InsightFace FaceAnalysis app (already prepare()d)
        img_chw_uint8: (3, H, W) uint8 RGB
        input_size: target aligned face size in pixels (e.g., 64)

    Returns:
        list of N patch tensors, each (1, 3, 32, 32) float32
    """
    aligned = align_face(detector, img_chw_uint8, input_size)
    if aligned is None:
        aligned = _center_crop_resize(img_chw_uint8, input_size)

    tensor = to_tensor(aligned)  # (1, 3, input_size, input_size) float32
    patches = extract_patches(tensor)  # list of N tensors, each (1, 3, 32, 32)
    return patches


def load_detector():
    """Load InsightFace FaceAnalysis for face detection and alignment."""
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1)
    return app


def init_orion_scheme(cfg: dict, params, io_mode: str) -> None:
    """
    Load orion config yaml, set io_mode and key/diag paths, call orion.init_scheme().

    Args:
        cfg:     submission config dict (from load_submission_config)
        params:  InstanceParams (provides iodir())
        io_mode: "save" (generates keys) or "load" (reads existing keys)
    """
    import orion
    keys_dir = params.iodir() / "public_keys"
    with open(cfg["orion_config"]) as f:
        config = yaml.safe_load(f)
    config["orion"]["io_mode"] = io_mode
    config["orion"]["diags_path"] = str((keys_dir / "diagonals.h5").resolve())
    config["orion"]["keys_path"]  = str((keys_dir / "keys.h5").resolve())
    orion.init_scheme(config)


def load_fit_patches(params) -> list:
    """
    Load the fit sample patches saved by client_key_generation.

    Loads params.iodir() / "public_keys" / "fit_sample.npy"
    Shape: (N, 1, 3, 32, 32) float32

    Returns list of N tensors, each (1, 3, 32, 32) float32.
    """
    fit_path = params.iodir() / "public_keys" / "fit_sample.npy"
    arr = np.load(fit_path)  # (N, 1, 3, 32, 32) float32
    return [torch.from_numpy(arr[k].copy()) for k in range(arr.shape[0])]


def build_pipeline_load(cfg: dict, params) -> tuple:
    """
    Load orion pipeline from saved keys/diagonals.

    Loads the SK from HDF5 (io_mode=load) then switches to io_mode=none before
    compile, so plaintext diagonals and rotation keys are recomputed entirely in
    Go memory. This avoids the ~30-min preload_all overhead while using the same
    SK as stage 6 ciphertexts.

    Returns: (pipeline, input_level, embedding_dim, n_patches)
    """
    import orion
    from orion.core import scheme as _scheme

    t0 = time.time()

    a, b, c = cfg["l2_poly_coeffs"]

    # Load model with submission-specified coefficients
    model = CryptoFaceNet(cfg["input_size"], l2_norm_coeffs=(a, b, c))
    load_cryptoface_checkpoint(model, cfg["ckpt_path"])
    for net in model.nets:
        net.init_orion_params()

    pipeline = PerImagePipeline(
        backbones=model.nets,
        linears=model.linear,
        normalization=model.normalization,
    )
    pipeline.eval()

    # Always load SK from HDF5 so it matches stage 6 ciphertexts.
    init_orion_scheme(cfg, params, "load")

    # Switch to in-memory mode: SK is already loaded above.
    # Compile now recomputes plaintext diagonals and rotation keys entirely
    # in Go memory — no HDF5 reads for diagonals, no preload_all needed.
    print("[common] switching to io_mode=none for compile", flush=True)
    _scheme.params.orion_params.io_mode = "none"
    _scheme.lt_evaluator.io_mode = "none"

    fit_patches = load_fit_patches(params)
    orion.fit(pipeline, fit_patches)
    input_level = orion.compile(pipeline)

    pipeline.he()

    n_patches = model.N
    embedding_dim = model.embedding_dim
    print(f"[common] build_pipeline_load done in {time.time()-t0:.1f}s  "
          f"input_level={input_level}  n_patches={n_patches}", flush=True)
    return pipeline, input_level, embedding_dim, n_patches
