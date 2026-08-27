import os
import sys
import glob
import random
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
import hydra
import numpy as np
import torch
import trimesh
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from scipy.spatial import cKDTree
from human_body_prior.tools.omni_tools import makepath, log2file

def set_seed(seed: int, deterministic: bool = False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    try:
        torch.use_deterministic_algorithms(bool(deterministic))
    except Exception:
        pass

def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _cfg_get(cfg: DictConfig, path: str, default=None):
    cur = cfg
    for k in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, DictConfig):
            if k not in cur:
                return default
            cur = cur[k]
        elif isinstance(cur, dict):
            cur = cur.get(k, default)
        else:
            try:
                cur = getattr(cur, k)
            except Exception:
                return default
    return cur

def _save_run_configs(cfg: DictConfig, logger):
    run_dir = HydraConfig.get().runtime.output_dir

    resolved_path = os.path.join(run_dir, "preprocess_resolved.yaml")
    OmegaConf.save(OmegaConf.to_container(cfg, resolve=True), resolved_path)
    logger(f"Saved resolved config to {resolved_path}")

    meta = {
        "dataset": {
            "source": "faust",
            "faust_root": str(cfg.faust_root),
            "work_dir": str(cfg.work_dir),
            "splits": OmegaConf.to_container(cfg.splits, resolve=True) if "splits" in cfg else {},
            "point_clouds": OmegaConf.to_container(cfg.point_clouds, resolve=True) if "point_clouds" in cfg else {},
        }
    }
    meta_path = os.path.join(run_dir, "dataset_meta.yaml")
    OmegaConf.save(meta, meta_path)
    logger(f"Saved dataset metadata to {meta_path}")


def _work_dir_has_existing_data(work_dir: str, current_run_dir: str) -> bool:
    """Ignore directories Hydra created for this invocation, but nothing else."""
    work_path = Path(work_dir).expanduser().resolve()
    run_path = Path(current_run_dir).expanduser().resolve()

    if not work_path.exists():
        return False

    for path in work_path.rglob("*"):
        resolved = path.resolve()
        is_current_run_ancestor = resolved in run_path.parents
        is_in_current_run = resolved == run_path or run_path in resolved.parents
        if not is_current_run_ancestor and not is_in_current_run:
            return True
    return False


def _write_pointcloud_trimesh(path: str, points_xyz: np.ndarray, sample_id: Optional[int] = None):
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if sample_id is None:
        trimesh.points.PointCloud(points_xyz).export(path)
        return

    # Add 'id' as a vertex attribute (trimesh supports custom vertex attributes)
    pc = trimesh.points.PointCloud(points_xyz)
    pc.vertices = points_xyz  # ensure float32
    pc.metadata = pc.metadata or {}
    pc.visual = pc.visual  # keep default
    pc.vertex_attributes = getattr(pc, "vertex_attributes", {})
    pc.vertex_attributes["id"] = np.full((points_xyz.shape[0],), int(sample_id), dtype=np.int32)
    pc.export(path)

def _load_mesh_vertices_faces(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    mesh = trimesh.load(path, process=False)
    if not hasattr(mesh, "vertices"):
        raise ValueError(f"Not a mesh/point cloud: {path}")
    v = np.asarray(mesh.vertices, dtype=np.float32)
    f = np.asarray(mesh.faces, dtype=np.int64) if hasattr(mesh, "faces") and mesh.faces is not None else None
    return v, f

def _apply_vertex_mask_to_mesh(vertices: np.ndarray, faces: Optional[np.ndarray], mask: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    mask = np.asarray(mask, dtype=bool)
    vertices_new = vertices[mask]

    if faces is None:
        return vertices_new, None

    old_to_new = np.full((mask.shape[0],), -1, dtype=np.int64)
    keep_idx = np.where(mask)[0]
    old_to_new[keep_idx] = np.arange(keep_idx.shape[0], dtype=np.int64)

    face_keep = np.all(mask[faces], axis=1)
    faces_new = faces[face_keep]
    faces_new = old_to_new[faces_new]
    if faces_new.size == 0:
        return vertices_new, None
    return vertices_new, faces_new

def _distance_filter_to_gt(
    gt_vertices: np.ndarray,
    scan_vertices: np.ndarray,
    scan_faces: Optional[np.ndarray],
    threshold_m: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    tree = cKDTree(gt_vertices)
    dists, _ = tree.query(scan_vertices)
    keep = dists <= float(threshold_m)

    if int(keep.sum()) == 0:
        return np.empty((0, 3), dtype=np.float32), None

    return _apply_vertex_mask_to_mesh(scan_vertices, scan_faces, keep)

def _sample_points_from_filtered_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_points: int,
    device: torch.device,
) -> np.ndarray:
    v = torch.tensor(vertices, dtype=torch.float32, device=device).unsqueeze(0)
    f = torch.tensor(faces, dtype=torch.int64, device=device).unsqueeze(0)
    mesh = Meshes(verts=v, faces=f)
    pts = sample_points_from_meshes(mesh, num_points)  # (1, P, 3)
    return pts[0].detach().cpu().numpy().astype(np.float32)

def _compute_centering_offset(
    gt_vertices: np.ndarray,
    scan_pc: np.ndarray,
    centering_enabled: bool,
    centering_mode: str,
    subsample_mask: Optional[np.ndarray],
) -> np.ndarray:
    if not centering_enabled:
        return np.zeros((3,), dtype=np.float32)

    if centering_mode == "gt_mask":
        if subsample_mask is None:
            raise ValueError("centering.according_to_mask='gt_mask' but no subsample_mask was provided/loaded.")
        idx = np.asarray(subsample_mask).astype(np.int64)
        if idx.ndim != 1:
            raise ValueError("subsample_mask must be a 1D array of vertex indices.")
        if idx.size == 0:
            raise ValueError("subsample_mask is empty.")
        if idx.max() >= gt_vertices.shape[0] or idx.min() < 0:
            raise ValueError(
                f"subsample_mask indices out of range: got min={idx.min()}, max={idx.max()}, but gt has N={gt_vertices.shape[0]}"
            )
        center_pts = gt_vertices[idx]
    elif centering_mode == "gt":
        center_pts = gt_vertices
    elif centering_mode == "scan":
        center_pts = scan_pc
    else:
        raise ValueError("centering.according_to_mask must be one of: 'gt_mask', 'gt', 'scan'")

    return center_pts.mean(axis=0).astype(np.float32)

def process_faust(cfg: DictConfig, logger):
    set_seed(int(_cfg_get(cfg, "dump.rnd_seed", 100)))

    faust_root = str(cfg.faust_root)
    work_dir = str(cfg.work_dir)
    os.makedirs(work_dir, exist_ok=True)

    scan_points = int(_cfg_get(cfg, "point_clouds.scan_points", 4096))
    device = _resolve_device(str(_cfg_get(cfg, "dump.device", "auto")))

    apply_scan_masks = bool(_cfg_get(cfg, "point_clouds.filtering.apply_scan_masks", False))
    dist_threshold_m = float(_cfg_get(cfg, "point_clouds.filtering.dist_threshold_m", 0.1))

    reg_prefix_from = str(_cfg_get(cfg, "point_clouds.filtering.mask_name.reg_prefix_from", "tr_reg_"))
    reg_prefix_to = str(_cfg_get(cfg, "point_clouds.filtering.mask_name.reg_prefix_to", "tr_gt_"))
    reg_ext_from = str(_cfg_get(cfg, "point_clouds.filtering.mask_name.reg_ext_from", ".ply"))
    reg_ext_to = str(_cfg_get(cfg, "point_clouds.filtering.mask_name.reg_ext_to", ".txt"))

    centering_enabled = bool(_cfg_get(cfg, "point_clouds.centering.enabled", True))
    centering_mode = str(_cfg_get(cfg, "point_clouds.centering.according_to_mask", "gt_mask"))
    if centering_mode not in {"gt_mask", "gt", "scan"}:
        raise ValueError("point_clouds.centering.according_to_mask must be one of: 'gt_mask', 'gt', 'scan'")
    subsample_mask_path = _cfg_get(cfg, "point_clouds.centering.subsample_mask_path", None)

    subsample_mask = None
    if centering_enabled and centering_mode == "gt_mask":
        if subsample_mask_path is None:
            raise ValueError("Centering according to mask is enabled, but subsample_mask_path is not set.")
        subsample_mask = np.load(str(subsample_mask_path)).astype(np.int64)

    out_base = os.path.join(work_dir, "point_clouds")

    splits_cfg = _cfg_get(cfg, "splits", {})
    if isinstance(splits_cfg, DictConfig):
        splits_cfg = OmegaConf.to_container(splits_cfg, resolve=True) or {}

    for split_name, sc in (splits_cfg or {}).items():
        enabled = bool((sc or {}).get("enabled", False))
        if not enabled:
            logger(f"[{split_name}] disabled; skipping.")
            continue

        split_dir = (sc or {}).get("split_dir", None)
        if not split_dir:
            logger(f"[{split_name}] no split_dir set; skipping.")
            continue

        split_root = os.path.join(faust_root, str(split_dir))
        registrations_folder = os.path.join(split_root, "registrations")
        scans_folder = os.path.join(split_root, "scans")
        gt_vertices_folder = os.path.join(split_root, "ground_truth_vertices")

        if not os.path.isdir(registrations_folder):
            raise FileNotFoundError(f"Missing registrations folder: {registrations_folder}")
        if not os.path.isdir(scans_folder):
            raise FileNotFoundError(f"Missing scans folder: {scans_folder}")
        if apply_scan_masks and (not os.path.isdir(gt_vertices_folder)):
            warnings.warn(f"apply_scan_masks=true but ground_truth_vertices folder missing: {gt_vertices_folder}")

        out_scans = os.path.join(out_base, split_name, "scans")
        out_gt = os.path.join(out_base, split_name, "gt")
        os.makedirs(out_scans, exist_ok=True)
        os.makedirs(out_gt, exist_ok=True)

        reg_files = sorted(glob.glob(os.path.join(registrations_folder, "*.ply")))
        if len(reg_files) == 0:
            logger(f"[{split_name}] no .ply files found in {registrations_folder}")
            continue

        logger(f"[{split_name}] found {len(reg_files)} registration meshes.")

        sample_id = 0
        for reg_path in reg_files:
            reg_file = os.path.basename(reg_path)
            scan_file = reg_file.replace("reg", "scan", 1)
            scan_path = os.path.join(scans_folder, scan_file)
            if not os.path.exists(scan_path):
                logger(f"[{split_name}] scan missing for {reg_file}: expected {scan_file}; skipping.")
                continue

            try:
                gt_vertices, _ = _load_mesh_vertices_faces(reg_path)  # GT are registration vertices
            except Exception as e:
                logger(f"[{split_name}] error loading registration {reg_file}: {e}; skipping.")
                continue

            try:
                scan_vertices, scan_faces = _load_mesh_vertices_faces(scan_path)
            except Exception as e:
                logger(f"[{split_name}] error loading scan {scan_file}: {e}; skipping.")
                continue

            if apply_scan_masks:
                mask_name = reg_file
                if reg_prefix_from and mask_name.startswith(reg_prefix_from):
                    mask_name = reg_prefix_to + mask_name[len(reg_prefix_from) :]
                else:
                    mask_name = mask_name.replace(reg_prefix_from, reg_prefix_to, 1)

                if reg_ext_from and mask_name.endswith(reg_ext_from):
                    mask_name = mask_name[: -len(reg_ext_from)] + reg_ext_to
                else:
                    mask_name = mask_name.replace(reg_ext_from, reg_ext_to)

                mask_path = os.path.join(gt_vertices_folder, mask_name)
                if os.path.exists(mask_path):
                    mask_data = np.loadtxt(mask_path).astype(np.float32)
                    if mask_data.shape[0] != scan_vertices.shape[0]:
                        logger(
                            f"[{split_name}] mask length mismatch for {reg_file}: "
                            f"mask={mask_data.shape[0]} vs scan_vertices={scan_vertices.shape[0]}; skipping."
                        )
                        continue
                    mask_bool = mask_data == 1
                    scan_vertices, scan_faces = _apply_vertex_mask_to_mesh(scan_vertices, scan_faces, mask_bool)
                else:
                    logger(f"[{split_name}] mask missing for {reg_file}: expected {mask_name}; continuing without mask.")

            scan_vertices_f, scan_faces_f = _distance_filter_to_gt(
                gt_vertices=gt_vertices,
                scan_vertices=scan_vertices,
                scan_faces=scan_faces,
                threshold_m=dist_threshold_m,
            )

            if scan_vertices_f.shape[0] < scan_points:
                logger(
                    f"[{split_name}] skipping {reg_file}: insufficient vertices after filtering "
                    f"({scan_vertices_f.shape[0]} < {scan_points})"
                )
                continue

            if scan_faces_f is None or scan_faces_f.shape[0] == 0:
                logger(f"[{split_name}] skipping {reg_file}: no valid faces after filtering")
                continue

            try:
                scan_pc = _sample_points_from_filtered_mesh(
                    vertices=scan_vertices_f,
                    faces=scan_faces_f,
                    num_points=scan_points,
                    device=device,
                )
            except Exception as e:
                logger(f"[{split_name}] sampling failed for {reg_file}: {e}; skipping.")
                continue

            offset = _compute_centering_offset(
                gt_vertices=gt_vertices,
                scan_pc=scan_pc,
                centering_enabled=centering_enabled,
                centering_mode=centering_mode,
                subsample_mask=subsample_mask,
            )
            gt_vertices_c = (gt_vertices - offset).astype(np.float32)
            scan_pc_c = (scan_pc - offset).astype(np.float32)

            base = os.path.splitext(reg_file)[0]
            out_gt_path = os.path.join(out_gt, f"{base}.ply")
            out_scan_path = os.path.join(out_scans, f"{base}.ply")

            _write_pointcloud_trimesh(out_gt_path, gt_vertices_c, sample_id=sample_id)
            _write_pointcloud_trimesh(out_scan_path, scan_pc_c, sample_id=sample_id)

            sample_id += 1

        logger(f"[{split_name}] wrote {sample_id} sample pairs.")

def prepare_faust(cfg: DictConfig, logger):
    work_dir = str(cfg.work_dir)
    os.makedirs(work_dir, exist_ok=True)

    _save_run_configs(cfg, logger)

    run_dir = HydraConfig.get().runtime.output_dir
    try:
        shutil.copy2(sys.argv[0], os.path.join(run_dir, os.path.basename(sys.argv[0])))
    except Exception:
        pass

    process_faust(cfg, logger=logger)

@hydra.main(version_base=None, config_path=".", config_name="config_faust")
def main(cfg: DictConfig):
    run_dir = HydraConfig.get().runtime.output_dir
    if _work_dir_has_existing_data(str(cfg.work_dir), run_dir):
        print(
            f"[FAUST Preprocessing] Work directory already contains data; "
            f"skipping preprocessing: {cfg.work_dir}"
        )
        return

    starttime = datetime.now().replace(microsecond=0)
    log_name = datetime.strftime(starttime, "%Y%m%d_%H%M")

    logger = log2file(makepath(run_dir, f"{log_name}.log", isfile=True))
    logger(f"[{str(_cfg_get(cfg, 'expr_code', 'FAUST Preprocessing'))}] Begin")
    logger(str(_cfg_get(cfg, "msg", "")))

    prepare_faust(cfg, logger=logger)

if __name__ == "__main__":
    main()