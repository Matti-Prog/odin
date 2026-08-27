import os
import sys
import glob
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional
import hydra
import numpy as np
import trimesh
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from human_body_prior.tools.omni_tools import makepath, log2file

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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
            "source": "shrec19",
            "shrec_root": str(cfg.shrec_root),
            "work_dir": str(cfg.work_dir),
            "splits": OmegaConf.to_container(cfg.splits, resolve=True) if "splits" in cfg else {},
            "point_clouds": OmegaConf.to_container(cfg.point_clouds, resolve=True) if "point_clouds" in cfg else {},
            "save_processed_meshes": bool(_cfg_get(cfg, "save_processed_meshes", False)),
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


def _axis_to_index(axis: str) -> int:
    a = axis.lower().strip()
    if a == "x":
        return 0
    if a == "y":
        return 1
    if a == "z":
        return 2
    raise ValueError(f"Invalid height_axis='{axis}'. Expected one of: x,y,z")

def _write_pointcloud_trimesh(path: str, points_xyz: np.ndarray):
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    trimesh.points.PointCloud(points_xyz).export(path)

def _write_mesh_trimesh(path: str, mesh: trimesh.Trimesh):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mesh.export(path)

def _sample_points_from_mesh_trimesh(mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Expected trimesh.Trimesh")
    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError("Mesh has no faces; cannot sample surface points.")
    pts, _ = trimesh.sample.sample_surface(mesh, num_points)
    return np.asarray(pts, dtype=np.float32)

def _compute_center(points: np.ndarray, enabled: bool) -> Optional[np.ndarray]:
    if not enabled:
        return None
    return points.mean(axis=0, keepdims=True).astype(np.float32)

def _compute_scale_factor(points: np.ndarray, cfg: DictConfig) -> float:
    s1 = bool(_cfg_get(cfg, "point_clouds.scaling.scale_to_target_height.enabled", False))
    s2 = bool(_cfg_get(cfg, "point_clouds.scaling.fixed_scale_factor.enabled", False))
    if int(s1) + int(s2) > 1:
        raise ValueError("Scaling modes are mutually exclusive. Enable only one of: scale_to_target_height, fixed_scale_factor.")

    if s1:
        target = float(_cfg_get(cfg, "point_clouds.scaling.scale_to_target_height.target_height", 1.8))
        axis = str(_cfg_get(cfg, "point_clouds.scaling.scale_to_target_height.height_axis", "y"))
        eps = float(_cfg_get(cfg, "point_clouds.scaling.scale_to_target_height.eps", 1e-8))
        ax = _axis_to_index(axis)
        h = float(points[:, ax].max() - points[:, ax].min())
        return float(target / max(h, eps))

    if s2:
        return float(_cfg_get(cfg, "point_clouds.scaling.fixed_scale_factor.factor", 1.0))

    return 1.0

def _apply_transform(points: np.ndarray, center: Optional[np.ndarray], scale: float) -> np.ndarray:
    out = np.asarray(points, dtype=np.float32)
    if center is not None:
        out = out - center
    out = out * scale
    return out.astype(np.float32)

def process_shrec19(cfg: DictConfig, logger):
    set_seed(int(_cfg_get(cfg, "dump.rnd_seed", 100)))

    shrec_root = str(cfg.shrec_root)
    work_dir = str(cfg.work_dir)
    os.makedirs(work_dir, exist_ok=True)

    off_dir = os.path.join(shrec_root, "off")
    if not os.path.isdir(off_dir):
        raise FileNotFoundError(f"Missing OFF folder: {off_dir}")

    scan_points = int(_cfg_get(cfg, "point_clouds.scan_points", 4096))
    center_enabled = bool(_cfg_get(cfg, "point_clouds.centering.enabled", True))
    save_processed_meshes = bool(_cfg_get(cfg, "save_processed_meshes", False))

    # output structure:
    #   work_dir/point_clouds/train/scans/*.ply
    #   work_dir/processed_meshes/train/scans/*.ply  (optional)
    out_scans = os.path.join(work_dir, "point_clouds", "train", "scans")
    os.makedirs(out_scans, exist_ok=True)

    out_meshes = os.path.join(work_dir, "processed_meshes", "train", "scans")
    if save_processed_meshes:
        os.makedirs(out_meshes, exist_ok=True)

    off_files = sorted(glob.glob(os.path.join(off_dir, "*.off")))
    if len(off_files) == 0:
        raise FileNotFoundError(f"No .off files found under: {off_dir}")

    logger(f"[shrec19] found {len(off_files)} OFF meshes.")

    written = 0
    skipped = 0
    written_meshes = 0

    for off_path in off_files:
        base = os.path.splitext(os.path.basename(off_path))[0]
        out_path = os.path.join(out_scans, f"{base}.ply")
        out_mesh_path = os.path.join(out_meshes, f"{base}.ply")

        try:
            mesh = trimesh.load(off_path, file_type="off", process=False)
        except Exception as e:
            logger(f"[shrec19] failed loading {off_path}: {e}")
            skipped += 1
            continue

        if not isinstance(mesh, trimesh.Trimesh):
            try:
                mesh = trimesh.util.concatenate(tuple(mesh.dump()))
            except Exception:
                logger(f"[shrec19] not a mesh: {off_path}")
                skipped += 1
                continue

        try:
            pts = _sample_points_from_mesh_trimesh(mesh, scan_points)
        except Exception as e:
            logger(f"[shrec19] sampling failed for {off_path}: {e}")
            skipped += 1
            continue

        center = _compute_center(pts, center_enabled)
        pts_centered = _apply_transform(pts, center, 1.0)
        scale = _compute_scale_factor(pts_centered, cfg)

        pts_processed = _apply_transform(pts, center, scale)
        _write_pointcloud_trimesh(out_path, pts_processed)
        written += 1

        if save_processed_meshes:
            try:
                mesh_processed = mesh.copy()
                verts = np.asarray(mesh_processed.vertices, dtype=np.float32)
                verts_processed = _apply_transform(verts, center, scale)
                mesh_processed.vertices = verts_processed
                _write_mesh_trimesh(out_mesh_path, mesh_processed)
                written_meshes += 1
            except Exception as e:
                logger(f"[shrec19] processed mesh save failed for {off_path}: {e}")

    logger(f"[shrec19] wrote {written} scans; skipped {skipped}.")
    if save_processed_meshes:
        logger(f"[shrec19] wrote {written_meshes} processed meshes.")

def prepare_shrec19(cfg: DictConfig, logger):
    work_dir = str(cfg.work_dir)
    os.makedirs(work_dir, exist_ok=True)

    _save_run_configs(cfg, logger)

    run_dir = HydraConfig.get().runtime.output_dir
    try:
        shutil.copy2(sys.argv[0], os.path.join(run_dir, os.path.basename(sys.argv[0])))
    except Exception:
        pass

    process_shrec19(cfg, logger=logger)

@hydra.main(version_base=None, config_path=".", config_name="config_shrec19")
def main(cfg: DictConfig):
    run_dir = HydraConfig.get().runtime.output_dir
    if _work_dir_has_existing_data(str(cfg.work_dir), run_dir):
        print(
            f"[SHREC19 Preprocessing] Work directory already contains data; "
            f"skipping preprocessing: {cfg.work_dir}"
        )
        return

    starttime = datetime.now().replace(microsecond=0)
    log_name = datetime.strftime(starttime, "%Y%m%d_%H%M")

    logger = log2file(makepath(run_dir, f"{log_name}.log", isfile=True))
    logger(f"[{str(_cfg_get(cfg, 'expr_code', 'SHREC19 Preprocessing'))}] Begin")
    logger(str(_cfg_get(cfg, "msg", "")))

    prepare_shrec19(cfg, logger=logger)

if __name__ == "__main__":
    main()