import os
import sys
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import h5py
import hydra
import numpy as np
import torch
import trimesh
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from plyfile import PlyData, PlyElement
from pytorch3d.ops import sample_farthest_points
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
            "source": "dfaust",
            "dfaust_root": str(cfg.dfaust_root),
            "work_dir": str(cfg.work_dir),
            "subjects": list(_cfg_get(cfg, "subjects", [])),
            "splits": OmegaConf.to_container(_cfg_get(cfg, "splits", {}), resolve=True),
            "point_clouds": OmegaConf.to_container(_cfg_get(cfg, "point_clouds", {}), resolve=True),
        }
    }
    meta_path = os.path.join(run_dir, "dataset_meta.yaml")
    OmegaConf.save(OmegaConf.create(meta), meta_path)
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


def _extract_frame_number(filename: str) -> int:
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split(".")
    return int(parts[-1])

def _get_sequences_for_subject(reg_file: h5py.File, subject_id: str) -> List[str]:
    return [seq for seq in reg_file.keys() if seq.startswith(subject_id + "_")]

def _sequence_to_mode(subject_split: Dict, sequence_name: str) -> Optional[str]:
    if sequence_name in subject_split.get("train", []):
        return "train"
    if sequence_name in subject_split.get("test", []):
        return "test"
    if sequence_name in subject_split.get("val", []):
        return "val"
    return None

def _apply_vertex_mask_to_mesh(
    vertices: np.ndarray,
    faces: Optional[np.ndarray],
    mask: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    mask = np.asarray(mask, dtype=bool)
    vertices_new = vertices[mask]

    if faces is None:
        return vertices_new, None

    old_to_new = np.full((mask.shape[0],), -1, dtype=np.int64)
    keep_idx = np.where(mask)[0]
    old_to_new[keep_idx] = np.arange(keep_idx.shape[0], dtype=np.int64)

    face_keep = np.all(mask[faces], axis=1)
    faces_new = faces[face_keep]
    if faces_new.size == 0:
        return vertices_new, None

    faces_new = old_to_new[faces_new]
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

def _sample_points_via_surface_and_fps(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_points: int,
    surface_samples: int,
    device: torch.device,
) -> np.ndarray:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    pts = trimesh.sample.sample_surface(mesh, int(surface_samples))[0].astype(np.float32)
    pts_tensor = torch.from_numpy(pts).unsqueeze(0).to(device=device, dtype=torch.float32)
    _, fps_idx = sample_farthest_points(pts_tensor, K=int(num_points))
    fps_idx_np = fps_idx[0].detach().cpu().numpy()

    return pts[fps_idx_np].astype(np.float32)

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


def _write_pointcloud_ply(path: str, points_xyz: np.ndarray, sample_id: int):
    points_xyz = np.asarray(points_xyz, dtype=np.float32)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    vertex_array = np.empty(
        points_xyz.shape[0],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("id", "i4")],
    )
    vertex_array["x"] = points_xyz[:, 0]
    vertex_array["y"] = points_xyz[:, 1]
    vertex_array["z"] = points_xyz[:, 2]
    vertex_array["id"] = int(sample_id)

    PlyData([PlyElement.describe(vertex_array, "vertex")]).write(path)

def _subject_output_dir(out_base: str, mode: str, gender: str, subject_id: str, sequence_name: str) -> str:
    return os.path.join(out_base, mode, gender, subject_id, sequence_name)

def _infer_subject_gender(
    reg_m_file: h5py.File,
    reg_f_file: h5py.File,
    subject_id: str,
) -> Tuple[List[str], Optional[str], Optional[h5py.File]]:
    seqs_m = _get_sequences_for_subject(reg_m_file, subject_id)
    seqs_f = _get_sequences_for_subject(reg_f_file, subject_id)

    if seqs_m and seqs_f:
        raise ValueError(f"Subject {subject_id} appears in both male and female registration files.")

    if seqs_m:
        return seqs_m, "male", reg_m_file
    if seqs_f:
        return seqs_f, "female", reg_f_file
    return [], None, None

def process_dfaust(cfg: DictConfig, logger):
    set_seed(int(_cfg_get(cfg, "dump.rnd_seed", 100)))

    dfaust_root = str(cfg.dfaust_root)
    work_dir = str(cfg.work_dir)
    os.makedirs(work_dir, exist_ok=True)

    device = _resolve_device(str(_cfg_get(cfg, "dump.device", "auto")))
    scan_points = int(_cfg_get(cfg, "point_clouds.scan_points", 4096))
    frame_interval = int(_cfg_get(cfg, "point_clouds.frame_selection.frame_interval", 5))
    middle_crop = float(_cfg_get(cfg, "point_clouds.frame_selection.middle_crop", 0.5))

    apply_scan_masks = bool(_cfg_get(cfg, "point_clouds.filtering.apply_scan_masks", False))
    dist_threshold_m = float(_cfg_get(cfg, "point_clouds.filtering.dist_threshold_m", 0.1))
    surface_samples = int(_cfg_get(cfg, "point_clouds.sampling.surface_samples", 50000))

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

    reg_m_path = os.path.join(dfaust_root, str(_cfg_get(cfg, "registrations.male", "registrations_m.hdf5")))
    reg_f_path = os.path.join(dfaust_root, str(_cfg_get(cfg, "registrations.female", "registrations_f.hdf5")))
    scans_root = os.path.join(dfaust_root, str(_cfg_get(cfg, "scans_dir", "scans")))
    masks_root = os.path.join(dfaust_root, str(_cfg_get(cfg, "masks_dir", "masks")))

    subjects = [str(x) for x in _cfg_get(cfg, "subjects", [])]
    exclude_frames = set(str(x) for x in _cfg_get(cfg, "exclude_frames", []))
    dataset_split = OmegaConf.to_container(_cfg_get(cfg, "dataset_split", {}), resolve=True) or {}

    out_base = os.path.join(work_dir, "point_clouds")
    enabled_splits_cfg = OmegaConf.to_container(_cfg_get(cfg, "splits", {}), resolve=True) or {}
    enabled_splits = {
        split_name
        for split_name, sc in enabled_splits_cfg.items()
        if bool((sc or {}).get("enabled", False))
    }

    global_sample_id = 0
    split_written_counts = {k: 0 for k in ["train", "val", "test"]}

    with h5py.File(reg_m_path, "r") as reg_m_file, h5py.File(reg_f_path, "r") as reg_f_file:
        for subject_id in subjects:
            subject_split = dataset_split.get(subject_id, {})
            if not subject_split:
                logger(f"[{subject_id}] no split entry found; skipping subject.")
                continue

            sequences, gender, reg_file = _infer_subject_gender(reg_m_file, reg_f_file, subject_id)
            if not sequences or gender is None or reg_file is None:
                logger(f"[{subject_id}] no registration sequences found; skipping subject.")
                continue

            for seq in sequences:
                seq_parts = seq.split("_")
                sequence_name = "_".join(seq_parts[1:])
                mode = _sequence_to_mode(subject_split, sequence_name)

                if mode is None:
                    logger(f"[{subject_id}] sequence {sequence_name} not assigned to train/val/test; skipping.")
                    continue

                if mode not in enabled_splits:
                    continue

                sequence_scans_path = os.path.join(scans_root, subject_id, sequence_name)
                if not os.path.isdir(sequence_scans_path):
                    logger(f"[{mode}] missing scans folder for {subject_id}/{sequence_name}; skipping.")
                    continue

                seq_out_dir = _subject_output_dir(
                    out_base=out_base,
                    mode=mode,
                    gender=gender,
                    subject_id=subject_id,
                    sequence_name=sequence_name,
                )
                os.makedirs(seq_out_dir, exist_ok=True)

                scan_files = [f for f in os.listdir(sequence_scans_path) if f.endswith(".ply")]
                scan_frames = sorted(
                    [(_extract_frame_number(f), f) for f in scan_files],
                    key=lambda x: x[0],
                )

                if len(scan_frames) == 0:
                    logger(f"[{mode}] no scan .ply files found for {subject_id}/{sequence_name}; skipping.")
                    continue

                total_frames = len(scan_frames)
                start_idx = int(((1.0 - middle_crop) / 2.0) * total_frames)
                end_idx = int(((1.0 + middle_crop) / 2.0) * total_frames)
                selected_indices = list(range(start_idx, end_idx, frame_interval))

                reg_vertices = reg_file[seq][:]
                n_reg_frames = reg_vertices.shape[2]
                selected_indices = [i for i in selected_indices if i < n_reg_frames]

                if n_reg_frames != len(scan_frames):
                    logger(
                        f"[{mode}] warning for {subject_id}/{sequence_name}: "
                        f"registration frames={n_reg_frames} vs scan frames={len(scan_frames)}"
                    )

                masks_file = None
                if apply_scan_masks:
                    masks_path = os.path.join(masks_root, f"{subject_id}_{sequence_name}.hdf5")
                    if os.path.exists(masks_path):
                        masks_file = h5py.File(masks_path, "r")
                    else:
                        logger(
                            f"[{mode}] mask file missing for {subject_id}/{sequence_name}: "
                            f"{os.path.basename(masks_path)}; continuing without mask."
                        )

                for idx in selected_indices:
                    frame_num, scan_file = scan_frames[idx]
                    frame_identifier = f"{subject_id}_{sequence_name}_{frame_num}"

                    if frame_identifier in exclude_frames:
                        logger(f"[{mode}] excluded frame {frame_identifier}")
                        continue

                    scan_path = os.path.join(sequence_scans_path, scan_file)
                    try:
                        mesh = trimesh.load(scan_path, process=False)
                    except Exception as e:
                        logger(f"[{mode}] failed loading scan {scan_path}: {e}; skipping.")
                        continue

                    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
                        logger(f"[{mode}] invalid mesh {scan_path}; skipping.")
                        continue

                    scan_vertices = np.asarray(mesh.vertices, dtype=np.float32)
                    scan_faces = np.asarray(mesh.faces, dtype=np.int64)

                    if scan_vertices.shape[0] < scan_points:
                        logger(
                            f"[{mode}] skipping {frame_identifier}: insufficient input vertices "
                            f"({scan_vertices.shape[0]} < {scan_points})"
                        )
                        continue

                    gt_vertices = np.asarray(reg_vertices[:, :, idx], dtype=np.float32)

                    if masks_file is not None:
                        frame_key = f"{frame_num:06d}"
                        if frame_key in masks_file:
                            mask = masks_file[frame_key][:].astype(bool)
                            if len(mask) != len(scan_vertices):
                                logger(
                                    f"[{mode}] mask length mismatch for {frame_identifier}: "
                                    f"mask={len(mask)} vs scan_vertices={len(scan_vertices)}; skipping."
                                )
                                continue

                            scan_vertices, scan_faces = _apply_vertex_mask_to_mesh(scan_vertices, scan_faces, mask)
                            if scan_faces is None or scan_faces.shape[0] == 0:
                                logger(f"[{mode}] skipping {frame_identifier}: no valid faces after mask.")
                                continue
                        else:
                            logger(f"[{mode}] mask for frame {frame_key} not found in {subject_id}/{sequence_name}")

                    scan_vertices_f, scan_faces_f = _distance_filter_to_gt(
                        gt_vertices=gt_vertices,
                        scan_vertices=scan_vertices,
                        scan_faces=scan_faces,
                        threshold_m=dist_threshold_m,
                    )

                    if scan_vertices_f.shape[0] == 0:
                        logger(f"[{mode}] skipping {frame_identifier}: no vertices after distance filtering.")
                        continue

                    if scan_vertices_f.shape[0] < scan_points:
                        logger(
                            f"[{mode}] skipping {frame_identifier}: insufficient vertices after filtering "
                            f"({scan_vertices_f.shape[0]} < {scan_points})"
                        )
                        continue

                    if scan_faces_f is None or scan_faces_f.shape[0] == 0:
                        logger(f"[{mode}] skipping {frame_identifier}: no valid faces after filtering.")
                        continue

                    try:
                        scan_pc = _sample_points_via_surface_and_fps(
                            vertices=scan_vertices_f,
                            faces=scan_faces_f,
                            num_points=scan_points,
                            surface_samples=surface_samples,
                            device=device,
                        )
                    except Exception as e:
                        logger(f"[{mode}] sampling failed for {frame_identifier}: {e}; skipping.")
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

                    out_scan_path = os.path.join(seq_out_dir, f"{frame_num}_scan.ply")
                    out_gt_path = os.path.join(seq_out_dir, f"{frame_num}_gt.ply")

                    _write_pointcloud_ply(out_scan_path, points_xyz=scan_pc_c, sample_id=global_sample_id)
                    _write_pointcloud_ply(out_gt_path, points_xyz=gt_vertices_c, sample_id=global_sample_id)

                    global_sample_id += 1
                    split_written_counts[mode] += 1

                if masks_file is not None:
                    masks_file.close()

    for split_name in ["train", "val", "test"]:
        if split_name in enabled_splits:
            logger(f"[{split_name}] wrote {split_written_counts[split_name]} sample pairs.")

def prepare_dfaust(cfg: DictConfig, logger):
    work_dir = str(cfg.work_dir)
    os.makedirs(work_dir, exist_ok=True)

    _save_run_configs(cfg, logger)

    run_dir = HydraConfig.get().runtime.output_dir
    try:
        shutil.copy2(sys.argv[0], os.path.join(run_dir, os.path.basename(sys.argv[0])))
    except Exception:
        pass

    process_dfaust(cfg, logger=logger)

@hydra.main(version_base=None, config_path=".", config_name="config_dfaust")
def main(cfg: DictConfig):
    run_dir = HydraConfig.get().runtime.output_dir
    if _work_dir_has_existing_data(str(cfg.work_dir), run_dir):
        print(
            f"[DFAUST Preprocessing] Work directory already contains data; "
            f"skipping preprocessing: {cfg.work_dir}"
        )
        return

    starttime = datetime.now().replace(microsecond=0)
    log_name = datetime.strftime(starttime, "%Y%m%d_%H%M")

    logger = log2file(makepath(run_dir, f"{log_name}.log", isfile=True))
    logger(f"[{str(_cfg_get(cfg, 'expr_code', 'DFAUST Preprocessing'))}] Begin")
    logger(str(_cfg_get(cfg, "msg", "")))

    prepare_dfaust(cfg, logger=logger)

if __name__ == "__main__":
    main()