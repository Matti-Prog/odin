# This script is based on the original AMASS processing code found in:
# https://github.com/nghorbani/amass

# We subsample "frames" (each yields SMPL params for a subject in a specific pose) from 4D AMASS sequences,
# saving them in 3 different formats (see stages I, II, III).
# Additionally, we can create pairs of ground-truth point clouds (SMPL vertices) 
# and virtual scan point clouds from the SMPL meshes.
# The behavior of this script is controlled via the config_amass.yaml file.

import os
import sys
import glob
import shutil
import json
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import hydra
import numpy as np
import tables as pytables
import torch
import torch.nn as nn
import trimesh
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import smplx
from human_body_prior.tools.omni_tools import makepath, log2file, copy2cpu as c2c
from human_body_prior.tools.rotation_tools import euler2em, em2euler, aa2matrot

# maps gender to numeric code used in SMPL-format datasets
gdr2num = {"male": -1, "neutral": 0, "female": 1}
gdr2num_rev = {v: k for k, v in gdr2num.items()}

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

def remove_Zrot(pose):
    noZ = em2euler(pose[:3].copy())
    noZ[2] = 0
    pose[:3] = euler2em(noZ).copy()
    return pose

def _write_pointcloud_trimesh(path: str, points_xyz: np.ndarray):
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    trimesh.points.PointCloud(points_xyz).export(path)

def segment_scan(gt_vertices, scan_vertices, vertex_to_id, chunk_size: int = 1024):
    B, N_g, _ = gt_vertices.shape
    mapping = torch.full((N_g,), -1, dtype=torch.long, device=gt_vertices.device)
    if vertex_to_id is not None:
        for i, lbl in vertex_to_id.items():
            mapping[int(i)] = int(lbl)

    idxs_all = []
    for b in range(B):
        gv = gt_vertices[b].unsqueeze(0)
        sv = scan_vertices[b]
        idxs_b = []
        for start in range(0, sv.shape[0], chunk_size):
            s = sv[start : start + chunk_size].unsqueeze(0)
            d = torch.cdist(s, gv)
            idxs_b.append(d.argmin(dim=-1).squeeze(0))
        idxs_all.append(torch.cat(idxs_b, dim=0))
    idxs = torch.stack(idxs_all, dim=0)
    return mapping[idxs]

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

def _split_is_active(cfg: DictConfig, split: str) -> bool:
    return bool(_cfg_get(cfg, f"splits.{split}", True))

def _pc_split_is_active(cfg: DictConfig, split: str) -> bool:
    return bool(_cfg_get(cfg, f"point_clouds.splits.{split}", False))

def _any_pc_split_active(cfg: DictConfig) -> bool:
    pc_splits = _cfg_get(cfg, "point_clouds.splits", {})
    if isinstance(pc_splits, DictConfig):
        pc_splits = OmegaConf.to_container(pc_splits, resolve=True) or {}
    if not isinstance(pc_splits, dict):
        return False
    return any(bool(v) for v in pc_splits.values())

def _build_scan_context(cfg: DictConfig, device: torch.device):
    seg_json_path = _cfg_get(cfg, "point_clouds.masks.smpl_vert_segmentation_json")
    subsample_mask_path = _cfg_get(cfg, "point_clouds.masks.subsample_mask_path")

    smpl_model_path = _cfg_get(cfg, "dump.smpl_model_path")
    model_type = _cfg_get(cfg, "dump.model_type", "smplh")
    num_betas = int(_cfg_get(cfg, "dump.num_betas", 16))
    batch_size_smpl = int(_cfg_get(cfg, "dump.batch_size_smpl", 16))

    part_to_vertex = json.load(open(seg_json_path))
    sorted_parts = sorted(part_to_vertex.keys())
    part_to_id = {part: idx for idx, part in enumerate(sorted_parts)}
    vertex_to_part = {vertex: part for part, vertices in part_to_vertex.items() for vertex in vertices}
    vertex_to_id = {int(vertex): part_to_id[part] for vertex, part in vertex_to_part.items()}

    subsample_mask = np.load(subsample_mask_path).astype(np.int64)
    subsample_mask_t = torch.tensor(subsample_mask, device=device, dtype=torch.long)

    rotmat = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32, device=device)

    gender_map = {-1: "male", 0: "neutral", 1: "female"}
    models = nn.ModuleDict()
    for code, name in gender_map.items():
        models[name] = smplx.create(
            model_path=smpl_model_path,
            model_type=model_type,
            gender=name,
            batch_size=batch_size_smpl,
            num_betas=num_betas,
            use_pca=False,
        ).to(device)

    smpl_faces = torch.from_numpy(models["male"].faces.astype(np.int64)).to(device)

    return {
        "vertex_to_id": vertex_to_id,
        "subsample_mask": subsample_mask_t,
        "rotmat": rotmat,
        "gender_map": gender_map,
        "models": models,
        "smpl_faces": smpl_faces,
    }

def dump_amass2pytorch(
    datasets: List[str],
    amass_dir: str,
    out_posepath: str,
    *,
    logger,
    work_dir: str,
    mode: str,  # "train"/"vald"/"test"
    cfg: DictConfig,
    scan_ctx: Optional[dict] = None,
):
    rnd_seed = int(_cfg_get(cfg, "dump.rnd_seed", 100))
    keep_rate = float(_cfg_get(cfg, "dump.keep_rate", 0.01))
    chunk_size = int(_cfg_get(cfg, "dump.chunk_size", 1024))
    scan_points = int(_cfg_get(cfg, "dump.scan_points", 4096))
    missing_parts_chunk_size = int(_cfg_get(cfg, "dump.missing_parts_chunk_size", 1024))

    device = _resolve_device(str(_cfg_get(cfg, "dump.device", "auto")))
    set_seed(rnd_seed)
    makepath(out_posepath, isfile=True)

    split_active = _split_is_active(cfg, mode)
    pc_split_active = _pc_split_is_active(cfg, mode)
    save_point_clouds = bool(split_active and pc_split_active)

    if (not split_active) and pc_split_active:
        warnings.warn(f"[{mode}] point_clouds.splits.{mode}=True but splits.{mode}=False. No point clouds saved.")

    aug_cfg = _cfg_get(cfg, f"point_clouds.augment.{mode}", {})
    if isinstance(aug_cfg, DictConfig):
        aug_cfg = OmegaConf.to_container(aug_cfg, resolve=True) or {}
    if not isinstance(aug_cfg, dict):
        aug_cfg = {}

    augment_limbs_prob = float(aug_cfg.get("augment_limbs_prob", 0.0)) if save_point_clouds else 0.0
    augment_partial_views_prob = float(aug_cfg.get("augment_partial_views_prob", 0.0)) if save_point_clouds else 0.0
    add_noise = bool(aug_cfg.get("add_noise", False)) if save_point_clouds else False
    noise_std = float(aug_cfg.get("noise_std", 0.0025)) if save_point_clouds else 0.0

    centering_mode = str(_cfg_get(cfg, "point_clouds.centering_according_to_mask", "gt_mask"))
    if centering_mode not in {"gt_mask", "gt", "scan"}:
        raise ValueError(
            "point_clouds.centering_according_to_mask must be one of: "
            "'gt_mask', 'gt', 'scan'"
        )

    if save_point_clouds and scan_ctx is None:
        scan_ctx = _build_scan_context(cfg, device)

    data_pose, data_betas, data_gender, data_trans = [], [], [], []
    data_dataset, data_subject, data_motion, data_frame = [], [], [], []

    for ds_name in datasets:
        patterns = ["*/*_poses.npz", "*/*_stageii.npz"]
        npz_fnames = [f for pat in patterns for f in glob.glob(os.path.join(amass_dir, ds_name, pat))]
        npz_fnames = sorted(npz_fnames)

        logger(f"randomly selecting data points from {ds_name}.")
        for npz_fname in tqdm(npz_fnames):
            try:
                cdata = np.load(npz_fname, allow_pickle=True)
            except Exception:
                logger(f"Could not read {npz_fname}! skipping..")
                continue

            N = len(cdata["poses"])
            ids = np.random.choice(
                list(range(int(0.1 * N), int(0.9 * N))),
                int(keep_rate * 0.8 * N),
                replace=False,
            )
            if len(ids) < 1:
                continue

            data_pose.extend(cdata["poses"][ids].astype(np.float32))
            data_trans.extend(cdata["trans"][ids].astype(np.float32))
            data_betas.extend(np.repeat(cdata["betas"][np.newaxis].astype(np.float32), len(ids), axis=0))
            data_gender.extend([gdr2num[str(cdata["gender"].astype(str))] for _ in ids])

            subj = os.path.basename(os.path.dirname(npz_fname))
            motion = os.path.basename(npz_fname).replace("_poses.npz", "").replace("_stageii.npz", "")
            data_dataset.extend([ds_name] * len(ids))
            data_subject.extend([subj] * len(ids))
            data_motion.extend([motion] * len(ids))
            data_frame.extend(ids.tolist())

            if not save_point_clouds:
                continue

            ctx = scan_ctx
            vertex_to_id = ctx["vertex_to_id"]
            subsample_mask = ctx["subsample_mask"]
            rotmat = ctx["rotmat"]
            gender_map = ctx["gender_map"]
            models = ctx["models"]
            smpl_faces = ctx["smpl_faces"]

            gender_str = str(cdata["gender"].astype(str))
            gender_code = gdr2num[gender_str]

            pa = torch.tensor(cdata["poses"][ids].astype(np.float32), device=device)
            betas = torch.tensor(
                np.repeat(cdata["betas"][np.newaxis].astype(np.float32), repeats=len(ids), axis=0),
                device=device,
            )
            trans = torch.tensor(cdata["trans"][ids].astype(np.float32), device=device)
            genders = torch.full((len(ids),), gender_code, device=device, dtype=torch.long)

            B = betas.shape[0]
            output_vertices = torch.empty(B, 6890, 3, device=device, dtype=torch.float32)

            go = pa[:, :3]
            bp = pa[:, 3:66]
            z45 = torch.zeros(B, 45, device=device)

            for code, name in gender_map.items():
                mask = genders == code
                if not mask.any():
                    continue
                m = models[name]
                b = int(mask.sum().item())
                verts = m(
                    betas=betas[mask],
                    global_orient=go[mask],
                    body_pose=bp[mask],
                    left_hand_pose=z45[:b],
                    right_hand_pose=z45[:b],
                    transl=trans[mask],
                    return_verts=True,
                )["vertices"]
                output_vertices[mask] = verts

            output_vertices = output_vertices @ rotmat

            faces = smpl_faces.unsqueeze(0).expand(B, -1, -1)
            meshes = Meshes(verts=output_vertices, faces=faces)

            scan_pc = sample_points_from_meshes(meshes, scan_points)

            if centering_mode == "gt_mask":
                pc_for_center = output_vertices[:, subsample_mask, :].clone()
            elif centering_mode == "gt":
                pc_for_center = output_vertices
            else:
                pc_for_center = scan_pc

            centroids = pc_for_center.mean(dim=1, keepdim=True)
            output_vertices_centered = output_vertices - centroids
            scan_pc = scan_pc - centroids

            if augment_partial_views_prob > 0.0 and augment_limbs_prob > 0.0:
                warnings.warn(
                    "Careful, you are using both partial view augmentation and partial limbs augmentation simultaneously",
                    UserWarning,
                    stacklevel=2,
                )

            # augment: partial views
            if augment_partial_views_prob > 0.0:
                meshes_centered = Meshes(verts=output_vertices_centered, faces=faces)
                dev = meshes_centered.device
                verts_pad = meshes_centered.verts_padded()
                centroid0 = verts_pad.mean(dim=1)
                bbox = meshes_centered.get_bounding_boxes()
                extents = bbox[:, 1] - bbox[:, 0]
                R = 2 * extents.max(dim=1).values

                angle = torch.rand(B, device=dev) * 2 * torch.pi
                dirs = torch.stack([angle.cos(), torch.zeros_like(angle), angle.sin()], dim=1)
                cam = centroid0 + dirs * R.unsqueeze(1)

                p, n = sample_points_from_meshes(meshes_centered, scan_points * 5, return_normals=True)
                view_vec = cam.unsqueeze(1) - p
                vis = (n * view_vec).sum(dim=-1) > 0

                scan_pc_new = torch.empty(B, scan_points, 3, device=dev)
                partial_mask = torch.rand(B, device=dev) < augment_partial_views_prob
                for b in range(B):
                    if partial_mask[b]:
                        idx = vis[b].nonzero(as_tuple=False).squeeze(1)
                        if idx.numel() >= scan_points:
                            choice = idx[torch.randperm(idx.numel(), device=dev)[:scan_points]]
                        elif idx.numel() > 0:
                            choice = torch.cat(
                                [idx, idx[torch.randint(idx.numel(), (scan_points - idx.numel(),), device=dev)]],
                                dim=0,
                            )
                        else:
                            choice = torch.randint(p.shape[1], (scan_points,), device=dev)
                    else:
                        P = p.shape[1]
                        choice = torch.randperm(P, device=dev)[:scan_points] if P >= scan_points else torch.randint(P, (scan_points,), device=dev)
                    scan_pc_new[b] = p[b, choice]
                scan_pc = scan_pc_new

            # augment: partial limbs
            if augment_limbs_prob > 0.0 and (vertex_to_id is not None):
                scan_labels = segment_scan(
                    output_vertices_centered,
                    scan_pc,
                    vertex_to_id,
                    chunk_size=missing_parts_chunk_size,
                )

                removable = [
                    # full limbs
                    #torch.tensor([2, 4, 5, 6], device=scan_labels.device),
                    #torch.tensor([12, 14, 15, 16], device=scan_labels.device),
                    #torch.tensor([10, 7, 3, 9], device=scan_labels.device),
                    #torch.tensor([20, 17, 13, 19], device=scan_labels.device),

                    # external limb halves
                    torch.tensor([4, 5, 6], device=scan_labels.device),
                    torch.tensor([14, 15, 16], device=scan_labels.device),
                    torch.tensor([7, 3, 9], device=scan_labels.device),
                    torch.tensor([17, 13, 19], device=scan_labels.device)
                ]

                _, Ns, _ = scan_pc.shape
                mask_miss = torch.rand(B, device=scan_pc.device) < augment_limbs_prob
                parts = torch.randint(len(removable), (B,), device=scan_pc.device)
                aug = torch.empty_like(scan_pc)

                for b in range(B):
                    if mask_miss[b]:
                        remove = removable[parts[b]]
                        lbl = scan_labels[b]
                        keep = scan_pc[b][~(lbl.unsqueeze(-1) == remove).any(-1)]
                        k = keep.size(0)
                        if k > 0:
                            idx = torch.randint(0, k, (Ns - k,), device=scan_pc.device)
                            aug[b] = torch.cat((keep, keep[idx]), dim=0)
                        else:
                            aug[b] = scan_pc[b]
                    else:
                        aug[b] = scan_pc[b]
                scan_pc = aug

            # noise after augmentations
            if add_noise:
                scan_pc = scan_pc + noise_std * torch.randn_like(scan_pc)

            # save point clouds
            base_dir = os.path.join(
                work_dir, "point_clouds", str(mode), str(ds_name), str(gender_str), str(subj), str(motion)
            )
            scan_np = scan_pc.detach().cpu().numpy()
            output_vertices_centered_np = output_vertices_centered.detach().cpu().numpy()

            for j, frame_id in enumerate(ids.tolist()):
                _write_pointcloud_trimesh(os.path.join(base_dir, f"{frame_id}_gt.ply"), output_vertices_centered_np[j])
                _write_pointcloud_trimesh(os.path.join(base_dir, f"{frame_id}_scan.ply"), scan_np[j])

    assert len(data_pose) != 0
    torch.save(torch.tensor(np.asarray(data_pose, np.float32)), out_posepath)
    base = out_posepath.replace("pose.pt", "{}")
    torch.save(torch.tensor(np.asarray(data_betas, np.float32)), base.format("betas.pt"))
    torch.save(torch.tensor(np.asarray(data_trans, np.float32)), base.format("trans.pt"))
    torch.save(torch.tensor(np.asarray(data_gender, np.int32)), base.format("gender.pt"))
    torch.save(data_dataset, base.format("dataset.pt"))
    torch.save(data_subject, base.format("subject.pt"))
    torch.save(data_motion, base.format("motion.pt"))
    torch.save(torch.tensor(data_frame, dtype=torch.int32), base.format("frame.pt"))
    return len(data_pose)

class AMASS_Augment(Dataset):
    def __init__(self, dataset_dir, dtype=torch.float32):
        self.ds = {}
        for data_fname in glob.glob(os.path.join(dataset_dir, "*.pt")):
            k = os.path.basename(data_fname).replace(".pt", "")
            self.ds[k] = torch.load(data_fname)
        self.dtype = dtype

    def __len__(self):
        return len(self.ds["trans"])

    def __getitem__(self, idx):
        sample = {k: self.ds[k][idx] for k in self.ds.keys()}
        sample["pose_matrot"] = aa2matrot(sample["pose"].view([-1, 3])).view(1, -1)
        return sample

def _save_run_configs(cfg: DictConfig, logger):
    run_dir = HydraConfig.get().runtime.output_dir

    resolved_path = os.path.join(run_dir, "preprocess_resolved.yaml")
    OmegaConf.save(OmegaConf.to_container(cfg, resolve=True), resolved_path)
    logger(f"Saved resolved config to {resolved_path}")

    # dataset metadata, e.g. to be used in training
    pc_splits = _cfg_get(cfg, "point_clouds.splits", {})
    pc_augment = _cfg_get(cfg, "point_clouds.augment", {})
    pc_masks = _cfg_get(cfg, "point_clouds.masks", {})

    if isinstance(pc_splits, DictConfig):
        pc_splits = OmegaConf.to_container(pc_splits, resolve=True) or {}
    if isinstance(pc_augment, DictConfig):
        pc_augment = OmegaConf.to_container(pc_augment, resolve=True) or {}
    if isinstance(pc_masks, DictConfig):
        pc_masks = OmegaConf.to_container(pc_masks, resolve=True) or {}

    meta = {
        "dataset": {
            "source": "amass",
            "amass_dir": str(cfg.amass_dir),
            "work_dir": str(cfg.work_dir),
            "stage_II_dir": "stage_II",
            "point_clouds_dir": "point_clouds",
            "point_clouds_any_split_active": bool(_any_pc_split_active(cfg)),
            "point_cloud_splits": pc_splits if isinstance(pc_splits, dict) else {},
            "point_cloud_augment": pc_augment if isinstance(pc_augment, dict) else {},
            "centering_according_to_mask": str(_cfg_get(cfg, "point_clouds.centering_according_to_mask", "gt_mask")),
            "subsample_mask_path": str(_cfg_get(cfg, "point_clouds.masks.subsample_mask_path")),
            "smpl_vert_segmentation_json": str(_cfg_get(cfg, "point_clouds.masks.smpl_vert_segmentation_json")),
            "splits": OmegaConf.to_container(cfg.splits, resolve=True) if "splits" in cfg else {},
            "masks": pc_masks if isinstance(pc_masks, dict) else {},
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


def prepare_amass(cfg: DictConfig, logger):
    work_dir = str(cfg.work_dir)
    os.makedirs(work_dir, exist_ok=True)

    _save_run_configs(cfg, logger)

    run_dir = HydraConfig.get().runtime.output_dir
    try:
        shutil.copy2(sys.argv[0], os.path.join(run_dir, os.path.basename(sys.argv[0])))
    except Exception:
        pass

    stageI_outdir = os.path.join(work_dir, "stage_I")
    os.makedirs(stageI_outdir, exist_ok=True)

    scan_ctx = None
    device = _resolve_device(str(_cfg_get(cfg, "dump.device", "auto")))

    # Stage I
    for split_name, datasets in cfg.amass_splits.items():
        if not _split_is_active(cfg, split_name):
            logger(f"[{split_name}] split disabled; skipping.")
            continue

        datasets = list(datasets)
        if len(datasets) == 0:
            logger(f"[{split_name}] no datasets assigned; skipping.")
            continue

        outpath = makepath(os.path.join(stageI_outdir, split_name, "pose.pt"), isfile=True)
        if os.path.exists(outpath):
            logger(f"[{split_name}] {outpath} exists; skipping stage I creation.")
            continue

        wants_pc = bool(_pc_split_is_active(cfg, split_name))
        if wants_pc and scan_ctx is None:
            scan_ctx = _build_scan_context(cfg, device)

        dump_amass2pytorch(
            datasets=list(datasets),
            amass_dir=str(cfg.amass_dir),
            out_posepath=outpath,
            logger=logger,
            work_dir=work_dir,
            mode=split_name,
            cfg=cfg,
            scan_ctx=scan_ctx,
        )

    # Stage II -> H5
    class AMASS_ROW(pytables.IsDescription):
        dataset = pytables.StringCol(32)
        subject = pytables.StringCol(32)
        motion = pytables.StringCol(64)
        frame = pytables.Int32Col()
        gender = pytables.Int16Col()
        pose = pytables.Float32Col(52 * 3)
        pose_matrot = pytables.Float32Col(52 * 9)
        betas = pytables.Float32Col(16)
        trans = pytables.Float32Col(3)

    stageII_outdir = makepath(os.path.join(work_dir, "stage_II"))
    batch_size = 256
    max_num_epochs = 1

    for split_name in cfg.amass_splits.keys():
        if not _split_is_active(cfg, split_name):
            continue

        datasets = list(datasets)
        if len(datasets) == 0:
            logger(f"[{split_name}] no datasets assigned; skipping stage II.")
            continue

        h5_outpath = os.path.join(stageII_outdir, f"{split_name}.h5")
        if os.path.exists(h5_outpath):
            continue

        split_stageI_dir = os.path.join(stageI_outdir, split_name)
        pose_path = os.path.join(split_stageI_dir, "pose.pt")
        trans_path = os.path.join(split_stageI_dir, "trans.pt")

        if not os.path.exists(pose_path) or not os.path.exists(trans_path):
            logger(f"[{split_name}] incomplete/missing Stage I files; skipping stage II.")
            continue

        ds = AMASS_Augment(split_stageI_dir)
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=12, drop_last=False)

        with pytables.open_file(h5_outpath, mode="w") as h5file:
            table = h5file.create_table("/", "data", AMASS_ROW)
            for _ in range(max_num_epochs):
                for bData in tqdm(dataloader):
                    for i in range(len(bData["trans"])):
                        table.row["dataset"] = bData["dataset"][i]
                        table.row["subject"] = bData["subject"][i]
                        table.row["motion"] = bData["motion"][i]
                        table.row["frame"] = int(bData["frame"][i])

                        for k in bData.keys():
                            if k not in ["dataset", "subject", "motion", "frame"]:
                                table.row[k] = c2c(bData[k][i])
                        table.row.append()
                    table.flush()

    # Stage III -> pt columns
    stageIII_outdir = makepath(os.path.join(work_dir, "stage_III"))

    for split_name in cfg.amass_splits.keys():
        if not _split_is_active(cfg, split_name):
            continue

        datasets = list(datasets)
        if len(datasets) == 0:
            logger(f"[{split_name}] no datasets assigned; skipping stage III.")
            continue

        h5_filepath = os.path.join(stageII_outdir, f"{split_name}.h5")
        if not os.path.exists(h5_filepath):
            continue

        with pytables.open_file(h5_filepath, mode="r") as h5file:
            data = h5file.get_node("/data")
            data_dict = {k: [] for k in data.colnames}
            for row in data:
                for k in data_dict.keys():
                    data_dict[k].append(row[k])

        for k, v in data_dict.items():
            outfname = makepath(os.path.join(stageIII_outdir, split_name, f"{k}.pt"), isfile=True)
            if os.path.exists(outfname):
                continue

            arr = np.asarray(v)
            if arr.dtype.kind in {"U", "S"}:
                torch.save(v, outfname)
            elif arr.dtype == object:
                if all(isinstance(x, (str, bytes, np.bytes_)) for x in v):
                    v_str = [
                        x.decode("utf-8", errors="ignore") if isinstance(x, (bytes, np.bytes_)) else x
                        for x in v
                    ]
                    torch.save(v_str, outfname)
                else:
                    torch.save(torch.from_numpy(np.asarray(arr, dtype=np.float32)), outfname)
            else:
                torch.save(torch.from_numpy(arr), outfname)

@hydra.main(version_base=None, config_path=".", config_name="config_amass")
def main(cfg: DictConfig):
    import smplx.body_models as bm
    bm.SMPL.SHAPE_SPACE_DIM = 16

    run_dir = HydraConfig.get().runtime.output_dir
    if _work_dir_has_existing_data(str(cfg.work_dir), run_dir):
        print(
            f"[AMASS Preprocessing] Work directory already contains data; "
            f"skipping preprocessing: {cfg.work_dir}"
        )
        return

    starttime = datetime.now().replace(microsecond=0)
    log_name = datetime.strftime(starttime, "%Y%m%d_%H%M")

    logger = log2file(makepath(run_dir, f"{log_name}.log", isfile=True))
    logger(f"[{str(cfg.expr_code)}] Begin")
    logger(str(cfg.msg))

    prepare_amass(cfg, logger=logger)

if __name__ == "__main__":
    main()