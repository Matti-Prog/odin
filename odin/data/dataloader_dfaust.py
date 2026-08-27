import os
import glob
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
import trimesh

inverse_gender_map = {"male": -1, "neutral": 0, "female": 1}

class DFaustPointCloudDataset(Dataset):
    """
    Folder structure:
      root / mode (train|val|test) / gender / sid / seq / *_scan.ply, *_gt.ply
    """
    def __init__(self, root, mode="train"):
        if mode not in ("train", "val", "test"):
            raise ValueError("mode must be 'train', 'val', or 'test'")

        self.root_dir = os.path.join(root, mode)
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        if not os.path.isdir(self.root_dir):
            return samples

        for gender in ("male", "female", "neutral"):
            gender_dir = os.path.join(self.root_dir, gender)
            if not os.path.isdir(gender_dir):
                continue

            for sid in sorted(os.listdir(gender_dir)):
                sid_dir = os.path.join(gender_dir, sid)
                if not os.path.isdir(sid_dir):
                    continue

                for seq in sorted(os.listdir(sid_dir)):
                    seq_dir = os.path.join(sid_dir, seq)
                    if not os.path.isdir(seq_dir):
                        continue

                    scan_files = sorted(glob.glob(os.path.join(seq_dir, "*_scan.ply")))
                    for scan_path in scan_files:
                        frame_str = os.path.basename(scan_path).split("_")[0]
                        gt_path = os.path.join(seq_dir, f"{frame_str}_gt.ply")
                        if not os.path.exists(gt_path):
                            continue

                        samples.append(
                            {
                                "gender": inverse_gender_map[gender],
                                "sid": sid,
                                "sequence": seq,
                                "frame": frame_str,
                                "scan_path": scan_path,
                                "gt_path": gt_path,
                            }
                        )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        scan_pc = torch.from_numpy(
            np.asarray(trimesh.load(s["scan_path"], process=False).vertices, dtype=np.float32)
        )
        gt_pc = torch.from_numpy(
            np.asarray(trimesh.load(s["gt_path"], process=False).vertices, dtype=np.float32)
        )

        return {
            "modality": "pc",
            "dataset_name": "dfaust",
            "subject": s["sid"],
            "motion": s["sequence"],
            "frame": int(s["frame"]),
            "gender": s["gender"],
            "gt_pc": gt_pc,
            "scan_pc": scan_pc,
        }

    def save_sample(self, *, sample: dict, output_dir):
        from pytorch3d.io import IO
        from pytorch3d.structures import Pointclouds

        io = IO()

        batch = sample["batch"]
        i = int(sample["index"])

        gender_id = int(batch["gender"][i].item()) if torch.is_tensor(batch["gender"]) else int(batch["gender"][i])
        if gender_id == -1:
            gender = "male"
        elif gender_id == 1:
            gender = "female"
        else:
            gender = "neutral"

        subject = str(batch["subject"][i])
        sequence = str(batch["motion"][i])
        frame = int(batch["frame"][i])
        frame_str = str(frame)

        base = Path(output_dir) / "dfaust" / gender / subject / sequence
        base.mkdir(parents=True, exist_ok=True)
        tag = str(sample.get("tag", "pred"))

        def _t(x):
            if x is None:
                return None
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x)
            if torch.is_tensor(x):
                return x
            return torch.tensor(x)

        def _pc(x):
            x = _t(x)
            if x is None:
                return None
            x = x.detach()
            if x.is_cuda:
                x = x.cpu()
            return x.to(torch.float32)

        def _np(x):
            if x is None:
                return None
            if isinstance(x, np.ndarray):
                return x
            if torch.is_tensor(x):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        def _save_pc(path: Path, pts: torch.Tensor):
            io.save_pointcloud(Pointclouds(pts.unsqueeze(0)), path=str(path))

        def _save_mesh_or_pc(path: Path, verts_any, faces_any):
            verts = _np(verts_any)
            faces = _np(faces_any)
            if verts is None:
                return
            if faces is not None and faces.ndim == 2 and faces.shape[1] == 3 and verts.ndim == 2:
                trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(str(path))
            else:
                _save_pc(path, _pc(verts_any))

        faces = sample.get("smpl_faces", None)

        if ("pred_pc" in sample) and ("gt_pc" in sample) and ("scan_pc" in sample):
            gt_pc = _pc(sample["gt_pc"])
            scan_pc = _pc(sample["scan_pc"])
            pred_pc = _pc(sample["pred_pc"])
            gt_pc_full = _pc(sample.get("gt_pc_full", gt_pc))

            _save_pc(base / f"{frame_str}_gt.ply", gt_pc)
            _save_mesh_or_pc(base / f"{frame_str}_gt_full_mesh.ply", gt_pc_full, faces)
            _save_pc(base / f"{frame_str}_scan.ply", scan_pc)
            _save_pc(base / f"{frame_str}_{tag}.ply", pred_pc)

            evolution = sample.get("evolution", None)
            if evolution is not None:
                evo_name = f"{frame_str}_evolution.pth" if tag == "pred" else f"{frame_str}_{tag}_evolution.pth"
                torch.save(evolution, str(base / evo_name))

        out_stage_i = sample.get("smpl_fit_stage_i", None)
        out_stage_ii = sample.get("smpl_fit_stage_ii", None)

        if out_stage_i is not None:
            _save_mesh_or_pc(base / f"{frame_str}_v2v_opt.ply", out_stage_i, faces)
        if out_stage_ii is not None:
            _save_mesh_or_pc(base / f"{frame_str}_chamfer_opt.ply", out_stage_ii, faces)

        params_stage_i = sample.get("smpl_fit_params_stage_i", None)
        if params_stage_i is not None:
            np.savez(
                str(base / f"{frame_str}_v2v_opt.npz"),
                pose=_np(params_stage_i.get("pose", None)),
                beta=_np(params_stage_i.get("beta", None)),
                trans=_np(params_stage_i.get("trans", None)),
                scale=_np(params_stage_i.get("scale", None)),
            )

        params_stage_ii = sample.get("smpl_fit_params_stage_ii", None)
        if params_stage_ii is not None:
            loss_val = params_stage_ii.get("loss", None)
            loss_np = _np(loss_val)
            loss_scalar = float(loss_np) if loss_np is not None and np.size(loss_np) == 1 else np.nan
            np.savez(
                str(base / f"{frame_str}_chamfer_opt.npz"),
                loss=loss_scalar,
                pose=_np(params_stage_ii.get("pose", None)),
                beta=_np(params_stage_ii.get("beta", None)),
                trans=_np(params_stage_ii.get("trans", None)),
                scale=_np(params_stage_ii.get("scale", None)),
            )

def simple_collate(batch):
    return {k: default_collate([b[k] for b in batch]) for k in batch[0]}

def create_dataloader(
    root,
    batch_size=4,
    num_workers=0,
    shuffle=True,
    mode="train",
):
    ds = DFaustPointCloudDataset(
        root,
        mode=mode,
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=simple_collate,
    )
