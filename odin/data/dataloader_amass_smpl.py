import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class AMASS_H5_Dataset(Dataset):
    def __init__(self, h5_path):
        self.f = h5py.File(h5_path, 'r')
        self.data = self.f['data']
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data[idx]
        return {
            "modality": "smpl", # SMPL data as opposed to point cloud data
            'dataset': "amass",
            'dataset_subset': row['dataset'].decode('utf-8'),
            'subject':        row['subject'].decode('utf-8'),
            'motion':         row['motion'].decode('utf-8'),
            'frame':          int(row['frame']),
            'betas':          torch.from_numpy(row['betas'].astype('float32')),
            #'dmpls':          torch.from_numpy(row['dmpl'].astype('float32')),
            'gender':         torch.tensor(int(row['gender']), dtype=torch.int64),
            'pose_aa':        torch.from_numpy(row['pose'].astype('float32')),
            'pose_rotmat':    torch.from_numpy(row['pose_matrot'].astype('float32')),
            'trans':          torch.from_numpy(row['trans'].astype('float32'))
        }
    
    def save_sample(self, *, sample: dict, output_dir):
        from pytorch3d.io import IO
        from pytorch3d.structures import Pointclouds
        import numpy as np
        import torch
        import trimesh
        from pathlib import Path

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

        dataset_subset = str(batch["dataset_subset"][i])
        subject = str(batch["subject"][i])
        sequence = str(batch["motion"][i])
        frame = int(batch["frame"][i])
        frame_str = str(frame)

        base = Path(output_dir) / "amass" / dataset_subset / gender / subject / sequence
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

        # Baseline saves (only if present)
        faces = sample.get("smpl_faces", None)

        if ("pred_pc" in sample) and ("gt_pc" in sample) and ("scan_pc" in sample):
            gt_pc = _pc(sample["gt_pc"])
            scan_pc = _pc(sample["scan_pc"])
            pred_pc = _pc(sample["pred_pc"])
            gt_pc_full = _pc(sample.get("gt_pc_full", gt_pc))

            _save_pc(base / f"{frame_str}_gt.ply", gt_pc)
            #_save_pc(base / f"{frame_str}_gt_full.ply", gt_pc_full)
            _save_mesh_or_pc(base / f"{frame_str}_gt_full_mesh.ply", gt_pc_full, faces)
            _save_pc(base / f"{frame_str}_scan.ply", scan_pc)
            _save_pc(base / f"{frame_str}_{tag}.ply", pred_pc)

            evolution = sample.get("evolution", None)
            if evolution is not None:
                evo_name = f"{frame_str}_evolution.pth" if tag == "pred" else f"{frame_str}_{tag}_evolution.pth"
                torch.save(evolution, str(base / evo_name))

        # SMPL fitting saves (only if present)
        out_stage_i = sample.get("smpl_fit_stage_i", None)
        out_stage_ii = sample.get("smpl_fit_stage_ii", None)

        if out_stage_i is not None:
            #_save_mesh_or_pc(base / f"{frame_str}_smpl_fit_stage_i.ply", out_stage_i, faces)
            _save_mesh_or_pc(base / f"{frame_str}_v2v_opt.ply", out_stage_i, faces)
        if out_stage_ii is not None:
            #_save_mesh_or_pc(base / f"{frame_str}_smpl_fit_stage_ii.ply", out_stage_ii, faces)
            _save_mesh_or_pc(base / f"{frame_str}_chamfer_opt.ply", out_stage_ii, faces)

        params_stage_i = sample.get("smpl_fit_params_stage_i", None)
        if params_stage_i is not None:
            np.savez(
                #str(base / f"{frame_str}_smpl_fit_params_stage_i.npz"),
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
                #str(base / f"{frame_str}_smpl_fit_params_stage_ii.npz"),
                str(base / f"{frame_str}_chamfer_opt.npz"),
                loss=loss_scalar,
                pose=_np(params_stage_ii.get("pose", None)),
                beta=_np(params_stage_ii.get("beta", None)),
                trans=_np(params_stage_ii.get("trans", None)),
                scale=_np(params_stage_ii.get("scale", None)),
            )

def create_dataloader(h5_path, batch_size=16, num_workers=4, shuffle=True, mode='train'):
    if os.path.isdir(h5_path):
        h5_file = os.path.join(h5_path, f'{mode}.h5')
    else:
        h5_file = h5_path
    ds = AMASS_H5_Dataset(h5_file)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)