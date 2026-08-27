import inspect
from typing import Optional
import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from tqdm import tqdm
from .model_utils import get_custom_betas
from .point_cloud_model import PointCloudModel
from pytorch3d.loss import chamfer_distance
from .pointnetpp.models import pointnet2_segmentation
import smplx
import numpy as np
import torch.nn as nn
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from diffusers import ModelMixin

def segment_scan(gt_vertices, scan_vertices, vertex_to_id):
    B, N_g, _ = gt_vertices.shape
    mapping = torch.full((N_g,), -1, dtype=torch.long, device=gt_vertices.device)
    for i, lbl in vertex_to_id.items():
        mapping[i] = lbl
    idxs = torch.cdist(scan_vertices, gt_vertices).argmin(dim=-1)
    return mapping[idxs]

class ConditionalPointCloudDiffusionModel(ModelMixin):
    def __init__(
        self,
        beta_start: float,
        beta_end: float,
        beta_schedule: str,
        point_cloud_model: str,
        point_cloud_model_embed_dim: int,
        num_point: int = 689,
        part_to_vertex: dict = None,
        subsample_mask: np.ndarray = None,
        smpl_root: str = None,
        augment_train=None,
        augment_val=None,
        augment_sample=None,
        scale_factor: float = 4.5,
        **kwargs,
    ):
        super().__init__()

        self.scale_factor = scale_factor

        # fix for https://github.com/vchoutas/smplx/issues/109
        import smplx.body_models as bm
        bm.SMPL.SHAPE_SPACE_DIM = 16

        self.augment_train = augment_train
        self.augment_val = augment_val
        self.augment_sample = augment_sample

        # PointNet++ for scan feature extraction
        self.pointnet = pointnet2_segmentation.PointNetv2Seg(
            checkpoint_path=None)    
        self.pointnet.eval() # overridden automatically if mode is train
        
        self.num_point = num_point

        # diffusion schedulers
        scheduler_kwargs = (
            {"trained_betas": get_custom_betas(beta_start, beta_end)}
            if beta_schedule == "custom"
            else {"beta_start": beta_start, "beta_end": beta_end, "beta_schedule": beta_schedule}
        )
        self.schedulers_map = {
            "ddpm": DDPMScheduler(**scheduler_kwargs, clip_sample=False),
            "ddim": DDIMScheduler(**scheduler_kwargs, clip_sample=False),
            "pndm": PNDMScheduler(**scheduler_kwargs),
        }
        self.scheduler = self.schedulers_map["ddpm"]

        # denoising transformer
        self.point_cloud_model = PointCloudModel(
            model_type=point_cloud_model,
            embed_dim=point_cloud_model_embed_dim,
            num_points=num_point,
        )
        
        # smpl models location
        self.smpl_root = smpl_root
        
        # for subsampling SMPL vertices consistently (689 out of 6890)
        self.subsample_mask = subsample_mask
        
        # part segmentation of SMPL (used for missing limb augmentation)
        self.part_to_vertex = part_to_vertex
        self.sorted_parts = sorted(self.part_to_vertex.keys())
        self.part_to_id = {part: idx for idx, part in enumerate(self.sorted_parts)}
        self.vertex_to_part = {vertex: part for part, vertices in self.part_to_vertex.items() for vertex in vertices}
        self.vertex_to_id = {int(vertex): self.part_to_id[part] for vertex, part in self.vertex_to_part.items()}

        # rotation matrix for making SMPL upright
        rot = torch.tensor([[1,0,0],[0,0,-1],[0,1,0]], dtype=torch.float32)
        self.register_buffer('rotmat', rot)
        
        # initialize batched SMPL models for each gender
        self.gender_map = {-1: 'male', 0: 'neutral', 1: 'female'}
        self.models = nn.ModuleDict()
        for code, name in self.gender_map.items():
            self.models[name] = smplx.create(
                model_path=self.smpl_root,
                model_type='smplh',
                gender=name,
                batch_size=16,
                num_betas=16,
                use_pca=False
            ).to(self.device)
            
        self.smpl_faces = torch.from_numpy(self.models['male'].faces.astype(np.int64))

    def _batch_to_device_(self, batch):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
        return batch

    # for switching between SMPL parameter and point cloud data
    def _get_modality_(self, batch) -> str:
        m = batch.get("modality", None)
        if m is None:
            raise KeyError("Missing batch['modality']. Expected 'pc' or 'smpl'.")
        if isinstance(m, (list, tuple)):
            modality = m[0]
        elif torch.is_tensor(m):
            modality = m[0]
        else:
            modality = m
        if modality not in ("pc", "smpl"):
            raise ValueError(f"Invalid modality='{modality}'. Expected 'pc' or 'smpl'.")
        return modality

    def _prepare_from_smpl_(self, batch, centering_according_to_mask: str = "scan"):
        if centering_according_to_mask not in {"gt_mask", "gt", "scan"}:
            raise ValueError(
                "centering_according_to_mask must be one of: "
                "'gt_mask', 'gt', 'scan'"
            )

        betas = batch["betas"]
        batch_size = betas.shape[0]
        output_vertices = torch.empty(batch_size, 6890, 3, device=self.device, dtype=torch.float32)

        if betas.shape[1] < 16:
            print(f"Warning: betas shape {betas.shape} is less than 16, which is unexpected.")

        pa = batch["pose_aa"]
        trans = batch["trans"]
        genders = batch["gender"].view(-1)

        # smpl forward pass which yields 3D mesh (vertices + faces)
        for code, name in self.gender_map.items():
            mask = (genders == code)
            if not mask.any():
                continue

            m = self.models[name]
            b = mask.sum()
            go = pa[mask, :3]       # (b,  3)
            bp = pa[mask, 3:66]     # (b, 63)
            lh = pa[mask, 66:111]   # (b, 45)
            rh = pa[mask, 111:156]  # (b, 45)
            z45 = torch.zeros(b, 45, device=self.device)

            verts = m(
                betas=betas[mask],
                global_orient=go,
                body_pose=bp,
                left_hand_pose=z45,   # lh,
                right_hand_pose=z45,  # rh,
                transl=trans[mask],
                return_verts=True,
            )["vertices"]
            output_vertices[mask] = verts

        output_vertices = output_vertices @ self.rotmat

        batch_size = output_vertices.shape[0]
        faces = self.smpl_faces.unsqueeze(0).expand(batch_size, -1, -1).to(self.device)

        # create (689,3) GT point cloud by subsampling SMPL vertices according to mask
        gt_pc = output_vertices[:, self.subsample_mask, :].clone()

        meshes = Meshes(verts=output_vertices, faces=faces)
        scan_pc = sample_points_from_meshes(meshes, 4096)

        if centering_according_to_mask == "gt_mask":
            pc_for_center = gt_pc
        elif centering_according_to_mask == "gt":
            pc_for_center = output_vertices
        else:
            pc_for_center = scan_pc

        centroids = pc_for_center.mean(dim=1, keepdim=True)
        gt_pc = gt_pc - centroids
        gt_pc_full = output_vertices.clone() - centroids
        scan_pc = scan_pc - centroids
        meshes = Meshes(verts=gt_pc_full, faces=faces)

        return {
            "gt_pc": gt_pc,
            "gt_pc_full": gt_pc_full,
            "scan_pc": scan_pc,
            "faces": faces,
            "meshes": meshes,
        }

    def _prepare_from_pc_(self, batch):
        scan_pc = batch["scan_pc"].float()

        # scans only (no GT)
        gt_pc = batch.get("gt_pc", None)
        if gt_pc is None:
            return {
                "gt_pc": None,
                "gt_pc_full": None,
                "scan_pc": scan_pc,
            }

        # has GT
        gt_pc = gt_pc.float()
        gt_pc_full = gt_pc.clone()
        gt_pc = gt_pc[:, self.subsample_mask, :]

        return {
            "gt_pc": gt_pc,
            "gt_pc_full": gt_pc_full,
            "scan_pc": scan_pc,
        }

    def _augment_add_noise_(self, scan_pc, augment_cfg):
        if (not getattr(augment_cfg, "add_noise", False)) or getattr(augment_cfg, "noise_std", 0.0) <= 0.0:
            return scan_pc
        return scan_pc + float(augment_cfg.noise_std) * torch.randn_like(scan_pc)

    def _augment_missing_limbs_(self, scan_pc, gt_pc, augment_cfg):
        missing_prob = float(getattr(augment_cfg, "augment_limbs_prob", 0.0))
        if missing_prob <= 0.0:
            return scan_pc

        if gt_pc is None:
            return scan_pc

        scan_labels = segment_scan(gt_pc, scan_pc, self.vertex_to_id)

         # each list of SMPL part labels corresponds to an outer limb for removal
        removable = [
            torch.tensor([4,5,6], device=scan_labels.device),
            torch.tensor([14,15,16], device=scan_labels.device),
            torch.tensor([7,3,9], device=scan_labels.device),
            torch.tensor([17,13,19], device=scan_labels.device)
        ]

        B, N, _ = scan_pc.shape
        mask = torch.rand(B, device=scan_pc.device) < missing_prob
        parts = torch.randint(len(removable), (B,), device=scan_pc.device)
        aug = torch.empty_like(scan_pc)
        for b in range(B):
            if mask[b]:
                remove = removable[parts[b]]
                lbl = scan_labels[b]
                keep = scan_pc[b][~(lbl.unsqueeze(-1) == remove).any(-1)]
                k = keep.size(0)
                if k > 0:
                    idx = torch.randint(0, k, (N - k,), device=scan_pc.device)
                    aug[b] = torch.cat((keep, keep[idx]), dim=0)
                else:
                    aug[b] = scan_pc[b]
            else:
                aug[b] = scan_pc[b]
        return aug

    def _augment_partial_views_(self, scan_pc, meshes, augment_cfg):
        partial_prob = float(getattr(augment_cfg, "augment_partial_views_prob", 0.0))
        if partial_prob <= 0.0:
            return scan_pc

        if meshes is None:
            return scan_pc

        B = scan_pc.shape[0]
        device = scan_pc.device
        scan_points = 4096

        mask = torch.rand(B, device=device) < partial_prob
        if not mask.any():
            return scan_pc

        verts = meshes.verts_padded()
        centroid = verts.mean(dim=1)
        bbox = meshes.get_bounding_boxes()
        extents = bbox[:, 1] - bbox[:, 0]
        R = 2 * extents.max(dim=1).values
        angle = torch.rand(B, device=device) * 2 * torch.pi
        dirs = torch.stack([angle.cos(), torch.zeros_like(angle), angle.sin()], dim=1)
        cam = centroid + dirs * R.unsqueeze(1)

        p, n = sample_points_from_meshes(meshes, scan_points * 5, return_normals=True)
        view_vec = cam.unsqueeze(1) - p
        vis = (n * view_vec).sum(dim=-1) > 0

        aug = scan_pc
        aug_mask = mask.nonzero(as_tuple=False).squeeze(1).tolist()
        for b in aug_mask:
            idx = vis[b].nonzero(as_tuple=False).squeeze(1)
            if idx.numel() >= scan_points:
                choice = idx[torch.randperm(idx.numel(), device=device)[:scan_points]]
            elif idx.numel() > 0:
                choice = torch.cat(
                    [idx, idx[torch.randint(idx.numel(), (scan_points - idx.numel(),), device=device)]],
                    dim=0,
                )
            else:
                choice = torch.randint(p.shape[1], (scan_points,), device=device)
            aug[b] = p[b, choice]
        return aug

    def forward_train(self, batch):        
        batch = self._batch_to_device_(batch)
        modality = self._get_modality_(batch)

        if modality == "smpl":
            prep = self._prepare_from_smpl_(batch)
            gt_pc = prep["gt_pc"]
            gt_pc_full = prep["gt_pc_full"]
            scan_pc = prep["scan_pc"]
            meshes = prep["meshes"]
            faces = prep["faces"]

            # on-the-fly augmentation
            if getattr(self.augment_train, "augment_partial_views_prob", 0.0) > 0.0:
                scan_pc = self._augment_partial_views_(scan_pc, meshes, self.augment_train)
            if getattr(self.augment_train, "augment_limbs_prob", 0.0) > 0.0:
                scan_pc = self._augment_missing_limbs_(scan_pc, gt_pc_full, self.augment_train)
            if getattr(self.augment_train, "add_noise", False) and getattr(self.augment_train, "noise_std", 0.0) > 0.0:
                scan_pc = self._augment_add_noise_(scan_pc, self.augment_train)

        elif modality == "pc":
            prep = self._prepare_from_pc_(batch)
            gt_pc = prep["gt_pc"]
            scan_pc = prep["scan_pc"]

        else:
            raise NotImplementedError()

        x_0 = gt_pc * self.scale_factor
        
        scan_pc_tensor = scan_pc

        # add diffusion noise
        B, N, D = x_0.shape
        noise = torch.randn_like(x_0)

        timestep = torch.randint(0, self.scheduler.num_train_timesteps, (B,), device=self.device, dtype=torch.long)
        x_t = self.scheduler.add_noise(x_0, noise, timestep)

        # extract PointNet++ features from scan point clouds
        global_features_pn, local_features_pn, _ = self.pointnet(scan_pc_tensor)
        scan_pc_tensor = scan_pc_tensor * self.scale_factor
        local_features_pn = local_features_pn.permute(0, 2, 1)
        
        # assign each noised GT point the local PointNet++ feature of its nearest scan point.
        distances = torch.cdist(x_t, scan_pc_tensor, p=2)  # inferred
        #distances = torch.cdist(x_0, scan_pc_tensor, p=2) # privileged, only for benchmarking
        closest_points_idx = torch.argmin(distances, dim=2).unsqueeze(2).expand(-1, -1, local_features_pn.size(2))
        local_features_pn = torch.gather(local_features_pn, dim=1, index=closest_points_idx)

        x_t_input = x_t

        # denoising transformer predicts noise of GT points enriched with PointNet++ features
        noise_pred = self.point_cloud_model(
            coords=x_t_input,
            global_features=global_features_pn,
            local_features=local_features_pn,
            t=timestep,
        )

        loss = F.mse_loss(noise_pred, noise)
        return loss


    @torch.no_grad()
    def forward_sample(
        self,
        scan_pc,
        num_points: int = 689,
        scheduler: Optional[str] = "ddpm",
        num_inference_steps: Optional[int] = 1000,
        eta: Optional[float] = 0.0,  # for DDIM
        return_sample_every_n_steps: int = -1,
        disable_tqdm: bool = False,
    ):
        device = self.device
        
        # get scheduler from mapping, or use self.scheduler if None
        scheduler = self.scheduler if scheduler is None else self.schedulers_map[scheduler]

        # number of points of noise point cloud
        N = num_points

        # batch size
        B = scan_pc.shape[0]

        # point dimension
        D = 3

        # sample pure Gaussian noise
        x_t = torch.randn(B, N, D, device=device)
        scan_pc_tensor = scan_pc.to(device=self.device)

        # set timesteps
        accepts_offset = "offset" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        extra_set_kwargs = {"offset": 1} if accepts_offset else {}
        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # ensure scheduler buffers are on the same device as model inputs. For torch 2.x compatibility.
        device = x_t.device
        if hasattr(scheduler, "timesteps"):
            scheduler.timesteps = scheduler.timesteps.to(device)
        for attr in ["betas", "alphas_cumprod", "alphas"]:
            if hasattr(scheduler, attr):
                tensor = getattr(scheduler, attr)
                if isinstance(tensor, torch.Tensor):
                    setattr(scheduler, attr, tensor.to(device))

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
        extra_step_kwargs = {"eta": eta} if accepts_eta else {}

        global_features_pn, local_features_pn, _ = self.pointnet(scan_pc_tensor)
        scan_pc_tensor = scan_pc_tensor * self.scale_factor
        local_features_pn = local_features_pn.permute(0, 2, 1)

        # loop over timesteps
        all_outputs = []
        return_all_outputs = (return_sample_every_n_steps > 0)
        progress_bar = tqdm(scheduler.timesteps.to(device), desc=f"Sampling ({x_t.shape})", disable=disable_tqdm)
        for i, t in enumerate(progress_bar):

            x_t_input = x_t

            distances = torch.cdist(x_t, scan_pc_tensor, p=2)  # inferred
            #distances = torch.cdist(gt_pc_tensor, scan_pc_tensor, p=2) # privileged
            closest_points_idx = torch.argmin(distances, dim=2).unsqueeze(2).expand(-1, -1, local_features_pn.size(2))
            local_features_pn_iter = torch.gather(local_features_pn, dim=1, index=closest_points_idx)

            noise_pred = self.point_cloud_model(
                coords=x_t_input,
                global_features=global_features_pn,
                local_features=local_features_pn_iter,
                t=t.reshape(1).expand(B),
            )

            # step
            x_t = scheduler.step(noise_pred, t, x_t, **extra_step_kwargs).prev_sample

            if return_all_outputs and (i % return_sample_every_n_steps == 0 or i == len(scheduler.timesteps) - 1):
                all_outputs.append(x_t)

        output = x_t / self.scale_factor

        if return_all_outputs:
            all_outputs = torch.stack(all_outputs, dim=1)  # (B, sample_steps, N, D)
            all_outputs = [o / self.scale_factor for o in all_outputs]

        return (output, all_outputs) if return_all_outputs else output


    def forward_validate(self, batch, return_outputs: bool = False):
        batch = self._batch_to_device_(batch)
        modality = self._get_modality_(batch)

        if modality == "smpl":
            prep = self._prepare_from_smpl_(batch)
            gt_pc = prep["gt_pc"]
            gt_pc_full = prep["gt_pc_full"]
            scan_pc = prep["scan_pc"]
            meshes = prep["meshes"]
            faces = prep["faces"]

            # On-the-fly augmentation
            if getattr(self.augment_val, "augment_partial_views_prob", 0.0) > 0.0:
                scan_pc = self._augment_partial_views_(scan_pc, meshes, self.augment_val)
            if getattr(self.augment_val, "augment_limbs_prob", 0.0) > 0.0:
                scan_pc = self._augment_missing_limbs_(scan_pc, gt_pc_full, self.augment_val)
            if getattr(self.augment_val, "add_noise", False) and getattr(self.augment_val, "noise_std", 0.0) > 0.0:
                scan_pc = self._augment_add_noise_(scan_pc, self.augment_val)

        elif modality == "pc":
            prep = self._prepare_from_pc_(batch)
            gt_pc = prep["gt_pc"]
            gt_pc_full = prep.get("gt_pc_full", None)
            scan_pc = prep["scan_pc"]

        else:
            raise NotImplementedError()

        num_points = gt_pc.shape[1]

        output = self.forward_sample(
            num_points=num_points,
            scan_pc=scan_pc,
            num_inference_steps=50,
            disable_tqdm=True,
        )

        pred_pc = output.to(torch.float32)
        gt_points = gt_pc.to(torch.float32)

        cd_loss, _ = chamfer_distance(pred_pc, gt_points)
        v2v_loss = torch.norm(pred_pc - gt_points, dim=-1).mean()

        if return_outputs:
            return cd_loss, v2v_loss, gt_pc, gt_pc_full, scan_pc, pred_pc

        return cd_loss, v2v_loss

    def forward(self, batch: dict, mode: str = "train", **kwargs):
        if mode == "train":
            return self.forward_train(batch)

        elif mode == "validate":
            return self.forward_validate(batch, **kwargs)

        elif mode == "sample":
            batch = self._batch_to_device_(batch)
            modality = self._get_modality_(batch)

            if modality == "smpl":
                prep = self._prepare_from_smpl_(batch)
                gt_pc = prep["gt_pc"]
                gt_pc_full = prep["gt_pc_full"]
                scan_pc = prep["scan_pc"]
                meshes = prep["meshes"]
                faces = prep["faces"]

                # on-the-fly augmentation
                if getattr(self.augment_sample, "augment_partial_views_prob", 0.0) > 0.0:
                    scan_pc = self._augment_partial_views_(scan_pc, meshes, self.augment_sample)
                if getattr(self.augment_sample, "augment_limbs_prob", 0.0) > 0.0:
                    scan_pc = self._augment_missing_limbs_(scan_pc, gt_pc_full, self.augment_sample)
                if getattr(self.augment_sample, "add_noise", False) and getattr(self.augment_sample, "noise_std", 0.0) > 0.0:
                    scan_pc = self._augment_add_noise_(scan_pc, self.augment_sample)

            elif modality == "pc":
                prep = self._prepare_from_pc_(batch)
                gt_pc = prep["gt_pc"]
                gt_pc_full = prep["gt_pc_full"]
                scan_pc = prep["scan_pc"]

                if gt_pc is None:
                    out, all_out = self.forward_sample(
                        num_points=int(self.num_point),
                        scan_pc=scan_pc,
                        **kwargs,
                    )
                    return None, None, scan_pc, out, all_out
            else:
                raise NotImplementedError()

            num_points = gt_pc.shape[1]

            out, all_out = self.forward_sample(
                num_points=num_points,
                scan_pc=scan_pc,
                **kwargs,
            )

            return gt_pc, gt_pc_full, scan_pc, out, all_out

        else:
            raise NotImplementedError()