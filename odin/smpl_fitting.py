from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.ops import knn_gather, knn_points
from pytorch3d.loss.chamfer import _handle_pointcloud_input, _validate_chamfer_reduction_inputs

import smplx.body_models as bm
bm.SMPL.SHAPE_SPACE_DIM = 300
import smplx

from smplfitter.pt import BodyModel, BodyFitter

from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser

def _get_smpl_fitting_cfg(cfg: Any) -> Any:
    if hasattr(cfg, "smpl_fitting"):
        return cfg.smpl_fitting
    if hasattr(cfg, "run") and hasattr(cfg.run, "smpl_fitting"):
        return cfg.run.smpl_fitting
    if hasattr(cfg, "run"):
        return cfg.run
    raise AttributeError("Could not find SMPL fitting config at cfg.smpl_fitting or cfg.run.smpl_fitting.")

class VposerPrior(torch.nn.Module):
    def __init__(self, vposer: torch.nn.Module):
        super().__init__()
        self.vposer = vposer

    def forward(self, pose_nonhand: torch.Tensor) -> torch.Tensor:
        enc = self.vposer.encode(pose_nonhand)
        latent = enc.mean
        return torch.mean(latent ** 2)

def _chamfer_single_direction(
    x: torch.Tensor,
    y: torch.Tensor,
    x_lengths: torch.Tensor,
    y_lengths: torch.Tensor,
    x_normals: Optional[torch.Tensor],
    y_normals: Optional[torch.Tensor],
    weights: Optional[torch.Tensor],
    batch_reduction: Optional[str],
    point_reduction: Optional[str],
    norm: int,
    abs_cosine: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    return_normals = (x_normals is not None) and (y_normals is not None)

    N, P1, D = x.shape
    is_x_hetero = (x_lengths != P1).any()
    x_mask = torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")

    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                z = (x.sum((1, 2)) * weights).sum() * 0.0
                return z, z
            z = (x.sum((1, 2)) * weights) * 0.0
            return z, z

    cham_norm_x = x.new_zeros(())
    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
    cham_x = x_nn.dists[..., 0]  # (N, P1)

    if is_x_hetero:
        cham_x[x_mask] = 0.0
    if weights is not None:
        cham_x *= weights.view(N, 1)

    if return_normals:
        y_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        cosine_sim = F.cosine_similarity(x_normals, y_normals_near, dim=2, eps=1e-6)
        cham_norm_x = 1 - (torch.abs(cosine_sim) if abs_cosine else cosine_sim)
        if is_x_hetero:
            cham_norm_x[x_mask] = 0.0
        if weights is not None:
            cham_norm_x *= weights.view(N, 1)

    if point_reduction is not None:
        cham_x = cham_x.sum(1)
        if return_normals:
            cham_norm_x = cham_norm_x.sum(1)

        if point_reduction == "mean":
            x_lengths_clamped = x_lengths.clamp(min=1)
            cham_x = cham_x / x_lengths_clamped
            if return_normals:
                cham_norm_x = cham_norm_x / x_lengths_clamped

        if batch_reduction is not None:
            cham_x = cham_x.sum()
            if return_normals:
                cham_norm_x = cham_norm_x.sum()
            if batch_reduction == "mean":
                div = weights.sum() if weights is not None else max(N, 1)
                cham_x = cham_x / div
                if return_normals:
                    cham_norm_x = cham_norm_x / div

    return cham_x, (cham_norm_x if return_normals else None)


def chamfer_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    x_lengths: Optional[torch.Tensor] = None,
    y_lengths: Optional[torch.Tensor] = None,
    x_normals: Optional[torch.Tensor] = None,
    y_normals: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
    batch_reduction: Optional[str] = "mean",
    point_reduction: Optional[str] = "mean",
    norm: int = 2,
    single_directional: bool = False,
    abs_cosine: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)
    if norm not in (1, 2):
        raise ValueError("Support for 1 or 2 norm only.")

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    cham_x, cham_norm_x = _chamfer_single_direction(
        x, y, x_lengths, y_lengths, x_normals, y_normals,
        weights, batch_reduction, point_reduction, norm, abs_cosine
    )

    if single_directional:
        return cham_x, cham_norm_x

    cham_y, cham_norm_y = _chamfer_single_direction(
        y, x, y_lengths, x_lengths, y_normals, x_normals,
        weights, batch_reduction, point_reduction, norm, abs_cosine
    )

    if point_reduction is not None:
        return cham_x + cham_y, (cham_norm_x + cham_norm_y) if cham_norm_x is not None else None

    return (cham_x, cham_y), (cham_norm_x, cham_norm_y) if cham_norm_x is not None else None


class ChamferDistance(torch.nn.Module):
    def forward(
        self,
        source_cloud: torch.Tensor,
        target_cloud: torch.Tensor,
        bidirectional: bool = False,
        reverse: bool = False,
        batch_reduction: Optional[str] = "mean",
        point_reduction: Optional[str] = "sum",
    ) -> torch.Tensor:
        if reverse:
            source_cloud, target_cloud = target_cloud, source_cloud
        dist, _ = chamfer_distance(
            source_cloud,
            target_cloud,
            single_directional=not bidirectional,
            batch_reduction=batch_reduction,
            point_reduction=point_reduction,
        )
        return dist

class OptimizationSMPL(torch.nn.Module):
    def __init__(self, batch_size: int, device: torch.device):
        super().__init__()
        self.pose = torch.nn.Parameter(torch.zeros(batch_size, 72, device=device))
        self.beta = torch.nn.Parameter(torch.zeros(batch_size, 300, device=device))
        self.trans = torch.nn.Parameter(torch.zeros(batch_size, 3, device=device))
        self.scale = torch.nn.Parameter(torch.ones(batch_size, device=device))

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.pose, self.beta, self.trans, self.scale

class SMPLFittingRunner:
    def __init__(self, cfg: Any, device: Optional[Union[str, torch.device]] = None):
        self.cfg = cfg
        self.fit_cfg = _get_smpl_fitting_cfg(cfg)

        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # subset indices (689,)
        self.sub_idx = np.load(self.cfg.assets.subsample_mask).astype(np.int64)
        self.sub_idx_t = torch.from_numpy(self.sub_idx).long().to(self.device)

        # Stage I: smplfitter BodyModel + BodyFitter
        self.body_model_native = BodyModel("smpl", 
                                           "neutral",
                                           model_root = str(self.cfg.assets.smpl_root) + "/smpl_smplfitter",
                                           num_betas=300).to(self.device)
        self.num_vertices = int(self.body_model_native.v_template.shape[0])

        fitter_mod = BodyFitter(self.body_model_native).to(self.device)
        self.fitter = torch.jit.script(fitter_mod)

        # Stage II: smplx SMPL model
        self._smplx_models: Dict[int, torch.nn.Module] = {}

        # VPoser
        self._vposer: Optional[torch.nn.Module] = None
        self._vposer_prior: Optional[VposerPrior] = None

        self._chamfer = ChamferDistance().to(self.device)

    def _get_smplx(self, batch_size: int) -> torch.nn.Module:      
        m = self._smplx_models.get(batch_size, None)
        if m is not None:
            return m
        smpl_root = str(self.cfg.assets.smpl_root)
        
        model = smplx.create(
            model_type="smpl",
            gender="neutral",
            num_betas=300,
            model_path=smpl_root,
            batch_size=batch_size,
        ).to(self.device)

        self._smplx_models[batch_size] = model

        return model

    def _ensure_vposer(self) -> None:
        if self._vposer is not None and self._vposer_prior is not None:
            return
        expr_dir = str(self.cfg.assets.vposer_checkpoint)
        vp, _ = load_model(
            expr_dir,
            model_code=VPoser,
            remove_words_in_model_weights="vp_model.",
            disable_grad=True,
        )
        vp = vp.to(self.device)
        self._vposer = vp
        self._vposer_prior = VposerPrior(vp).to(self.device)

    @torch.no_grad()
    def _stage1_v2v(self, pred_pc_689: torch.Tensor) -> Dict[str, Any]:
        pred_pc_689 = pred_pc_689.to(self.device, dtype=torch.float32)
        B = int(pred_pc_689.shape[0])

        expanded = torch.zeros((B, self.num_vertices, 3), device=self.device, dtype=torch.float32)
        expanded[:, self.sub_idx_t, :] = pred_pc_689

        mask = torch.zeros((B, self.num_vertices), device=self.device, dtype=torch.float32)
        mask[:, self.sub_idx_t] = 1.0

        res = self.fitter.fit(
            expanded,
            num_iter=int(self.fit_cfg.v2v_num_iter),
            beta_regularizer=float(self.fit_cfg.v2v_beta_regularizer),
            final_adjust_rots=bool(self.fit_cfg.v2v_final_adjust_rots),
            vertex_weights=mask,
        )

        pose = res["pose_rotvecs"]
        betas = res["shape_betas"]
        trans = res["trans"]
        scale = torch.ones((B,), device=self.device, dtype=torch.float32)

        out_native = self.body_model_native(pose, betas, trans)
        out_vertices = out_native["vertices"]

        return {
            "out_v2v": out_vertices.detach().cpu().numpy(),
            "params_v2v": {
                "pose": pose.detach(),
                "beta": betas.detach(),
                "trans": trans.detach(),
                "scale": scale.detach(),
            },
        }

    def _stage2_chamfer(
        self,
        init_params: Dict[str, torch.Tensor],
        scan_pc: torch.Tensor,
        bidir: int,
    ) -> Dict[str, Any]:
        self._ensure_vposer()
        assert self._vposer_prior is not None

        scan_pc = scan_pc.to(self.device, dtype=torch.float32)
        B = int(scan_pc.shape[0])

        smpl_model = self._get_smplx(B)

        params_smpl = OptimizationSMPL(B, self.device).to(self.device)

        params_smpl.pose = torch.nn.Parameter(init_params["pose"].to(self.device))
        params_smpl.beta = torch.nn.Parameter(init_params["beta"].to(self.device))
        params_smpl.trans = torch.nn.Parameter(init_params["trans"].to(self.device))
        params_smpl.scale = torch.nn.Parameter(init_params["scale"].to(self.device))

        optimizer = torch.optim.Adam(params_smpl.parameters(), lr=float(self.fit_cfg.chamfer_lr))

        iterations = int(self.fit_cfg.chamfer_iterations)
        prior_w = float(self.fit_cfg.chamfer_prior_weight)
        beta_reg = float(self.fit_cfg.chamfer_beta_reg)

        last_loss: Optional[torch.Tensor] = None

        for i in range(iterations):
            pose_full, beta, trans, scale = params_smpl()

            out = smpl_model.forward(
                body_pose=pose_full[:, 3:72],
                global_orient=pose_full[:, :3],
                betas=beta,
                get_skin=True,
            )
            verts_smpl = (out.vertices + trans.unsqueeze(1)) * scale.unsqueeze(1).unsqueeze(2)

            if bidir == 0:
                d1 = torch.sqrt(self._chamfer(scan_pc, verts_smpl, bidirectional=False, reverse=False)).mean()
                d2 = torch.sqrt(self._chamfer(verts_smpl, scan_pc, bidirectional=False, reverse=False)).mean()
                cham = d1 + d2
            elif bidir == 1:
                cham = torch.sqrt(self._chamfer(scan_pc, verts_smpl, bidirectional=False, reverse=False)).mean()
            elif bidir == -1:
                cham = torch.sqrt(self._chamfer(verts_smpl, scan_pc, bidirectional=False, reverse=False)).mean()
            else:
                raise ValueError(f"bidir must be in {{-1,0,1}}, got {bidir}")
        
            prior_loss = self._vposer_prior(pose_full[:, :63])
            #prior_loss = self._vposer_prior(pose_full[:, 3:66])

            beta_loss = (beta ** 2).mean()
            loss = cham + prior_w * prior_loss + beta_reg * beta_loss
            last_loss = loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            lr0 = float(self.fit_cfg.chamfer_lr)
            for g in optimizer.param_groups:
                g["lr"] = lr0 * float(iterations - i) / float(iterations)
        
        with torch.no_grad():
            pose_full, beta, trans, scale = params_smpl()
            out = smpl_model.forward(
                body_pose=pose_full[:, 3:72],
                global_orient=pose_full[:, :3],
                betas=beta,
                get_skin=True,
            )
            verts_smpl = (out.vertices + trans.unsqueeze(1)) * scale.unsqueeze(1).unsqueeze(2)

        return {
            "out_cham": verts_smpl.detach().cpu().numpy(),
            "params_cham": {
                "loss": (last_loss.detach() if last_loss is not None else torch.tensor(float("nan"), device=self.device)),
                "pose": pose_full.detach(),
                "beta": beta.detach(),
                "trans": trans.detach(),
                "scale": scale.detach(),
            },
        }

    def fit(
        self,
        *,
        pred_pc_689: torch.Tensor,
        scan_pc: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        mode = str(self.fit_cfg.mode).lower()
        if mode not in ("none", "v2v", "v2v_chamfer"):
            raise ValueError(f"Unknown SMPL fitting mode: {mode}")

        if mode == "none":
            return {}

        if mode == "v2v_chamfer" and scan_pc is None:
            raise ValueError("scan_pc must be provided for mode='v2v_chamfer'.")

        pred_pc_689 = pred_pc_689.detach()
        if scan_pc is not None:
            scan_pc = scan_pc.detach()

        B = int(pred_pc_689.shape[0])
        chunk = int(self.fit_cfg.batch_size) if hasattr(self.fit_cfg, "batch_size") else B
        chunk = max(1, chunk)

        out_v2v_all: Optional[np.ndarray] = None
        out_cham_all: Optional[np.ndarray] = None

        params_v2v_cat: Dict[str, torch.Tensor] = {}
        params_cham_cat: Dict[str, torch.Tensor] = {}

        for s in range(0, B, chunk):
            e = min(B, s + chunk)

            res1 = self._stage1_v2v(pred_pc_689[s:e])
            
            out_v2v = res1["out_v2v"]
            p1 = res1["params_v2v"]

            out_v2v_all = out_v2v if out_v2v_all is None else np.concatenate([out_v2v_all, out_v2v], axis=0)
            for k, v in p1.items():
                params_v2v_cat[k] = v if k not in params_v2v_cat else torch.cat([params_v2v_cat[k], v], dim=0)

            if mode == "v2v_chamfer":
                assert scan_pc is not None
                bidir = int(getattr(self.fit_cfg, "chamfer_bidir", 1))
                res2 = self._stage2_chamfer(
                    init_params=p1,
                    scan_pc=scan_pc[s:e],
                    bidir=bidir,
                )
                out_cham = res2["out_cham"]
                p2 = res2["params_cham"]

                out_cham_all = out_cham if out_cham_all is None else np.concatenate([out_cham_all, out_cham], axis=0)
                for k, v in p2.items():
                    params_cham_cat[k] = v if k not in params_cham_cat else torch.cat([params_cham_cat[k], v], dim=0)
        
        out: Dict[str, Any] = {
            "out_v2v": out_v2v_all,
            "params_v2v": params_v2v_cat,
        }
        if mode == "v2v_chamfer":
            out["out_cham"] = out_cham_all
            out["params_cham"] = params_cham_cat

        return out