from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from hydra.core.config_store import ConfigStore
from hydra.conf import RunDir
from omegaconf import MISSING

@dataclass
class CustomHydraRunDir(RunDir):
    dir: str = './outputs/${run.name}/${now:%Y-%m-%d--%H-%M-%S}'

# Path to the project (repo) root
@dataclass
class PathsConfig:
    project_root: str = MISSING

@dataclass
class AssetsConfig:
    part_to_vertex: str = "${paths.project_root}/assets/smpl_vert_segmentation.json"
    subsample_mask: str = "${paths.project_root}/assets/smpl_subsampled_indices.npy"
    vposer_checkpoint: str = "${paths.project_root}/assets/vposer/V02_05"
    smpl_root: str = "${paths.project_root}/support_data/body_models"

@dataclass
class RunConfig:
    name: str = 'debug'
    job: str = 'train'
    mixed_precision: str = 'no'
    cpu: bool = False
    seed: int = 42
    save_val_samples: bool = False
    limit_train_batches: Optional[int] = None
    max_steps: int = 100_000
    checkpoint_freq: int = 10_000
    val_freq: int = 1_000_000 #25_000
    log_step_freq: int = 25
    print_step_freq: int = 100

    # -- Inference --

    # Use this dataset split when sampling (run.job == "sample")
    sample_split: str = "test" # "train", "vald", "test"
    
    num_inference_steps: int = 100 # <=1000
    diffusion_scheduler: Optional[str] = 'ddpm' #'ddpm', ddim, pndm

    # Single or multiple samples (predictions) per input;
    # keep all, or only the best according to Chamfer Distance between input scan and sample
    num_samples: int = 1
    save_mode: str = "best"  # "all", "best"
    chamfer_mode: int = 0 # 0: bidirectional, 1: scan->pred, -1: pred->scan
    
    num_sample_batches: Optional[int] = None
    sample_from_ema: bool = False # exponential moving average; unused
    sample_save_evolutions: bool = False  # yields .pth with intermediate point clouds of backward process

@dataclass
class DataloaderConfig:
    batch_size: int = 16
    num_workers: int = 0

@dataclass
class SMPLFittingConfig:
    """
    Controls optional SMPL fitting during run.job == 'sample'.
    mode:
      - "none": disable fitting
      - "v2v": run stage I only (v2v fitting via smplfitter)
      - "v2v_chamfer": run stage I + stage II (v2v fitting + Chamfer refinement)
    """
    mode: str = "v2v_chamfer"  # "none", "v2v", "v2v_chamfer

    # General
    batch_size: int = 64  # fitting batch size; can differ from diffusion batch size
    save_meshes: bool = True          # write .ply meshes
    save_params: bool = True          # write .npz params

    # Stage I
    v2v_num_iter: int = 2
    v2v_beta_regularizer: float = 1e-4
    v2v_final_adjust_rots: bool = False

    # Stage II
    chamfer_iterations: int = 1000
    chamfer_lr: float = 2e-2
    chamfer_bidir: int = 1  # 0=bidirectional, 1=scan->smpl, -1=smpl->scan
    chamfer_prior_weight: float = 1e-3
    chamfer_beta_reg: float = 0.2

@dataclass
class PointCloudDiffusionModelConfig:
    # Diffusion
    beta_start: float = 1e-5
    beta_end: float = 8e-3
    beta_schedule: str = "linear"

    # Denoiser
    point_cloud_model: str = "transformer_full"
    point_cloud_model_embed_dim: int = 1024

    # Scaling
    scale_factor: float = "${dataset.scale_factor}"

    # Vertices in diffusion point cloud
    num_point: int = 689

@dataclass
class LoggingConfig:
    wandb: bool = False
    wandb_project: str = "odin"
    wandb_entity: str = ""

# ================================ AMASS dataset config ==============================
# Sub-configs + main config for the AMASS dataset. We can use:
# - Point clouds (if available for a split and enabled via use_pc_if_available.<split>)
# - SMPL-format data (fallback when PC is unavailable or disabled for that split)

# Per split: Use point cloud data (if available), as opposed to SMPL-format data.
@dataclass
class UsePCIfAvailableConfig:
    train: bool = False
    vald: bool = False
    test: bool = True

# On-the-fly augmentation configs; apply only to SMPL data (not point clouds).
# Point clouds (e.g. for testing) can be augmented in AMASS preprocessing code.
@dataclass
class TrainOnTheFlyAugmentConfig:
    augment_limbs_prob: float = 0.0 # prob of a sample missing one external half of limb (e.g. arm + hand)
    augment_partial_views_prob: float = 0.0 # prob of using a partial scan from a single viewpoint around the vertical axis
    add_noise: bool = True
    noise_std: float = 0.0025

@dataclass
class ValOnTheFlyAugmentConfig:
    augment_limbs_prob: float = 0.0
    augment_partial_views_prob: float = 0.0
    add_noise: bool = True
    noise_std: float = 0.0025

@dataclass
class TestOnTheFlyAugmentConfig:
    augment_limbs_prob: float = 0.0
    augment_partial_views_prob: float = 0.0
    add_noise: bool = True
    noise_std: float = 0.0025

@dataclass
class AMASSConfig:
    """
    Assumes AMASS dataset produced by preprocessing code.
    The root also contains dataset metadata, resolved at runtime against this config.
    root:
      Parent directory containing Stage_I, Stage_II, Stage_III, optionally point_clouds, 
      and hydra_runs/**/dataset_meta.yaml
    """
    type: str = "amass"
    root: str = MISSING

    # Per-split preference: use point clouds if available
    use_pc_if_available: UsePCIfAvailableConfig = field(default_factory=UsePCIfAvailableConfig)

    scale_factor: float = 4.5 # Empirically determined scale factor for diffusion to work on SMPL-scale data

    # Explicit per-split on-the-fly augmentation (SMPL only)
    augment_train: TrainOnTheFlyAugmentConfig = field(default_factory=TrainOnTheFlyAugmentConfig)
    augment_vald: ValOnTheFlyAugmentConfig = field(default_factory=ValOnTheFlyAugmentConfig)
    augment_test: TestOnTheFlyAugmentConfig = field(default_factory=TestOnTheFlyAugmentConfig)

# ================================ DFAUST dataset config =============================
@dataclass
class DFAUSTConfig:
    """
    Assumes DFAUST dataset produced by preprocessing code.
      root/
        point_clouds/<split>/<gender>/<subject_id>/<sequence_name>/*.ply
        hydra_runs/**/dataset_meta.yaml
    """
    type: str = "dfaust"
    root: str = MISSING
    scale_factor: float = 4.5

# ================================ FAUST dataset config ==============================
@dataclass
class FAUSTConfig:
    """
    Assumes FAUST dataset produced by preprocessing code.
      root/
        point_clouds/<split>/{scans,gt}/*.ply
        hydra_runs/**/dataset_meta.yaml
    """
    type: str = "faust"
    root: str = MISSING
    scale_factor: float = 4.5

# ================================ SHREC19 dataset config ============================
@dataclass
class SHREC19Config:
    """
    Assumes SHREC19 dataset produced by preprocessing code.
      root/
        point_clouds/train/scans/*.ply
        hydra_runs/**/dataset_meta.yaml
    """
    type: str = "shrec19"
    root: str = MISSING
    scale_factor: float = 4.5

# Needed by hydra; keep
@dataclass
class LossConfig:
    diffusion_weight: float = 1.0
    rgb_weight: float = 1.0
    consistency_weight: float = 1.0

@dataclass
class CheckpointConfig:
    resume: Optional[str] = None
    resume_training: bool = True
    resume_training_optimizer: bool = True
    resume_training_scheduler: bool = True
    resume_training_state: bool = True

@dataclass
class ExponentialMovingAverageConfig:
    use_ema: bool = False
    # # From Diffusers EMA (should probably switch)
    # ema_inv_gamma: float = 1.0
    # ema_power: float = 0.75
    # ema_max_decay: float = 0.9999
    decay: float = 0.999
    update_every: int = 20

@dataclass
class OptimizerConfig:
    type: str
    name: str
    lr: float = 5e-4
    pointnet_lr_mult: float = 10.0 # learning rate multiplier for PointNet (feature extractor)
    weight_decay: float = 0.0
    scale_learning_rate_with_batch_size: bool = False
    gradient_accumulation_steps: int = 1
    clip_grad_norm: Optional[float] = 5.0
    kwargs: Dict = field(default_factory=lambda: dict())

@dataclass
class AdadeltaOptimizerConfig(OptimizerConfig):
    type: str = 'torch'
    name: str = 'Adadelta'
    kwargs: Dict = field(default_factory=lambda: dict(
        weight_decay=1e-6,
    ))

@dataclass
class AdamOptimizerConfig(OptimizerConfig):
    type: str = 'torch'
    name: str = 'AdamW'
    weight_decay: float = 1e-6
    kwargs: Dict = field(default_factory=lambda: dict(betas=(0.95, 0.999)))

@dataclass
class SchedulerConfig:
    type: str
    kwargs: Dict = field(default_factory=lambda: dict())

@dataclass
class LinearSchedulerConfig(SchedulerConfig):
    type: str = 'transformers'
    kwargs: Dict = field(default_factory=lambda: dict(
        name='linear',
        num_warmup_steps=0,
        num_training_steps="${run.max_steps}",
    ))

@dataclass
class CosineSchedulerConfig(SchedulerConfig):
    type: str = 'transformers'
    kwargs: Dict = field(default_factory=lambda: dict(
        name='cosine',
        num_warmup_steps=2000,
        num_training_steps="${run.max_steps}",
    ))

@dataclass
class ProjectConfig:
    paths: PathsConfig
    run: RunConfig
    assets: AssetsConfig
    smpl_fitting: SMPLFittingConfig
    logging: LoggingConfig
    dataloader: DataloaderConfig
    loss: LossConfig
    model: PointCloudDiffusionModelConfig
    ema: ExponentialMovingAverageConfig
    checkpoint: CheckpointConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig

    defaults: List[Any] = field(default_factory=lambda: [
        'custom_hydra_run_dir',
        {'run': 'default'},
        {"paths": "default"},
        {"assets": "default"},
        {"smpl_fitting": "default"},
        {'logging': 'default'},
        {'model': 'diffrec'},
        {'dataset': 'amass'},
        {'dataloader': 'default'},
        {'ema': 'default'},
        {'loss': 'default'},
        {'checkpoint': 'default'},
        {'optimizer': 'adam'},
        {'scheduler': 'cosine'}
    ])

cs = ConfigStore.instance()
cs.store(name='custom_hydra_run_dir', node=CustomHydraRunDir, package="hydra.run")
cs.store(group='run', name='default', node=RunConfig)
cs.store(group='logging', name='default', node=LoggingConfig)
cs.store(group='model', name='diffrec', node=PointCloudDiffusionModelConfig)
cs.store(group="assets", name="default", node=AssetsConfig)
cs.store(group="smpl_fitting", name="default", node=SMPLFittingConfig)
cs.store(group="paths", name="default", node=PathsConfig)
cs.store(group='dataloader', name='default', node=DataloaderConfig)
cs.store(group='loss', name='default', node=LossConfig)
cs.store(group='ema', name='default', node=ExponentialMovingAverageConfig)
cs.store(group='checkpoint', name='default', node=CheckpointConfig)
cs.store(group='optimizer', name='adadelta', node=AdadeltaOptimizerConfig)
cs.store(group='optimizer', name='adam', node=AdamOptimizerConfig)
cs.store(group='scheduler', name='linear', node=LinearSchedulerConfig)
cs.store(group='scheduler', name='cosine', node=CosineSchedulerConfig)
cs.store(name='config', node=ProjectConfig)

# Datasets
cs.store(group='dataset', name='faust', node=FAUSTConfig)
cs.store(group="dataset", name="amass", node=AMASSConfig)
cs.store(group='dataset', name='dfaust', node=DFAUSTConfig)
cs.store(group="dataset", name="shrec19", node=SHREC19Config)