from config.structured import ProjectConfig
from .model import ConditionalPointCloudDiffusionModel
import numpy as np

def get_model(
    cfg: ProjectConfig,
    part_to_vertex: dict,
    subsample_mask: np.ndarray,
    smpl_root: str,
):
    # Optional per-split augment configs:
    ds = cfg.dataset
    augment_train = getattr(ds, "augment_train", None)
    augment_vald = getattr(ds, "augment_vald", None)
    augment_test = getattr(ds, "augment_test", None)

    model = ConditionalPointCloudDiffusionModel(
        **cfg.model,
        part_to_vertex=part_to_vertex,
        subsample_mask=subsample_mask,
        smpl_root=smpl_root,

        # Bridge naming mismatch between dataset and model
        augment_train=augment_train,
        augment_val=augment_vald,
        augment_sample=augment_test,
    )
    return model