import os
from pathlib import Path
import warnings
import yaml

# Resolve dataset metadata against config and construct dataloaders

# DFAUST
def find_latest_dfaust_meta(root: str) -> str:
    root_p = Path(root).expanduser().resolve()
    hydra_runs = root_p / "hydra_runs"
    if not hydra_runs.exists():
        raise FileNotFoundError(f"[DFAUST] Missing hydra_runs folder: {hydra_runs}")

    candidates = list(hydra_runs.glob("**/dataset_meta.yaml")) + list(hydra_runs.glob("**/dataset_meta.yml"))
    if not candidates:
        raise FileNotFoundError(f"[DFAUST] No dataset_meta.yaml found under: {hydra_runs}")

    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(newest)


def resolve_dfaust_dataloader(cfg):
    """
    DFAUST is point-cloud only.
    Uses the split requested by the job:
      - train job: train loader required, val loader optional
      - sample job: cfg.run.sample_split must exist and is exposed as loaders['test']
    """
    if cfg.run.job not in ("train", "sample"):
        raise ValueError(f"[DFAUST] Unsupported run.job='{cfg.run.job}' (expected 'train' or 'sample').")

    meta_path = find_latest_dfaust_meta(cfg.dataset.root)
    with open(meta_path, "r") as f:
        meta = yaml.safe_load(f)

    d = meta.get("dataset", {})
    splits_cfg = d.get("splits", {}) or {}

    available_splits = {
        s: bool((splits_cfg.get(s, {}) or {}).get("enabled", False))
        for s in ("train", "val", "test")
    }

    pc_root = os.path.join(str(cfg.dataset.root), "point_clouds")

    print(f"[DFAUST] manifest: {meta_path}")
    print(f"[DFAUST] point_clouds_dir: {pc_root}")
    print(f"[DFAUST] available splits: {available_splits}")

    from . import dataloader_dfaust

    loaders = {}

    def _make_loader(split: str, shuffle: bool):
        if not available_splits[split]:
            return None
        return dataloader_dfaust.create_dataloader(
            root=pc_root,
            batch_size=cfg.dataloader.batch_size,
            num_workers=cfg.dataloader.num_workers,
            shuffle=shuffle,
            mode=split,
        )

    if cfg.run.job == "train":
        loaders["train"] = _make_loader("train", shuffle=True)
        if loaders["train"] is None:
            raise RuntimeError("[DFAUST] Training requires split 'train', but it is unavailable in the manifest.")

        loaders["vald"] = _make_loader("val", shuffle=False)
        if loaders["vald"] is None:
            print("[DFAUST][WARN] No validation split available; proceeding without val dataloader.")

        loaders["test"] = None

    else:  # sample
        sample_split = str(getattr(cfg.run, "sample_split", "test"))
        if sample_split not in ("train", "val", "test"):
            raise ValueError(f"[DFAUST] Invalid sample_split='{sample_split}' (expected train, val, test).")

        loader = _make_loader(sample_split, shuffle=False)
        if loader is None:
            raise RuntimeError(
                f"[DFAUST] Sampling requested split='{sample_split}', but it is unavailable per manifest."
            )

        loaders["train"] = None
        loaders["vald"] = None
        loaders["test"] = loader

    loaders["_dfaust_meta_path"] = meta_path
    loaders["_dfaust_available_splits"] = available_splits
    return loaders

# SHREC19
def find_latest_shrec19_meta(root: str) -> str:
    root_p = Path(root).expanduser().resolve()
    hydra_runs = root_p / "hydra_runs"
    if not hydra_runs.exists():
        raise FileNotFoundError(f"[SHREC19] Missing hydra_runs folder: {hydra_runs}")

    candidates = list(hydra_runs.glob("**/dataset_meta.yaml")) + list(hydra_runs.glob("**/dataset_meta.yml"))
    if not candidates:
        raise FileNotFoundError(f"[SHREC19] No dataset_meta.yaml found under: {hydra_runs}")

    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(newest)


def resolve_shrec19_dataloader(cfg):
    """
    SHREC19 has scans only (no GT), so only support run.job == 'sample'.
    Uses the single 'train' split as the only split, for minimal dataset convention.
    """
    if cfg.run.job != "sample":
        raise RuntimeError("[SHREC19] Only run.job='sample' is supported (dataset has no GT).")

    meta_path = find_latest_shrec19_meta(cfg.dataset.root)
    with open(meta_path, "r") as f:
        meta = yaml.safe_load(f)

    d = meta.get("dataset", {})
    splits_cfg = d.get("splits", {}) or {}
    train_cfg = (splits_cfg.get("train", {}) or {})
    train_enabled = bool(train_cfg.get("enabled", False))
    if not train_enabled:
        raise RuntimeError(f"[SHREC19] Manifest indicates splits.train.enabled=false (meta={meta_path}).")

    pc_root = os.path.join(str(cfg.dataset.root), "point_clouds")

    print(f"[SHREC19] manifest: {meta_path}")
    print(f"[SHREC19] point_clouds_dir: {pc_root}")
    print("[SHREC19] using split: train (only)")

    from . import dataloader_shrec19
    loader = dataloader_shrec19.create_dataloader(
        root=pc_root,
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.num_workers,
        shuffle=False,
        mode="train",
    )
    return {"train": None, "vald": None, "test": loader, "_shrec19_meta_path": meta_path}

# FAUST
def find_latest_faust_meta(root: str) -> str:
    root_p = Path(root).expanduser().resolve()
    hydra_runs = root_p / "hydra_runs"
    if not hydra_runs.exists():
        raise FileNotFoundError(f"[FAUST] Missing hydra_runs folder: {hydra_runs}")

    candidates = list(hydra_runs.glob("**/dataset_meta.yaml")) + list(hydra_runs.glob("**/dataset_meta.yml"))
    if not candidates:
        raise FileNotFoundError(f"[FAUST] No dataset_meta.yaml found under: {hydra_runs}")

    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(newest)

def resolve_faust_dataloader(cfg):
    """
    Use train split, because it has GT.
    """
    meta_path = find_latest_faust_meta(cfg.dataset.root)
    with open(meta_path, "r") as f:
        meta = yaml.safe_load(f)

    d = meta.get("dataset", {})
    splits_cfg = d.get("splits", {}) or {}

    train_cfg = (splits_cfg.get("train", {}) or {})
    train_enabled = bool(train_cfg.get("enabled", False))
    if not train_enabled:
        raise RuntimeError(
            f"[FAUST] Manifest indicates splits.train.enabled=false (meta={meta_path}). "
            "This simplified resolver requires the FAUST train split."
        )

    pc_root = os.path.join(str(cfg.dataset.root), "point_clouds")

    print(f"[FAUST] manifest: {meta_path}")
    print(f"[FAUST] point_clouds_dir: {pc_root}")
    print("[FAUST] using split: train (only)")

    from . import dataloader_faust
    loader = dataloader_faust.create_dataloader(
        root=pc_root,
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.num_workers,
        shuffle=True,
        mode="train",
    )
    return {
        "train": loader,
        "vald": None,
        "test": loader,
        "_faust_meta_path": meta_path,
    }

# AMASS
def find_latest_amass_meta(root: str) -> str:
    root_p = Path(root).expanduser().resolve()
    hydra_runs = root_p / "hydra_runs"
    if not hydra_runs.exists():
        raise FileNotFoundError(f"[AMASS] Missing hydra_runs folder: {hydra_runs}")

    candidates = list(hydra_runs.glob("**/dataset_meta.yaml")) + list(hydra_runs.glob("**/dataset_meta.yml"))
    if not candidates:
        raise FileNotFoundError(f"[AMASS] No dataset_meta.yaml found under: {hydra_runs}")

    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(newest)


def _resolve_amass_data_path(root: str, recorded_path, expected_name: str) -> str:
    """Resolve new relative metadata and remain compatible with legacy manifests."""
    root_path = Path(root).expanduser().resolve()

    if recorded_path is None:
        return str(root_path / expected_name)

    path = Path(str(recorded_path)).expanduser()
    if not path.is_absolute():
        return str(root_path / path)

    if path.exists():
        return str(path)

    fallback = root_path / expected_name
    if fallback.exists():
        warnings.warn(
            f"[AMASS] Manifest path does not exist: {path}. "
            f"Using path relative to dataset.root instead: {fallback}",
            stacklevel=2,
        )
        return str(fallback)

    return str(path)


def resolve_amass_dataloaders(cfg) -> dict:
    if cfg.run.job not in ("train", "sample"):
        raise ValueError(f"[AMASS] Unsupported run.job='{cfg.run.job}' (expected 'train' or 'sample').")

    meta_path = find_latest_amass_meta(cfg.dataset.root)
    with open(meta_path, "r") as f:
        meta = yaml.safe_load(f)

    d = meta.get("dataset", {})
    splits_avail = d.get("splits", {}) or {}
    pc_splits_avail = d.get("point_cloud_splits", {}) or {}
    pc_root = _resolve_amass_data_path(
        cfg.dataset.root, d.get("point_clouds_dir"), "point_clouds"
    )
    smpl_data_root = _resolve_amass_data_path(
        cfg.dataset.root, d.get("stage_II_dir"), "stage_II"
    )

    available_splits = {s: bool(splits_avail.get(s, False)) for s in ("train", "vald", "test")}
    available_pc_splits = {s: bool(pc_splits_avail.get(s, False)) for s in ("train", "vald", "test")}

    print(f"[AMASS] manifest: {meta_path}")
    print(f"[AMASS] available splits (SMPL): {available_splits}")
    print(f"[AMASS] available splits (PC):   {available_pc_splits}")
    print(f"[AMASS] point_clouds_dir: {pc_root}")

    loaders = {}

    def _make_loader(split: str, shuffle: bool):
        # split must at least have SMPL data; point clouds optional
        if not available_splits[split]:
            return None

        # decide if PC are used
        want_pc = getattr(cfg.dataset.use_pc_if_available, split)
        use_pc = want_pc and available_pc_splits[split]

        if use_pc:
            if pc_root is None:
                raise RuntimeError(f"[AMASS] PC requested for split='{split}' but manifest has no point_clouds_dir.")
            from . import dataloader_amass_pointcloud
            return dataloader_amass_pointcloud.create_dataloader(
                root=pc_root,
                batch_size=cfg.dataloader.batch_size,
                num_workers=cfg.dataloader.num_workers,
                shuffle=shuffle,
                mode=split,
            )
        else:
            if smpl_data_root is None:
                raise RuntimeError(f"[AMASS] SMPL requested for split='{split}' but manifest has no stage_II_dir.")
            from . import dataloader_amass_smpl
            return dataloader_amass_smpl.create_dataloader(
                h5_path=smpl_data_root,
                mode=split,
                batch_size=cfg.dataloader.batch_size,
                shuffle=shuffle,
                num_workers=cfg.dataloader.num_workers,
            )

    if cfg.run.job == "train":
        loaders["train"] = _make_loader("train", shuffle=True)
        if loaders["train"] is None:
            raise RuntimeError("[AMASS] Training requires a train dataloader, but split 'train' is unavailable in manifest.")

        loaders["vald"] = _make_loader("vald", shuffle=False)
        if loaders["vald"] is None:
            print("[AMASS][WARN] No validation split available; proceeding without val dataloader.")

    else:  # sample
        sample_split = getattr(cfg.run, "sample_split", "test")
        if sample_split not in ("train", "vald", "test"):
            raise ValueError(f"[AMASS] Invalid sample_split='{sample_split}' (expected train, vald, test).")

        loaders[sample_split] = _make_loader(sample_split, shuffle=False)
        if loaders[sample_split] is None:
            raise RuntimeError(
                f"[AMASS] Sampling requested split='{sample_split}', but it is unavailable per manifest/config."
            )

        loaders["test"] = loaders[sample_split]

    # attach meta info (optional)
    loaders["_amass_meta_path"] = meta_path
    loaders["_amass_available_splits"] = available_splits
    loaders["_amass_available_pc_splits"] = available_pc_splits
    return loaders

def resolve_dataloaders(cfg) -> dict:
    dataset_type = str(cfg.dataset.type)

    if dataset_type == "amass":
        return resolve_amass_dataloaders(cfg)

    if dataset_type == "dfaust":
        return resolve_dfaust_dataloader(cfg)

    if dataset_type == "faust":
        return resolve_faust_dataloader(cfg)

    if dataset_type == "shrec19":
        return resolve_shrec19_dataloader(cfg)

    raise ValueError(f"Unsupported dataset.type='{dataset_type}'")