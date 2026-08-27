import datetime
import csv
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, List, Optional
import hydra
import torch
import wandb
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
import training_utils
from model import get_model, ConditionalPointCloudDiffusionModel
from config.structured import ProjectConfig
import json
import numpy as np
from data.dataset_resolver import resolve_dataloaders

torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()

@hydra.main(config_path='config', config_name='config', version_base='1.1')
def main(cfg: ProjectConfig): 
    assert Path(cfg.paths.project_root).exists()

    training_utils.set_seed(cfg.run.seed)
    
    # accelerator
    accelerator = Accelerator(mixed_precision=cfg.run.mixed_precision, cpu=cfg.run.cpu, 
        gradient_accumulation_steps=cfg.optimizer.gradient_accumulation_steps)

    # logging
    training_utils.setup_distributed_print(accelerator.is_main_process)
    if cfg.logging.wandb and accelerator.is_main_process:
        wandb.init(project=cfg.logging.wandb_project, 
                   name=cfg.run.name, 
                   job_type=cfg.run.job, 
                   config=OmegaConf.to_container(cfg),
                   entity=cfg.logging.wandb_entity)
        wandb.run.log_code(root=hydra.utils.get_original_cwd(),
            include_fn=lambda p: any(p.endswith(ext) for ext in ('.py', '.json', '.yaml', '.md', '.txt.', '.gin')),
            exclude_fn=lambda p: any(s in p for s in ('output', 'tmp', 'wandb', '.git', '.vscode')))
        cfg: ProjectConfig = DictConfig(wandb.config.as_dict())
    
    print(f'Current working directory: {os.getcwd()}') # outputs, checkpoints, logs saved here
    
    # model
    model: ConditionalPointCloudDiffusionModel = get_model(cfg,
                      part_to_vertex=json.load(open(cfg.assets.part_to_vertex)),
                      subsample_mask=np.load(cfg.assets.subsample_mask),
                      smpl_root=cfg.assets.smpl_root,
                      )

    # exponential moving average
    if cfg.ema.use_ema:
        from torch_ema import ExponentialMovingAverage
        model_ema = ExponentialMovingAverage(model.parameters(), decay=cfg.ema.decay)
        model_ema.to(accelerator.device)
        print('initialized model EMA')
    else:
        model_ema = None

    no_decay = ("bias", "LayerNorm.weight")
    base_lr = float(cfg.optimizer.lr)
    pn_mult = float(getattr(cfg.optimizer, "pointnet_lr_mult", 1.0)) # lr multiplier for PointNet++

    def make_groups(named_params, lr, wd):
        decay, nodecay = [], []
        for n, p in named_params:
            if not p.requires_grad:
                continue
            (nodecay if any(nd in n for nd in no_decay) else decay).append(p)
        groups = []
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": wd})
        if nodecay:
            groups.append({"params": nodecay, "lr": lr, "weight_decay": 0.0})
        return groups

    param_groups = []
    param_groups += make_groups(model.point_cloud_model.named_parameters(), base_lr, cfg.optimizer.weight_decay)
    param_groups += make_groups(model.pointnet.named_parameters(), base_lr * pn_mult, cfg.optimizer.weight_decay)

    optimizer: torch.optim.Optimizer = torch.optim.AdamW(param_groups, **cfg.optimizer.kwargs)
    scheduler = training_utils.get_scheduler(cfg, optimizer)

    #resume from checkpoint and create initial training state
    train_state: training_utils.TrainState = training_utils.resume_from_checkpoint(cfg, model, optimizer, scheduler, model_ema)

    # resolve dataset metadata against dataset config and instantiate dataloaders
    dataloader_train = None
    dataloader_val = None
    dataloader_test = None

    if cfg.dataset.type == "amass":
        loaders = resolve_dataloaders(cfg)
        dataloader_train = loaders.get("train", None)
        dataloader_val = loaders.get("vald", None)
        dataloader_test = loaders.get("test", None)

    elif cfg.dataset.type == "faust":
        loaders = resolve_dataloaders(cfg)
        dataloader_train = loaders["train"]
        dataloader_val = loaders["vald"]
        dataloader_test = loaders["test"]

    elif cfg.dataset.type == "dfaust":
        loaders = resolve_dataloaders(cfg)
        dataloader_train = loaders.get("train", None)
        dataloader_val = loaders.get("vald", None)
        dataloader_test = loaders.get("test", None)

    elif cfg.dataset.type == "shrec19":
        loaders = resolve_dataloaders(cfg)
        dataloader_test = loaders["test"]

    else:
        raise ValueError(f"Unsupported dataset type: {cfg.dataset.type}")
    
    # compute total training batch size
    total_batch_size = cfg.dataloader.batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps

    # prepare objects only if they exist
    job = str(cfg.run.job).lower()
    is_train = (job == "train")
    to_prepare = [model]

    if is_train:
        # in training, optimizer/scheduler and train loader must exist.
        to_prepare += [optimizer, scheduler, dataloader_train]
        if dataloader_val is not None:
            to_prepare.append(dataloader_val)
    else:
        if "dataloader_test" in locals() and dataloader_test is not None:
            to_prepare.append(dataloader_test)
        elif dataloader_val is not None:
            to_prepare.append(dataloader_val)

    prepared = accelerator.prepare(*to_prepare)
    
    it = iter(prepared)
    model = next(it)

    if is_train:
        optimizer = next(it)
        scheduler = next(it)
        dataloader_train = next(it)
        if dataloader_val is not None:
            dataloader_val = next(it)
    else:
        if "dataloader_test" in locals() and dataloader_test is not None:
            dataloader_test = next(it)
        elif dataloader_val is not None:
            dataloader_val = next(it)

    # sample
    if cfg.run.job == 'sample':
        # whether or not to use EMA parameters for sampling
        if cfg.run.sample_from_ema:
            assert model_ema is not None
            model_ema.to(accelerator.device)
            sample_context = model_ema.average_parameters
        else:
            sample_context = nullcontext
        with sample_context():
            sample_metrics = sample(
                cfg=cfg,
                model=model,
                dataloader = dataloader_test, # sample_split determines which split is used for sampling
                accelerator=accelerator,
            )
        if cfg.logging.wandb and accelerator.is_main_process:
            wandb.finish()
        time.sleep(5)
        return

    # Info
    print(f'***** Starting training at {datetime.datetime.now()} *****')
    print(f'    Dataset train size: {len(dataloader_train.dataset):_}')
    print(f'    Dataset val size: {len(dataloader_val.dataset):_}' if dataloader_val is not None else '    Dataset val size: None')
    print(f'    Dataloader train size: {len(dataloader_train):_}')
    print(f'    Dataloader val size: {len(dataloader_val):_}' if dataloader_val is not None else '    Dataloader val size: None')
    print(f'    Batch size per device = {cfg.dataloader.batch_size}')
    print(f'    Total train batch size (w. parallel, dist & accum) = {total_batch_size}')
    print(f'    Gradient Accumulation steps = {cfg.optimizer.gradient_accumulation_steps}')
    print(f'    Max training steps = {cfg.run.max_steps}')
    print(f'    Training state = {train_state}')

    # train loop
    while True:
        # train progress bar
        log_header = f'Epoch: [{train_state.epoch}]'
        metric_logger = training_utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('step', training_utils.SmoothedValue(window_size=1, fmt='{value:.0f}'))
        metric_logger.add_meter('lr', training_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        progress_bar: Iterable[Any] = metric_logger.log_every(dataloader_train, cfg.run.print_step_freq, 
            header=log_header)
        
        for i, batch in enumerate(progress_bar):            
            if (cfg.run.limit_train_batches is not None) and (i >= cfg.run.limit_train_batches): break
            model.train()

            # gradient accumulation
            with accelerator.accumulate(model):
                loss = model(batch, mode='train')
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    if cfg.optimizer.clip_grad_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.clip_grad_norm)
                    grad_norm_clipped = training_utils.compute_grad_norm(model.parameters())

                optimizer.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    scheduler.step()
                    train_state.step += 1

                # exit training if loss is NaN
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)

            if accelerator.sync_gradients:
                # logging
                log_dict = {
                    'lr': optimizer.param_groups[0]["lr"],
                    'step': train_state.step,
                    'train_loss': loss_value,
                    'grad_norm_clipped': grad_norm_clipped,
                }
                metric_logger.update(**log_dict)
                if (cfg.logging.wandb and accelerator.is_main_process and train_state.step % cfg.run.log_step_freq == 0):
                    wandb.log(log_dict, step=train_state.step)
            
                # update EMA
                if cfg.ema.use_ema and train_state.step % cfg.ema.update_every == 0:
                    model_ema.update(model.parameters())

                # checkpoint saving
                if accelerator.is_main_process and (train_state.step % cfg.run.checkpoint_freq == 0):
                    checkpoint_dict = {
                        'model': accelerator.unwrap_model(model).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': train_state.epoch,
                        'step': train_state.step,
                        'best_val': train_state.best_val,
                        'model_ema': model_ema.state_dict() if model_ema else {},
                        'cfg': cfg
                    }
                    checkpoint_path = 'checkpoint-latest.pth'
                    accelerator.save(checkpoint_dict, checkpoint_path)
                    print(f'Saved checkpoint to {Path(checkpoint_path).resolve()}')
                
                # validate
                if (dataloader_val is not None and len(dataloader_val) > 0 and cfg.run.val_freq > 0 and train_state.step % cfg.run.val_freq == 0):
                    validate(
                        cfg=cfg,
                        model=model,
                        dataloader_val=dataloader_val,
                        accelerator=accelerator,
                        num_batches=None,
                        step = train_state.step,
                    )

                # end training after set number of steps
                if train_state.step >= cfg.run.max_steps:
                    print(f'Ending training at: {datetime.datetime.now()}')
                    print(f'Final train state: {train_state}')
                    wandb.finish()
                    time.sleep(5)
                    return

        # epoch complete; log and continue training
        train_state.epoch += 1

        # gather stats from all processes
        metric_logger.synchronize_between_processes(device=accelerator.device)
        print(f'{log_header}  Average stats --', metric_logger)


@torch.no_grad()
def validate(
    *,
    cfg: ProjectConfig,
    model: torch.nn.Module,
    dataloader_val: Iterable,
    accelerator: Accelerator,
    num_batches: Optional[int] = None,
    step: Optional[int] = None,
):
    model.eval()

    save_val_samples = bool(getattr(cfg.run, "save_val_samples", False))
    if save_val_samples:
        run_dir = Path(os.getcwd())
        step_str = f"step_{step}" if step is not None else "unknown_step"
        output_dir = run_dir / "val_samples" / step_str
        output_dir.mkdir(parents=True, exist_ok=True)
    
    metric_logger_val = training_utils.MetricLogger(delimiter="  ")
    progress_bar_val = metric_logger_val.log_every(dataloader_val, cfg.run.print_step_freq, "Val Chamfer")
    
    for batch_idx, batch in enumerate(progress_bar_val):
        if num_batches is not None and batch_idx >= num_batches:
            break

        if save_val_samples:
            loss, v2v_val, gt_pc, gt_pc_full, scan_pc, pred_pc = model(
                batch, mode="validate",
                return_outputs=True,
            )

            first = next(iter(batch.values()))
            B = first.shape[0] if torch.is_tensor(first) else len(first)
            for i in range(B):
                sample = {
                    "batch": batch,
                    "index": i,
                    "gt_pc": gt_pc[i],
                    "gt_pc_full": gt_pc_full[i] if gt_pc_full is not None else gt_pc[i],
                    "scan_pc": scan_pc[i],
                    "pred_pc": pred_pc[i],
                    "tag": "pred",
                }
                dataloader_val.dataset.save_sample(sample=sample, output_dir=output_dir)
        else:
            loss, v2v_val = model(batch, mode="validate")
        
        loss_value = loss.item()
        v2v_value = v2v_val.item()
        metric_logger_val.update(val_chamfer_loss=loss_value, val_v2v_loss=v2v_value)
    
    metric_logger_val.synchronize_between_processes(device=accelerator.device)
    avg_val_loss = metric_logger_val.meters['val_chamfer_loss'].global_avg

    if cfg.logging.wandb and accelerator.is_main_process:
        wandb.log({
            'val_chamfer_loss': avg_val_loss,
            'val_v2v_loss': metric_logger_val.meters['val_v2v_loss'].global_avg
        }, step=step)
    return

def sample(
    *,
    cfg: ProjectConfig,
    model: torch.nn.Module,
    dataloader: Iterable,
    accelerator: Accelerator,
    output_dir: str = "sample",
):
    from pytorch3d.implicitron.dataset.data_loader_map_provider import FrameData
    from tqdm import tqdm
    import numpy as np
    import torch
    from pathlib import Path
    from pytorch3d.loss import chamfer_distance

    model.eval()

    progress_bar: Iterable[FrameData] = tqdm(dataloader, disable=(not accelerator.is_main_process))

    num_samples = int(cfg.run.num_samples)
    save_mode = str(cfg.run.save_mode).lower()  # "all" or "best"
    save_evol = bool(cfg.run.sample_save_evolutions)
    chamfer_mode = int(cfg.run.chamfer_mode)  # 0: bidirectional, 1: scan->pred, -1: pred->scan

    output_dir = Path(output_dir)

    # name subdir based on save mode and number of samples
    if num_samples <= 1:
        sample_dir_name = "sample"
    elif save_mode == "best":
        sample_dir_name = f"sample_best_of_{num_samples}"
    else:
        sample_dir_name = f"sample_{num_samples}"

    output_dir = output_dir.parent / sample_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    compute_losses_requested = (save_mode == "best") # do not compute losses in case of multiple samples per input

    loss_meter = None
    runner = None

    if compute_losses_requested:
        from eval_losses import LossMeter
        loss_meter = LossMeter(
            labels={
                "pc": "Diffusion Sample",
                "v2v": "Stage I SMPL Fitting",
                "v2v_chamfer": "Stage II SMPL Fitting",
            },
            metric_labels={
                "v2v": "Average Vertex-to-Vertex Loss \\[cm]",
                "ch": "Chamfer Loss",
            },
            descriptions={
                "pc": "Diffusion sample vs subsampled GT point cloud (689 vertices)",
                "v2v": "SMPL fit to diffusion sample via smplfitter vs full GT point cloud (6890 vertices)",
                "v2v_chamfer": "SMPL refined to scan via Chamfer objective vs full GT point cloud (6890 vertices)",
            },
            print_descriptions=True,
        )

    from smpl_fitting import SMPLFittingRunner
    runner = SMPLFittingRunner(cfg, device=accelerator.device)

    # get SMPL faces to save SMPL GT and/or SMPL fit point clouds as full meshes.
    faces_np = None
    if hasattr(runner, "body_model_native") and hasattr(runner.body_model_native, "faces"):
        faces_np = np.asarray(runner.body_model_native.faces)

    for batch_idx, batch in enumerate(progress_bar):
        progress_bar.set_description(f"Processing batch {batch_idx:4d} / {len(dataloader):4d}")
        if cfg.run.num_sample_batches is not None and batch_idx >= cfg.run.num_sample_batches:
            break

        first = next(iter(batch.values()))
        B = first.shape[0] if torch.is_tensor(first) else len(first) # batch size

        if chamfer_mode not in (-1, 0, 1):
            raise ValueError(f"cfg.run.chamfer_mode must be one of [-1, 0, 1], got {chamfer_mode}")

        # best-of-k buffers (used only when save_mode == "best" and num_samples > 1)
        if save_mode == "best" and num_samples > 1:
            best_chamfer = torch.full((B,), float("inf"), device="cpu")
            best_pred = [None] * B
            best_evol = [None] * B

        # keep last outputs for single-gen best-mode fallback
        last_gt_pc = None
        last_scan_pc = None
        last_gt_pc_full = None
        last_output = None
        last_all_outputs = None

        for sample_idx in range(num_samples):
            # generate one candidate
            gt_pc, gt_pc_full, scan_pc, output, all_outputs = model(
                batch,
                mode="sample",
                return_sample_every_n_steps=1,
                scheduler=cfg.run.diffusion_scheduler,
                num_inference_steps=cfg.run.num_inference_steps,
                disable_tqdm=(not accelerator.is_main_process),
            )

            last_gt_pc = gt_pc
            last_gt_pc_full = gt_pc_full
            last_scan_pc = scan_pc
            last_output = output
            last_all_outputs = all_outputs

            # update best-of-k (according to Chamfer)
            if save_mode == "best" and num_samples > 1:
                generated_points = output.cuda()
                scan_points = scan_pc.cuda()

                if chamfer_mode == 0:
                    cd_loss, _ = chamfer_distance(generated_points, scan_points, batch_reduction=None)
                elif chamfer_mode == 1:
                    cd_loss, _ = chamfer_distance(
                        scan_points, generated_points, batch_reduction=None, single_directional=True
                    )
                else:  # chamfer_mode == -1
                    cd_loss, _ = chamfer_distance(
                        generated_points, scan_points, batch_reduction=None, single_directional=True
                    )

                cd_loss = cd_loss.detach().cpu()

                for i in range(B):
                    if cd_loss[i].item() < best_chamfer[i].item():
                        best_chamfer[i] = cd_loss[i]
                        best_pred[i] = output[i].detach().clone()
                        if save_evol:
                            best_evol[i] = all_outputs[i]

            # if we want all samples, we save them at every round
            if save_mode == "all":
                tag = f"pred-{sample_idx}" if num_samples > 1 else "pred"
                for i in range(B):
                    sample_item = {
                        "batch": batch,
                        "index": i,
                        "scan_pc": scan_pc[i],
                        "pred_pc": output[i],
                        "tag": tag,
                        "smpl_faces": faces_np
                    }
                    if gt_pc is not None:
                        sample_item["gt_pc"] = gt_pc[i]
                    if gt_pc_full is not None:
                        sample_item["gt_pc_full"] = gt_pc_full[i]
                    if save_evol:
                        sample_item["evolution"] = all_outputs[i]
                    dataloader.dataset.save_sample(sample=sample_item, output_dir=output_dir)

        # if we want the best of k, we save the best once after k rounds
        if save_mode == "best":
            if num_samples > 1:
                for i in range(B):
                    if best_pred[i] is None:
                        continue
                    sample_item = {
                        "batch": batch,
                        "index": i,
                        "scan_pc": last_scan_pc[i],
                        "pred_pc": best_pred[i],
                        "tag": "pred",
                        "smpl_faces": faces_np
                    }
                    if last_gt_pc is not None:
                        sample_item["gt_pc"] = last_gt_pc[i]
                    if last_gt_pc_full is not None:
                        sample_item["gt_pc_full"] = last_gt_pc_full[i]
                    if save_evol:
                        sample_item["evolution"] = best_evol[i]
                    dataloader.dataset.save_sample(sample=sample_item, output_dir=output_dir)
            else:
                # single-sample fallback
                for i in range(B):
                    sample_item = {
                        "batch": batch,
                        "index": i,
                        "scan_pc": last_scan_pc[i],
                        "pred_pc": last_output[i],
                        "tag": "pred",
                        "smpl_faces": faces_np
                    }
                    if last_gt_pc is not None:
                        sample_item["gt_pc"] = last_gt_pc[i]
                    if last_gt_pc_full is not None:
                        sample_item["gt_pc_full"] = last_gt_pc_full[i]
                    if save_evol:
                        sample_item["evolution"] = last_all_outputs[i]
                    dataloader.dataset.save_sample(sample=sample_item, output_dir=output_dir)

        # SMPL fitting
        fit_res = {}
        has_gt = (last_gt_pc is not None) and (last_gt_pc_full is not None)

        # select pred for fitting: best-of-k or single-gen
        if save_mode == "best" and num_samples > 1:
            pred_fit = torch.stack([p for p in best_pred], dim=0)
        else:
            pred_fit = last_output

        scan_fit = last_scan_pc
        fit_res = runner.fit(pred_pc_689=pred_fit, scan_pc=scan_fit)

        # compute losses only if requested AND ground truth exists
        compute_losses = bool(compute_losses_requested and has_gt and (loss_meter is not None))
        if compute_losses:
            gt_eval = last_gt_pc
            gt_eval_full = last_gt_pc_full

            loss_meter.update("pc", gt=gt_eval, pred=pred_fit)

            if "out_v2v" in fit_res:
                v2v_verts = torch.from_numpy(fit_res["out_v2v"]).to(gt_eval.device)
                loss_meter.update("v2v", gt=gt_eval_full, pred=v2v_verts)

            if "out_cham" in fit_res:
                cham_verts = torch.from_numpy(fit_res["out_cham"]).to(gt_eval.device)
                loss_meter.update("v2v_chamfer", gt=gt_eval_full, pred=cham_verts)

        out_v2v_np = fit_res.get("out_v2v", None)
        out_cham_np = fit_res.get("out_cham", None)
        params_v2v = fit_res.get("params_v2v", None)
        params_cham = fit_res.get("params_cham", None)

        for i in range(B):
            s = {
                "batch": batch,
                "index": i,
                "smpl_faces": faces_np,
            }

            if out_v2v_np is not None:
                s["smpl_fit_stage_i"] = out_v2v_np[i]
            if out_cham_np is not None:
                s["smpl_fit_stage_ii"] = out_cham_np[i]

            if params_v2v is not None:
                s["smpl_fit_params_stage_i"] = {
                    "pose": params_v2v["pose"][i],
                    "beta": params_v2v["beta"][i],
                    "trans": params_v2v["trans"][i],
                    "scale": params_v2v["scale"][i],
                }

            if params_cham is not None:
                loss_val = params_cham.get("loss", None)
                loss_i = loss_val[i] if (torch.is_tensor(loss_val) and loss_val.ndim > 0) else loss_val
                s["smpl_fit_params_stage_ii"] = {
                    "loss": loss_i,
                    "pose": params_cham["pose"][i],
                    "beta": params_cham["beta"][i],
                    "trans": params_cham["trans"][i],
                    "scale": params_cham["scale"][i],
                }

            dataloader.dataset.save_sample(sample=s, output_dir=output_dir)

    print("Saved samples to:")
    print(output_dir.absolute())

    if compute_losses_requested and loss_meter is not None and accelerator.is_main_process:
        for line in loss_meter.summary_lines():
            print(line)
        summary = loss_meter.summary()
        return {
            "diffusion_v2v_cm": summary["Diffusion Sample/Average Vertex-to-Vertex Loss \\[cm]"],
            "diffusion_chamfer": summary["Diffusion Sample/Chamfer Loss"],
            "stage_i_v2v_cm": summary["Stage I SMPL Fitting/Average Vertex-to-Vertex Loss \\[cm]"],
            "stage_i_chamfer": summary["Stage I SMPL Fitting/Chamfer Loss"],
            "stage_ii_v2v_cm": summary["Stage II SMPL Fitting/Average Vertex-to-Vertex Loss \\[cm]"],
            "stage_ii_chamfer": summary["Stage II SMPL Fitting/Chamfer Loss"],
        }

    return {}

if __name__ == '__main__':
    main()
