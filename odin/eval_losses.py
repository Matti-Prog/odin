import numpy as np
import torch
from pytorch3d.loss import chamfer_distance

def v2v_loss_torch(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return 100.0 * torch.norm(pred - gt, dim=-1).mean(dim=1)

def chamfer_loss_torch(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    cd, _ = chamfer_distance(gt, pred, batch_reduction=None)
    return cd

class LossMeter:
    def __init__(
        self,
        *,
        labels: dict[str, str] | None = None,
        metric_labels: dict[str, str] | None = None,
        descriptions: dict[str, str] | None = None,
        print_descriptions: bool = False,
    ):
        self.sums: dict[str, float] = {}
        self.counts: dict[str, int] = {}

        self.labels = labels or {}
        self.metric_labels = metric_labels or {}
        self.descriptions = descriptions or {}
        self.print_descriptions = print_descriptions

    def _fmt_name(self, name: str) -> str:
        return self.labels.get(name, name)

    def _fmt_metric(self, metric: str) -> str:
        return self.metric_labels.get(metric, metric)

    @torch.no_grad()
    def update(self, name: str, gt: torch.Tensor, pred: torch.Tensor):
        gt = gt.to(torch.float32)
        pred = pred.to(torch.float32)

        v2v = v2v_loss_torch(gt, pred).detach().cpu().numpy()
        ch  = chamfer_loss_torch(gt, pred).detach().cpu().numpy()

        base = self._fmt_name(name)
        k_v2v = f"{base}/{self._fmt_metric('v2v')}"
        k_ch  = f"{base}/{self._fmt_metric('ch')}"

        self.sums[k_v2v] = self.sums.get(k_v2v, 0.0) + float(v2v.sum())
        self.sums[k_ch]  = self.sums.get(k_ch, 0.0)  + float(ch.sum())
        self.counts[k_v2v] = self.counts.get(k_v2v, 0) + int(v2v.shape[0])
        self.counts[k_ch]  = self.counts.get(k_ch, 0)  + int(ch.shape[0])

    def summary(self) -> dict:
        out = {}
        for k, s in self.sums.items():
            c = self.counts.get(k, 0)
            out[k] = (s / c) if c > 0 else float("nan")
        return out

    def summary_lines(self) -> list[str]:
        avg = self.summary()

        by_prefix: dict[str, list[tuple[str, float]]] = {}
        for k, v in avg.items():
            prefix, metric = k.split("/", 1)
            by_prefix.setdefault(prefix, []).append((metric, v))

        order = [self._fmt_name(k) for k in ("pc", "v2v", "v2v_chamfer")]
        
        lines: list[str] = []
        
        for prefix in sorted(by_prefix.keys(), key=lambda p: (order.index(p) if p in order else 10**9, p)):
            if self.print_descriptions:
                desc = None
                for orig, d in self.descriptions.items():
                    if self._fmt_name(orig) == prefix:
                        desc = d
                        break
                if desc:
                    width = 85
                    title = f" {prefix} "
                    left = max(0, (width - len(title)) // 2)
                    right = max(0, width - len(title) - left)
                    lines.append("-" * left + title + "-" * right)
                    lines.append(desc)
            for metric, val in sorted(by_prefix[prefix], key=lambda x: x[0]):
                #lines.append(f"{prefix}/{metric}: {val:.6f}")
                lines.append(f"{metric}: {val:.6f}")
        return lines