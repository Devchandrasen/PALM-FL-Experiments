from __future__ import annotations

from typing import Iterable

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    normalize_latent: bool = False,
    max_batches: int | None = None,
) -> dict:
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    total = 0
    correct = 0
    running_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x, normalize_latent=normalize_latent)
        loss = criterion(logits, y)

        preds = logits.argmax(dim=-1)
        total += y.numel()
        correct += (preds == y).sum().item()
        running_loss += float(loss.item()) * y.size(0)

        for t, p in zip(y.view(-1), preds.view(-1)):
            confusion[int(t.item()), int(p.item())] += 1

    if total == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "loss": 0.0}

    tp = confusion.diag().float()
    fp = confusion.sum(dim=0).float() - tp
    fn = confusion.sum(dim=1).float() - tp
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-12)
    macro_f1 = float(f1.mean().item())

    return {
        "accuracy": float(correct / total),
        "macro_f1": macro_f1,
        "loss": float(running_loss / total),
    }


def summarize_metric_list(metrics: Iterable[dict]) -> dict:
    metrics = list(metrics)
    if not metrics:
        return {}
    summary = {}
    for key in metrics[0]:
        values = [float(m[key]) for m in metrics]
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        summary[key] = mean
        summary[f"{key}_std"] = var**0.5
    return summary
