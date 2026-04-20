from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import math
import torch


def clip_tensor_by_l2(x: torch.Tensor, max_norm: float) -> torch.Tensor:
    norm = x.norm(p=2)
    if norm <= max_norm or max_norm <= 0:
        return x
    return x * (max_norm / (norm + 1e-12))


@dataclass
class StatsDPMechanism:
    clip_norm: float
    count_clip: float
    noise_multiplier: float
    enabled: bool = True
    releases_per_round: int = 3
    count_threshold: float = 1.0

    def privatize(
        self,
        means: torch.Tensor,
        variances: torch.Tensor,
        counts: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        means = clip_tensor_by_l2(means, self.clip_norm)
        variances = clip_tensor_by_l2(torch.clamp(variances, min=1e-6), self.clip_norm)
        counts = clip_tensor_by_l2(torch.clamp(counts, min=0.0), self.count_clip)

        if self.enabled and self.noise_multiplier > 0:
            means = means + torch.randn_like(means) * (self.noise_multiplier * self.clip_norm)
            variances = variances + torch.randn_like(variances) * (self.noise_multiplier * self.clip_norm)
            counts = counts + torch.randn_like(counts) * (self.noise_multiplier * self.count_clip)

        counts = torch.clamp(counts, min=0.0)
        variances = torch.clamp(variances, min=1e-6)
        mask = counts >= float(self.count_threshold)
        return {
            "means": means,
            "variances": variances,
            "counts": counts,
            "mask": mask,
        }


@dataclass
class GaussianZCDPAccountant:
    noise_multiplier: float
    delta: float
    releases_per_round: int = 3
    enabled: bool = True
    participations: Dict[int, int] = field(default_factory=dict)

    def step(self, client_id: int) -> None:
        if not self.enabled:
            return
        self.participations[client_id] = self.participations.get(client_id, 0) + 1

    def epsilon(self, client_id: int) -> float:
        if not self.enabled:
            return 0.0
        if self.noise_multiplier <= 0:
            return float("inf")
        rounds = self.participations.get(client_id, 0)
        if rounds == 0:
            return 0.0
        rho = rounds * self.releases_per_round / (2.0 * self.noise_multiplier**2)
        return rho + 2.0 * math.sqrt(rho * math.log(1.0 / self.delta))

    def summary(self) -> dict:
        if not self.participations:
            return {"max_epsilon": 0.0, "mean_epsilon": 0.0, "num_participating_clients": 0}
        eps = [self.epsilon(cid) for cid in self.participations]
        return {
            "max_epsilon": float(max(eps)),
            "mean_epsilon": float(sum(eps) / len(eps)),
            "num_participating_clients": int(len(eps)),
        }
