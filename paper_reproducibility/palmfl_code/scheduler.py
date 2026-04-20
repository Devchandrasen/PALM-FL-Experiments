from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class DeviceProfile:
    client_id: int
    bandwidth_mbps: float
    compute_units: float
    battery_j: float
    max_battery_j: float
    tx_energy_per_mb: float
    compute_energy_per_step: float
    recharge_j_per_round: float
    availability: float


class MobileAwareScheduler:
    def __init__(
        self,
        num_clients: int,
        num_classes: int,
        scheduler_cfg: dict,
        seed: int,
    ) -> None:
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.cfg = scheduler_cfg
        self.rng = np.random.default_rng(seed)

        self.max_clients_per_round = None
        profile_csv = self.cfg.get("profile_csv")
        self.profiles = self._load_profiles(profile_csv) if profile_csv else self._sample_profiles()
        self.history: Dict[int, dict] = {
            cid: {
                "last_round": -1,
                "last_loss": 1.0,
                "last_accuracy": 0.0,
                "participations": 0,
                "num_samples": 0,
                "unique_labels": 0,
                "label_hist": [0.0 for _ in range(num_classes)],
                "model_size_mb": 1.0,
                "last_predicted_time": 0.0,
                "last_predicted_energy": 0.0,
            }
            for cid in range(num_clients)
        }

    @staticmethod
    def _float_from_row(row: dict, names: tuple[str, ...], default: float) -> float:
        for name in names:
            value = row.get(name)
            if value in (None, ""):
                continue
            try:
                return float(value)
            except ValueError:
                continue
        return float(default)

    def _load_profiles(self, profile_csv: str | Path) -> Dict[int, DeviceProfile]:
        path = Path(profile_csv).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"scheduler.profile_csv does not exist: {path}")

        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"scheduler.profile_csv is empty: {path}")

        by_client: dict[int, dict] = {}
        for row in rows:
            try:
                by_client[int(row.get("client_id", ""))] = row
            except ValueError:
                continue

        profiles: Dict[int, DeviceProfile] = {}
        for cid in range(self.num_clients):
            if cid in by_client:
                row = by_client[cid]
            else:
                row = rows[int(self.rng.integers(0, len(rows)))]

            max_battery = self._float_from_row(row, ("max_battery_j", "battery_capacity_j"), 220.0)
            battery = self._float_from_row(row, ("battery_j",), max_battery)
            profiles[cid] = DeviceProfile(
                client_id=cid,
                bandwidth_mbps=max(0.1, self._float_from_row(row, ("bandwidth_mbps", "bandwidth_Mbps"), 10.0)),
                compute_units=max(0.1, self._float_from_row(row, ("compute_units",), 3.0)),
                battery_j=max(0.0, battery),
                max_battery_j=max(1.0, max_battery),
                tx_energy_per_mb=max(0.0, self._float_from_row(row, ("tx_energy_per_mb",), 0.20)),
                compute_energy_per_step=max(0.0, self._float_from_row(row, ("compute_energy_per_step",), 0.010)),
                recharge_j_per_round=max(0.0, self._float_from_row(row, ("recharge_j_per_round",), 4.0)),
                availability=min(1.0, max(0.0, self._float_from_row(row, ("availability",), 0.95))),
            )
        return profiles

    def _sample_profiles(self) -> Dict[int, DeviceProfile]:
        profiles = {}
        for cid in range(self.num_clients):
            max_battery = float(self.rng.uniform(120.0, 300.0))
            profiles[cid] = DeviceProfile(
                client_id=cid,
                bandwidth_mbps=float(self.rng.uniform(5.0, 50.0)),
                compute_units=float(self.rng.uniform(1.0, 8.0)),
                battery_j=max_battery,
                max_battery_j=max_battery,
                tx_energy_per_mb=float(self.rng.uniform(0.08, 0.40)),
                compute_energy_per_step=float(self.rng.uniform(0.003, 0.02)),
                recharge_j_per_round=float(self.rng.uniform(2.0, 8.0)),
                availability=float(self.rng.uniform(0.75, 0.98)),
            )
        return profiles

    def register_client(self, client_id: int, num_samples: int, unique_labels: int, model_num_params: int) -> None:
        self.history[client_id]["num_samples"] = int(num_samples)
        self.history[client_id]["unique_labels"] = int(unique_labels)
        self.history[client_id]["model_size_mb"] = float(model_num_params * 4 / (1024**2))

    def estimate_costs(
        self,
        client_id: int,
        upload_mb: float,
        download_mb: float,
        local_steps: int,
    ) -> dict:
        profile = self.profiles[client_id]
        hist = self.history[client_id]
        model_size_mb = max(0.1, hist["model_size_mb"])

        upload_time = (upload_mb * 8.0) / max(profile.bandwidth_mbps, 1e-6)
        download_time = (download_mb * 8.0) / max(profile.bandwidth_mbps, 1e-6)
        compute_time = (local_steps * model_size_mb) / max(profile.compute_units, 1e-6) * 0.20
        total_time = upload_time + download_time + compute_time

        tx_energy = (upload_mb + download_mb) * profile.tx_energy_per_mb
        compute_energy = local_steps * model_size_mb * profile.compute_energy_per_step
        total_energy = tx_energy + compute_energy

        return {
            "upload_mb": upload_mb,
            "download_mb": download_mb,
            "predicted_time_s": total_time,
            "predicted_energy_j": total_energy,
        }

    def _utility(self, client_id: int, round_idx: int, costs: dict, class_deficit: np.ndarray | None = None) -> float:
        hist = self.history[client_id]
        profile = self.profiles[client_id]

        max_samples = max(h["num_samples"] for h in self.history.values())
        data_score = hist["num_samples"] / max(max_samples, 1)
        diversity_score = hist["unique_labels"] / max(self.num_classes, 1)
        deficit_score = 0.0
        if class_deficit is not None:
            client_hist = np.asarray(hist.get("label_hist", []), dtype=float)
            if client_hist.size == self.num_classes and client_hist.sum() > 0:
                client_dist = client_hist / max(float(client_hist.sum()), 1e-12)
                deficit = np.asarray(class_deficit, dtype=float)
                if deficit.sum() > 0:
                    deficit = deficit / max(float(deficit.sum()), 1e-12)
                deficit_score = float(np.dot(client_dist, deficit))
        staleness = round_idx - hist["last_round"] if hist["last_round"] >= 0 else round_idx + 1
        staleness_score = min(staleness / 5.0, 1.0)
        battery_score = profile.battery_j / max(profile.max_battery_j, 1e-6)
        loss_score = min(max(hist["last_loss"], 0.0) / 2.5, 1.0)

        utility_weights = self.cfg.get("utility_weights", {})
        data_w = float(utility_weights.get("data", 1.0))
        diversity_w = float(utility_weights.get("diversity", 0.8))
        deficit_w = float(utility_weights.get("deficit", 0.0))
        staleness_w = float(utility_weights.get("staleness", 0.3))
        battery_w = float(utility_weights.get("battery", 0.4))
        loss_w = float(utility_weights.get("loss", 0.3))

        uplink_budget = float(self.cfg.get("uplink_budget_mb", 10.0))
        downlink_budget = float(self.cfg.get("downlink_budget_mb", 10.0))
        energy_budget = float(self.cfg.get("energy_budget_j", 250.0))
        time_budget = float(self.cfg.get("round_time_budget_s", 200.0))

        cost_weights = self.cfg.get("cost_weights", {})
        upload_penalty = float(cost_weights.get("upload", 1.0)) * min(costs["upload_mb"] / max(uplink_budget, 1e-6), 2.0)
        download_penalty = float(cost_weights.get("download", 0.5)) * min(costs["download_mb"] / max(downlink_budget, 1e-6), 2.0)
        energy_penalty = float(cost_weights.get("energy", 0.8)) * min(costs["predicted_energy_j"] / max(energy_budget, 1e-6), 2.0)
        time_penalty = float(cost_weights.get("time", 0.8)) * min(costs["predicted_time_s"] / max(time_budget, 1e-6), 2.0)

        utility = (
            data_w * data_score
            + diversity_w * diversity_score
            + deficit_w * deficit_score
            + staleness_w * staleness_score
            + battery_w * battery_score
            + loss_w * loss_score
            - upload_penalty
            - download_penalty
            - energy_penalty
            - time_penalty
        )
        return float(utility)

    def select_clients(
        self,
        round_idx: int,
        max_clients: int,
        upload_mb: float,
        download_mb: float,
        local_steps: int,
        class_deficit: np.ndarray | None = None,
    ) -> tuple[List[int], Dict[int, dict]]:
        self.max_clients_per_round = max_clients
        candidates: List[Tuple[int, float, dict]] = []

        min_battery_fraction = float(self.cfg.get("min_battery_fraction", 0.15))
        for cid in range(self.num_clients):
            profile = self.profiles[cid]
            if self.rng.random() > profile.availability:
                continue
            if profile.battery_j / profile.max_battery_j < min_battery_fraction:
                continue

            costs = self.estimate_costs(cid, upload_mb=upload_mb, download_mb=download_mb, local_steps=local_steps)
            utility = self._utility(cid, round_idx, costs, class_deficit=class_deficit)
            if utility <= -10.0:
                continue
            candidates.append((cid, utility, costs))

        if not candidates:
            # Fallback: pick the most charged client.
            cid = max(self.profiles, key=lambda k: self.profiles[k].battery_j)
            costs = self.estimate_costs(cid, upload_mb=upload_mb, download_mb=download_mb, local_steps=local_steps)
            return [cid], {cid: costs}

        # Greedy selection under budgets by utility-to-cost ratio.
        uplink_budget = float(self.cfg.get("uplink_budget_mb", 10.0))
        downlink_budget = float(self.cfg.get("downlink_budget_mb", 10.0))
        energy_budget = float(self.cfg.get("energy_budget_j", 250.0))
        time_budget = float(self.cfg.get("round_time_budget_s", 200.0))

        def ratio(item: Tuple[int, float, dict]) -> float:
            _, utility, costs = item
            denom = 1.0
            denom += costs["upload_mb"] / max(uplink_budget, 1e-6)
            denom += costs["download_mb"] / max(downlink_budget, 1e-6)
            denom += costs["predicted_energy_j"] / max(energy_budget, 1e-6)
            denom += costs["predicted_time_s"] / max(time_budget, 1e-6)
            return utility / denom

        candidates.sort(key=ratio, reverse=True)

        selected: List[int] = []
        selected_info: Dict[int, dict] = {}
        used_up = 0.0
        used_down = 0.0
        used_energy = 0.0
        used_time = 0.0

        for cid, utility, costs in candidates:
            if len(selected) >= max_clients:
                break
            if used_up + costs["upload_mb"] > uplink_budget:
                continue
            if used_down + costs["download_mb"] > downlink_budget:
                continue
            if used_energy + costs["predicted_energy_j"] > energy_budget:
                continue
            if max(used_time, costs["predicted_time_s"]) > time_budget:
                continue
            selected.append(cid)
            selected_info[cid] = dict(costs)
            selected_info[cid]["utility"] = utility
            used_up += costs["upload_mb"]
            used_down += costs["download_mb"]
            used_energy += costs["predicted_energy_j"]
            used_time = max(used_time, costs["predicted_time_s"])

        if not selected:
            # Ensure progress even when budgets are too tight.
            cid, utility, costs = candidates[0]
            selected = [cid]
            selected_info = {cid: dict(costs)}
            selected_info[cid]["utility"] = utility

        return selected, selected_info

    def update_after_round(self, round_idx: int, updates: Dict[int, dict]) -> None:
        for cid, update in updates.items():
            hist = self.history[cid]
            profile = self.profiles[cid]
            hist["last_round"] = round_idx
            hist["last_loss"] = float(update.get("avg_loss", hist["last_loss"]))
            hist["last_accuracy"] = float(update.get("train_accuracy", hist["last_accuracy"]))
            hist["participations"] += 1
            hist["last_predicted_time"] = float(update.get("predicted_time_s", 0.0))
            hist["last_predicted_energy"] = float(update.get("predicted_energy_j", 0.0))
            if "counts" in update:
                counts = np.asarray(update["counts"], dtype=float)
                if counts.size == self.num_classes:
                    hist["label_hist"] = np.maximum(counts, 0.0).astype(float).tolist()

            profile.battery_j = max(0.0, profile.battery_j - float(update.get("predicted_energy_j", 0.0)))

        # Recharge all clients a bit after the round.
        for profile in self.profiles.values():
            profile.battery_j = min(profile.max_battery_j, profile.battery_j + profile.recharge_j_per_round)

    def state_dict(self) -> dict:
        return {
            "history": self.history,
            "profiles": {cid: vars(profile) for cid, profile in self.profiles.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self.history = state["history"]
        profiles = {}
        for cid, values in state["profiles"].items():
            profiles[int(cid)] = DeviceProfile(**values)
        self.profiles = profiles
