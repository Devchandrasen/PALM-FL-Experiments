from __future__ import annotations

import argparse
import csv
import math
import statistics
import urllib.request
from pathlib import Path


RAW_BASE = "https://raw.githubusercontent.com/mobilebandwidth/mobilebandwidth.github.io/main/data"
TRACE_FILES = ["4G.csv", "5G.csv", "wifi4.csv", "wifi5.csv", "wifi6.csv"]


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with urllib.request.urlopen(url, timeout=60) as response:
        path.write_bytes(response.read())


def safe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except ValueError:
        return None
    if not math.isfinite(x):
        return None
    return x


def load_records(raw_dir: Path) -> list[dict[str, str | float]]:
    records: list[dict[str, str | float]] = []
    for filename in TRACE_FILES:
        path = raw_dir / filename
        source = filename.removesuffix(".csv")
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                bandwidth = safe_float(row.get("bandwidth_Mbps"))
                if bandwidth is None or bandwidth <= 0.1 or bandwidth > 2000:
                    continue
                records.append(
                    {
                        "source": source,
                        "brand": row.get("brand", ""),
                        "model": row.get("model", ""),
                        "network_type": row.get("network_type", source),
                        "bandwidth_mbps": float(bandwidth),
                    }
                )
    return records


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
    return float(values[idx])


def choose_by_quantile(records: list[dict[str, str | float]], source: str, q: float) -> dict[str, str | float]:
    subset = [r for r in records if str(r["source"]) == source]
    if not subset:
        subset = records
    subset = sorted(subset, key=lambda r: float(r["bandwidth_mbps"]))
    idx = min(len(subset) - 1, max(0, round(q * (len(subset) - 1))))
    return subset[idx]


def build_profiles(records: list[dict[str, str | float]], num_clients: int) -> list[dict[str, str | float]]:
    sources = ["4G", "4G", "4G", "5G", "5G", "wifi4", "wifi5", "wifi5", "wifi6", "wifi6"]
    quantiles = [0.05, 0.15, 0.35, 0.25, 0.65, 0.20, 0.35, 0.70, 0.45, 0.85]
    source_compute = {"4G": 2.0, "5G": 4.5, "wifi4": 3.0, "wifi5": 5.0, "wifi6": 6.5}
    source_tx = {"4G": 0.30, "5G": 0.22, "wifi4": 0.12, "wifi5": 0.10, "wifi6": 0.08}

    profiles: list[dict[str, str | float]] = []
    for cid in range(num_clients):
        source = sources[cid % len(sources)]
        q = quantiles[cid % len(quantiles)]
        rec = choose_by_quantile(records, source, q)
        bandwidth = float(rec["bandwidth_mbps"])
        max_battery = 150.0 + 15.0 * cid
        battery_frac = 0.62 + 0.035 * (cid % 9)
        profiles.append(
            {
                "client_id": cid,
                "source": rec["source"],
                "brand": rec["brand"],
                "model": rec["model"],
                "network_type": rec["network_type"],
                "bandwidth_mbps": round(bandwidth, 4),
                "compute_units": round(source_compute[source] + 0.5 * q, 4),
                "battery_j": round(max_battery * min(0.95, battery_frac), 4),
                "max_battery_j": round(max_battery, 4),
                "tx_energy_per_mb": round(source_tx[source] * (1.0 + 0.15 * (1.0 - q)), 5),
                "compute_energy_per_step": round(0.004 + 0.0025 * (cid % 5), 5),
                "recharge_j_per_round": round(2.0 + 0.4 * (cid % 7), 4),
                "availability": round(0.82 + 0.015 * (cid % 8), 4),
            }
        )
    return profiles


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(records: list[dict[str, str | float]]) -> list[dict[str, str | float]]:
    rows = []
    for source in ["4G", "5G", "wifi4", "wifi5", "wifi6", "all"]:
        subset = records if source == "all" else [r for r in records if str(r["source"]) == source]
        values = [float(r["bandwidth_mbps"]) for r in subset]
        rows.append(
            {
                "source": source,
                "n": len(values),
                "mean_mbps": round(statistics.mean(values), 4) if values else 0.0,
                "p10_mbps": round(percentile(values, 0.10), 4),
                "p50_mbps": round(percentile(values, 0.50), 4),
                "p90_mbps": round(percentile(values, 0.90), 4),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic trace-derived scheduler profiles from public mobile bandwidth traces.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/mobilebandwidth/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/mobilebandwidth/real_mobile_profiles.csv"))
    parser.add_argument("--summary", type=Path, default=Path("analysis/mobile_trace_summary.csv"))
    parser.add_argument("--num-clients", type=int, default=10)
    args = parser.parse_args()

    for filename in TRACE_FILES:
        download(f"{RAW_BASE}/{filename}", args.raw_dir / filename)
    records = load_records(args.raw_dir)
    if not records:
        raise RuntimeError("No valid bandwidth records were loaded.")

    profiles = build_profiles(records, num_clients=args.num_clients)
    write_csv(args.out, profiles)
    write_csv(args.summary, summarize(records))
    print(f"Loaded {len(records)} real mobile bandwidth records")
    print(f"Wrote {args.out}")
    print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()
