from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

from .utils import ensure_dir


@dataclass
class DatasetMetadata:
    name: str
    num_classes: int
    channels: int
    image_size: int
    mean: tuple[float, ...]
    std: tuple[float, ...]


class RandomImageDataset(Dataset):
    """Small synthetic dataset for smoke tests.

    Samples are noisy class prototypes, which makes the task non-trivial
    but easy enough for quick debugging.
    """

    def __init__(
        self,
        size: int,
        num_classes: int,
        channels: int,
        image_size: int,
        seed: int = 0,
    ) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.num_classes = num_classes
        self.channels = channels
        self.image_size = image_size
        self.size = size

        self.labels = torch.randint(0, num_classes, (size,), generator=generator)
        class_templates = torch.randn(num_classes, channels, image_size, image_size, generator=generator)
        noise = 0.35 * torch.randn(size, channels, image_size, image_size, generator=generator)
        self.images = class_templates[self.labels] + noise
        self.images = self.images.float()

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.images[idx], int(self.labels[idx].item())


class TensorImageDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, transform: Callable | None = None) -> None:
        self.images = images
        self.labels = labels.long()
        self.targets = self.labels.tolist()
        self.transform = transform

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        x = self.images[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.labels[idx].item())


def _read_idx(path: Path) -> np.ndarray:
    open_fn = gzip.open if path.suffix == ".gz" else open
    with open_fn(path, "rb") as f:
        magic, = struct.unpack(">I", f.read(4))
        ndim = magic & 0xFF
        shape = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(shape)


def _load_mnist_from_raw(root: str, name: str, train: bool, transform: Callable) -> Dataset:
    dataset_dir = "FashionMNIST" if name in {"fmnist", "fashionmnist", "fashion-mnist"} else "MNIST"
    raw_dir = Path(root) / dataset_dir / "raw"
    split = "train" if train else "t10k"
    image_path = raw_dir / f"{split}-images-idx3-ubyte"
    label_path = raw_dir / f"{split}-labels-idx1-ubyte"
    if not image_path.exists():
        image_path = image_path.with_suffix(image_path.suffix + ".gz")
    if not label_path.exists():
        label_path = label_path.with_suffix(label_path.suffix + ".gz")
    if not image_path.exists() or not label_path.exists():
        raise FileNotFoundError(
            f"Missing cached {dataset_dir} IDX files under {raw_dir}. "
            "Install torchvision for automatic download or pre-download the dataset."
        )
    images = torch.from_numpy(_read_idx(image_path).copy()).unsqueeze(1)
    labels = torch.from_numpy(_read_idx(label_path).astype(np.int64))
    return TensorImageDataset(images=images, labels=labels, transform=transform)


def _pil_or_array_to_tensor(img: Image.Image | np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(img, torch.Tensor):
        tensor = img.float()
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor
    arr = np.array(img, copy=True)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float() / 255.0
    return tensor


def _normalize(x: torch.Tensor, mean: tuple[float, ...], std: tuple[float, ...]) -> torch.Tensor:
    mean_t = torch.tensor(mean, dtype=x.dtype).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype).view(-1, 1, 1)
    return (x - mean_t) / std_t


def build_transform(meta: DatasetMetadata, train: bool, enable_augmentation: bool) -> Callable:
    def transform(img: Image.Image | np.ndarray | torch.Tensor) -> torch.Tensor:
        x = _pil_or_array_to_tensor(img)
        if train and enable_augmentation:
            # Lightweight augmentation without torchvision transforms.
            if meta.channels == 3:
                if torch.rand(1).item() < 0.5:
                    x = torch.flip(x, dims=[2])  # horizontal flip
                pad = 4
                x = F.pad(x.unsqueeze(0), (pad, pad, pad, pad), mode="reflect").squeeze(0)
                top = torch.randint(0, 2 * pad + 1, (1,)).item()
                left = torch.randint(0, 2 * pad + 1, (1,)).item()
                x = x[:, top : top + meta.image_size, left : left + meta.image_size]
        x = _normalize(x, meta.mean, meta.std)
        return x

    return transform


def get_metadata(dataset_name: str, fake_num_classes: int = 10, fake_channels: int = 3, fake_image_size: int = 32) -> DatasetMetadata:
    name = dataset_name.lower()
    if name == "mnist":
        return DatasetMetadata("mnist", 10, 1, 28, (0.1307,), (0.3081,))
    if name in {"fmnist", "fashionmnist", "fashion-mnist"}:
        return DatasetMetadata("fashionmnist", 10, 1, 28, (0.2860,), (0.3530,))
    if name == "cifar10":
        return DatasetMetadata("cifar10", 10, 3, 32, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    if name == "fake":
        if fake_channels == 1:
            mean = (0.5,)
            std = (0.5,)
        else:
            mean = tuple([0.5] * fake_channels)
            std = tuple([0.5] * fake_channels)
        return DatasetMetadata("fake", fake_num_classes, fake_channels, fake_image_size, mean, std)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _load_torchvision_dataset(name: str, root: str, train: bool, transform: Callable, download: bool) -> Dataset:
    try:
        import torchvision.datasets as tv_datasets  # type: ignore
    except Exception as exc:
        if name in {"mnist", "fmnist", "fashionmnist", "fashion-mnist"}:
            return _load_mnist_from_raw(root=root, name=name, train=train, transform=transform)
        raise ImportError(
            "torchvision could not be imported. Install a torch/torchvision pair "
            "with matching versions, or use dataset.name=fake for smoke tests."
        ) from exc

    if name == "mnist":
        return tv_datasets.MNIST(root=root, train=train, transform=transform, download=download)
    if name in {"fmnist", "fashionmnist", "fashion-mnist"}:
        return tv_datasets.FashionMNIST(root=root, train=train, transform=transform, download=download)
    if name == "cifar10":
        return tv_datasets.CIFAR10(root=root, train=train, transform=transform, download=download)
    raise ValueError(f"Unsupported torchvision dataset: {name}")


def build_dataset(cfg: Dict, train: bool) -> tuple[Dataset, DatasetMetadata]:
    dataset_cfg = cfg["dataset"]
    name = dataset_cfg["name"].lower()
    meta = get_metadata(
        name,
        fake_num_classes=int(dataset_cfg.get("fake_num_classes", 10)),
        fake_channels=int(dataset_cfg.get("fake_channels", 3)),
        fake_image_size=int(dataset_cfg.get("fake_image_size", 32)),
    )
    transform = build_transform(meta, train=train, enable_augmentation=bool(dataset_cfg.get("augment", False)))

    if name == "fake":
        size = int(dataset_cfg["fake_train_size"] if train else dataset_cfg["fake_test_size"])
        dataset_seed = int(dataset_cfg.get("dataset_seed", cfg.get("seed", 0)))
        dataset = RandomImageDataset(
            size=size,
            num_classes=meta.num_classes,
            channels=meta.channels,
            image_size=meta.image_size,
            seed=dataset_seed + (0 if train else 1),
        )
        return dataset, meta

    root = ensure_dir(dataset_cfg.get("root", "./data"))
    dataset = _load_torchvision_dataset(name, str(root), train=train, transform=transform, download=bool(dataset_cfg.get("download", True)))
    return dataset, meta


def extract_targets(dataset: Dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        targets = getattr(dataset, "targets")
    elif hasattr(dataset, "labels"):
        targets = getattr(dataset, "labels")
    elif isinstance(dataset, Subset):
        full_targets = extract_targets(dataset.dataset)
        return full_targets[np.array(dataset.indices)]
    else:
        raise AttributeError("Dataset does not expose .targets or .labels")
    if isinstance(targets, torch.Tensor):
        return targets.cpu().numpy()
    return np.asarray(targets)


def dirichlet_partition(
    labels: Sequence[int] | np.ndarray,
    num_clients: int,
    alpha: float,
    min_size: int,
    seed: int = 0,
) -> List[List[int]]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    num_classes = int(labels.max() + 1)
    min_partition_size = 0

    while min_partition_size < min_size:
        idx_batch: List[List[int]] = [[] for _ in range(num_clients)]
        for cls in range(num_classes):
            idx_cls = np.where(labels == cls)[0]
            rng.shuffle(idx_cls)
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array(
                [p * (len(idx_batch[j]) < len(labels) / num_clients) for j, p in enumerate(proportions)]
            )
            proportions = proportions / proportions.sum()
            split_points = (np.cumsum(proportions) * len(idx_cls)).astype(int)[:-1]
            class_splits = np.split(idx_cls, split_points)
            for j, cls_idx in enumerate(class_splits):
                idx_batch[j].extend(cls_idx.tolist())
        min_partition_size = min(len(client_idx) for client_idx in idx_batch)

    for client_idx in idx_batch:
        rng.shuffle(client_idx)
    return idx_batch


def build_client_loaders(cfg: Dict) -> tuple[Dict[int, DataLoader], DataLoader, Dict[int, dict], DatasetMetadata]:
    train_dataset, meta = build_dataset(cfg, train=True)
    test_dataset, _ = build_dataset(cfg, train=False)

    labels = extract_targets(train_dataset)
    system_cfg = cfg["system"]
    dataset_cfg = cfg["dataset"]

    proxy_size = int(dataset_cfg.get("public_proxy_size", 0) or 0)
    if proxy_size > 0:
        proxy_seed = int(dataset_cfg.get("public_proxy_seed", cfg.get("seed", 0)))
        rng = np.random.default_rng(proxy_seed)
        perm = rng.permutation(len(labels))
        proxy_set = set(int(i) for i in perm[: min(proxy_size, len(labels))])
        train_pool_indices = np.array([i for i in range(len(labels)) if i not in proxy_set], dtype=np.int64)
    else:
        train_pool_indices = np.arange(len(labels), dtype=np.int64)

    partition_labels = labels[train_pool_indices]
    partition_indices = dirichlet_partition(
        labels=partition_labels,
        num_clients=int(system_cfg["num_clients"]),
        alpha=float(dataset_cfg.get("dirichlet_alpha", 0.3)),
        min_size=int(dataset_cfg.get("min_client_samples", 10)),
        seed=int(dataset_cfg.get("partition_seed", cfg.get("seed", 0))),
    )
    client_indices = [[int(train_pool_indices[i]) for i in part] for part in partition_indices]

    requested_device = str(cfg["system"].get("resolved_device", cfg["system"].get("device", "auto")))
    pin_memory = requested_device.startswith("cuda") and torch.cuda.is_available()
    train_loaders: Dict[int, DataLoader] = {}
    client_meta: Dict[int, dict] = {}

    for client_id, indices in enumerate(client_indices):
        subset = Subset(train_dataset, indices)
        batch_size = int(dataset_cfg.get("batch_size", 64))
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(dataset_cfg.get("num_workers", 0)),
            pin_memory=pin_memory,
            drop_last=False,
        )
        train_loaders[client_id] = loader

        client_labels = labels[np.array(indices)]
        hist = np.bincount(client_labels, minlength=meta.num_classes)
        client_meta[client_id] = {
            "num_samples": int(len(indices)),
            "label_hist": hist.tolist(),
            "unique_labels": int((hist > 0).sum()),
        }

    test_loader = DataLoader(
        test_dataset,
        batch_size=int(dataset_cfg.get("eval_batch_size", 256)),
        shuffle=False,
        num_workers=int(dataset_cfg.get("num_workers", 0)),
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loaders, test_loader, client_meta, meta


def build_public_proxy_loader(cfg: Dict) -> tuple[DataLoader, DatasetMetadata]:
    train_dataset, meta = build_dataset(cfg, train=True)
    dataset_cfg = cfg["dataset"]
    proxy_size = int(dataset_cfg.get("public_proxy_size", 0) or 0)
    if proxy_size <= 0:
        raise ValueError("dataset.public_proxy_size must be positive for a public proxy loader")
    labels = extract_targets(train_dataset)
    proxy_seed = int(dataset_cfg.get("public_proxy_seed", cfg.get("seed", 0)))
    rng = np.random.default_rng(proxy_seed)
    indices = rng.permutation(len(labels))[: min(proxy_size, len(labels))].tolist()
    subset = Subset(train_dataset, [int(i) for i in indices])
    requested_device = str(cfg["system"].get("resolved_device", cfg["system"].get("device", "auto")))
    pin_memory = requested_device.startswith("cuda") and torch.cuda.is_available()
    loader = DataLoader(
        subset,
        batch_size=int(dataset_cfg.get("proxy_batch_size", dataset_cfg.get("eval_batch_size", 128))),
        shuffle=False,
        num_workers=int(dataset_cfg.get("num_workers", 0)),
        pin_memory=pin_memory,
        drop_last=False,
    )
    return loader, meta
