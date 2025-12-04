"""
Data loading utilities for ModelNet40 point cloud classification.
Uses PyTorch Geometric for data handling.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
import numpy as np

try:
    from torch_geometric.datasets import ModelNet
    from torch_geometric.transforms import SamplePoints, NormalizeScale, Compose
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False
    print("Warning: torch-geometric not installed. Install with: pip install torch-geometric")


class ModelNet40Dataset(Dataset):
    """Dataset wrapper for ModelNet40 that returns (points, label, mask) tuples."""
    
    def __init__(
        self,
        root: str = 'data/ModelNet40',
        train: bool = True,
        num_points: int = 1024,
        use_normals: bool = False,
        normalize: bool = True,
        feature_stats: Optional[dict] = None
    ):
        """
        Args:
            root: Root directory for dataset storage
            train: If True, use training set; else test set
            num_points: Number of points to sample from each mesh
            use_normals: If True, include surface normals (6D features); else just xyz (3D)
            normalize: Whether to normalize point clouds
            feature_stats: Dict with 'mean' and 'std' for normalization (computed if None)
        """
        if not HAS_TORCH_GEOMETRIC:
            raise ImportError("torch-geometric is required. Install with: pip install torch-geometric")
        
        self.num_points = num_points
        self.use_normals = use_normals
        self.normalize = normalize
        
        # Transforms: sample points and normalize to unit sphere
        transforms = [
            SamplePoints(num_points, include_normals=use_normals),
        ]
        if normalize:
            transforms.append(NormalizeScale())
        
        pre_transform = Compose(transforms)
        
        # Load ModelNet40
        self.dataset = ModelNet(
            root=root,
            name='40',
            train=train,
            pre_transform=pre_transform
        )
        
        # Compute or use provided feature stats for additional normalization
        if normalize and feature_stats is None:
            self.feature_stats = self._compute_stats()
        elif normalize:
            self.feature_stats = feature_stats
        else:
            self.feature_stats = None
    
    def _compute_stats(self) -> dict:
        """Compute mean and std across all point clouds."""
        all_points = []
        for i in range(len(self.dataset)):
            data = self.dataset[i]
            points = data.pos  # (num_points, 3)
            if self.use_normals and hasattr(data, 'normal'):
                points = torch.cat([points, data.normal], dim=-1)  # (num_points, 6)
            all_points.append(points)
        
        all_points = torch.cat(all_points, dim=0)  # (total_points, feature_dim)
        mean = all_points.mean(dim=0)
        std = all_points.std(dim=0)
        std = torch.clamp(std, min=1e-6)
        return {'mean': mean, 'std': std}
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            points: (num_points, feature_dim) point cloud features
            label: scalar class label (0-39)
            mask: (num_points,) boolean mask (all True for point clouds)
        """
        data = self.dataset[idx]
        
        # Get points (and optionally normals)
        points = data.pos  # (num_points, 3)
        if self.use_normals and hasattr(data, 'normal'):
            points = torch.cat([points, data.normal], dim=-1)  # (num_points, 6)
        
        # Apply feature normalization
        if self.normalize and self.feature_stats is not None:
            points = (points - self.feature_stats['mean']) / self.feature_stats['std']
        
        # Get label
        label = data.y.item()
        
        # Create mask (all True for point clouds - no padding)
        mask = torch.ones(self.num_points, dtype=torch.bool)
        
        return points, torch.tensor(label, dtype=torch.long), mask
    
    @property
    def num_classes(self) -> int:
        return 40
    
    @property
    def feature_dim(self) -> int:
        return 6 if self.use_normals else 3


def create_modelnet_dataloaders(
    root: str = 'data/ModelNet40',
    num_points: int = 1024,
    use_normals: bool = False,
    batch_size: int = 32,
    normalize: bool = True,
    num_workers: int = 0,
    seed: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, dict]:
    """
    Create train/test dataloaders for ModelNet40.
    
    Note: ModelNet40 only has train/test splits (no validation).
    For validation, you can split the training set manually.
    
    Args:
        root: Root directory for dataset storage
        num_points: Number of points to sample per mesh
        use_normals: Include surface normals (6D) or just xyz (3D)
        batch_size: Batch size for dataloaders
        normalize: Whether to normalize features
        num_workers: Number of dataloader workers
        seed: Random seed (for reproducibility)
    
    Returns:
        train_loader, test_loader, stats_dict
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Create training dataset first to compute stats
    train_dataset = ModelNet40Dataset(
        root=root,
        train=True,
        num_points=num_points,
        use_normals=use_normals,
        normalize=normalize,
        feature_stats=None  # Will compute stats
    )
    
    # Use training stats for test set
    feature_stats = train_dataset.feature_stats
    
    test_dataset = ModelNet40Dataset(
        root=root,
        train=False,
        num_points=num_points,
        use_normals=use_normals,
        normalize=normalize,
        feature_stats=feature_stats
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    stats = {
        'feature_stats': feature_stats,
        'num_classes': train_dataset.num_classes,
        'feature_dim': train_dataset.feature_dim,
        'num_points': num_points,
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
    }
    
    return train_loader, test_loader, stats


def create_modelnet_dataloaders_with_val(
    root: str = 'data/ModelNet40',
    num_points: int = 1024,
    use_normals: bool = False,
    batch_size: int = 32,
    normalize: bool = True,
    val_split: float = 0.1,
    num_workers: int = 0,
    seed: Optional[int] = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Create train/val/test dataloaders for ModelNet40.
    
    Splits the training set to create a validation set.
    
    Returns:
        train_loader, val_loader, test_loader, stats_dict
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Create full training dataset
    full_train_dataset = ModelNet40Dataset(
        root=root,
        train=True,
        num_points=num_points,
        use_normals=use_normals,
        normalize=normalize,
        feature_stats=None
    )
    
    feature_stats = full_train_dataset.feature_stats
    
    # Split training into train/val
    n_total = len(full_train_dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    
    generator = torch.Generator().manual_seed(seed) if seed else None
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset,
        [n_train, n_val],
        generator=generator
    )
    
    # Create test dataset
    test_dataset = ModelNet40Dataset(
        root=root,
        train=False,
        num_points=num_points,
        use_normals=use_normals,
        normalize=normalize,
        feature_stats=feature_stats
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    stats = {
        'feature_stats': feature_stats,
        'num_classes': 40,
        'feature_dim': 6 if use_normals else 3,
        'num_points': num_points,
        'train_size': n_train,
        'val_size': n_val,
        'test_size': len(test_dataset),
    }
    
    return train_loader, val_loader, test_loader, stats


# Class names for ModelNet40
MODELNET40_CLASSES = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf',
    'bottle', 'bowl', 'car', 'chair', 'cone',
    'cup', 'curtain', 'desk', 'door', 'dresser',
    'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp',
    'laptop', 'mantel', 'monitor', 'night_stand', 'person',
    'piano', 'plant', 'radio', 'range_hood', 'sink',
    'sofa', 'stairs', 'stool', 'table', 'tent',
    'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
]


if __name__ == '__main__':
    print("Testing ModelNet40 DataLoaders")
    print("=" * 50)
    
    # Test with validation split
    train_loader, val_loader, test_loader, stats = create_modelnet_dataloaders_with_val(
        num_points=1024,
        use_normals=False,
        batch_size=8,
        val_split=0.1,
        seed=42
    )
    
    print(f"Dataset stats: {stats}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test a batch
    for points, labels, masks in train_loader:
        print(f"\nBatch shapes:")
        print(f"  Points: {points.shape}")  # (B, num_points, feature_dim)
        print(f"  Labels: {labels.shape}")  # (B,)
        print(f"  Masks: {masks.shape}")    # (B, num_points)
        print(f"  Points range: [{points.min():.3f}, {points.max():.3f}]")
        print(f"  Labels in batch: {labels.tolist()}")
        print(f"  Class names: {[MODELNET40_CLASSES[l] for l in labels.tolist()]}")
        break

