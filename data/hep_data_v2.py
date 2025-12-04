"""
Data loading utilities for improved HEP benchmark (v2).
Uses resonance reconstruction task that requires learning particle relationships.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List
import numpy as np

try:
    from data.simulate_hep_events_v2 import generate_resonance_dataset, generate_jet_tagging_dataset
except ImportError:
    from simulate_hep_events_v2 import generate_resonance_dataset, generate_jet_tagging_dataset


class ResonanceDataset(Dataset):
    """Dataset for resonance reconstruction with variable-length particle sequences."""
    
    def __init__(
        self,
        events: List[torch.Tensor],
        targets: torch.Tensor,
        lengths: torch.Tensor,
        max_length: int = 256,
        normalize_features: bool = True,
        normalize_targets: bool = True,
        feature_stats: Optional[dict] = None,
        target_stats: Optional[dict] = None
    ):
        self.events = events
        self.targets = targets
        self.lengths = lengths
        self.max_length = max_length
        self.normalize_features = normalize_features
        self.normalize_targets = normalize_targets
        
        # Compute normalization stats if needed
        if normalize_features and feature_stats is None:
            self.feature_stats = self._compute_feature_stats()
        elif normalize_features:
            self.feature_stats = feature_stats
        else:
            self.feature_stats = None
            
        if normalize_targets and target_stats is None:
            self.target_stats = self._compute_target_stats()
        elif normalize_targets:
            self.target_stats = target_stats
        else:
            self.target_stats = None
    
    def _compute_feature_stats(self) -> dict:
        """Compute mean and std across all particles."""
        all_features = torch.cat(self.events, dim=0)
        mean = all_features.mean(dim=0)
        std = all_features.std(dim=0)
        std = torch.clamp(std, min=1e-6)
        return {'mean': mean, 'std': std}
    
    def _compute_target_stats(self) -> dict:
        """Compute mean and std for targets."""
        mean = self.targets.mean(dim=0)
        std = self.targets.std(dim=0)
        std = torch.clamp(std, min=1e-6)
        return {'mean': mean, 'std': std}
    
    def __len__(self) -> int:
        return len(self.events)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        particles = self.events[idx].clone()
        target = self.targets[idx].clone()
        length = min(self.lengths[idx].item(), self.max_length)
        
        # Normalize features
        if self.normalize_features and self.feature_stats is not None:
            particles = (particles - self.feature_stats['mean']) / self.feature_stats['std']
        
        # Normalize targets
        if self.normalize_targets and self.target_stats is not None:
            target = (target - self.target_stats['mean']) / self.target_stats['std']
        
        # Truncate or pad
        if len(particles) > self.max_length:
            particles = particles[:self.max_length]
            length = self.max_length
        
        feature_dim = particles.shape[1]
        
        # Pad to max_length
        if len(particles) < self.max_length:
            padding = torch.zeros(self.max_length - len(particles), feature_dim, dtype=particles.dtype)
            particles = torch.cat([particles, padding], dim=0)
        
        # Create attention mask
        mask = torch.zeros(self.max_length, dtype=torch.bool)
        mask[:length] = True
        
        return particles, target, mask
    
    def denormalize_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        """Convert normalized predictions back to original scale."""
        if self.target_stats is not None:
            return predictions * self.target_stats['std'] + self.target_stats['mean']
        return predictions


def create_resonance_dataloaders(
    n_train: int = 10000,
    n_val: int = 2000,
    n_test: int = 2000,
    min_background: int = 20,
    max_background: int = 150,
    max_length: int = 256,
    batch_size: int = 32,
    difficulty: str = 'medium',
    normalize_features: bool = True,
    normalize_targets: bool = True,
    device: str = 'cpu',
    seed: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Create train/val/test dataloaders for resonance reconstruction.
    
    Args:
        difficulty: 'easy' (signal flag exposed), 'medium' (must identify signal), 'hard' (more noise)
    
    Returns:
        train_loader, val_loader, test_loader, stats_dict
    """
    # Generate datasets
    train_events, train_targets, train_lengths = generate_resonance_dataset(
        n_train, min_background, max_background, difficulty=difficulty,
        device=device, seed=seed
    )
    val_events, val_targets, val_lengths = generate_resonance_dataset(
        n_val, min_background, max_background, difficulty=difficulty,
        device=device, seed=None if seed is None else seed + n_train
    )
    test_events, test_targets, test_lengths = generate_resonance_dataset(
        n_test, min_background, max_background, difficulty=difficulty,
        device=device, seed=None if seed is None else seed + n_train + n_val
    )
    
    # Compute stats from training data only
    train_dataset_temp = ResonanceDataset(
        train_events, train_targets, train_lengths, max_length,
        normalize_features=False, normalize_targets=False
    )
    feature_stats = train_dataset_temp._compute_feature_stats() if normalize_features else None
    target_stats = train_dataset_temp._compute_target_stats() if normalize_targets else None
    
    # Create datasets
    train_dataset = ResonanceDataset(
        train_events, train_targets, train_lengths, max_length,
        normalize_features, normalize_targets, feature_stats, target_stats
    )
    val_dataset = ResonanceDataset(
        val_events, val_targets, val_lengths, max_length,
        normalize_features, normalize_targets, feature_stats, target_stats
    )
    test_dataset = ResonanceDataset(
        test_events, test_targets, test_lengths, max_length,
        normalize_features, normalize_targets, feature_stats, target_stats
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    stats = {
        'feature_stats': feature_stats,
        'target_stats': target_stats,
    }
    
    return train_loader, val_loader, test_loader, stats


class JetTaggingDataset(Dataset):
    """Dataset for jet classification (quark vs gluon)."""
    
    def __init__(
        self,
        events: List[torch.Tensor],
        targets: torch.Tensor,
        lengths: torch.Tensor,
        max_length: int = 128,
        normalize: bool = True,
        feature_stats: Optional[dict] = None
    ):
        self.events = events
        self.targets = targets
        self.lengths = lengths
        self.max_length = max_length
        self.normalize = normalize
        
        if normalize and feature_stats is None:
            self.feature_stats = self._compute_stats()
        elif normalize:
            self.feature_stats = feature_stats
        else:
            self.feature_stats = None
    
    def _compute_stats(self) -> dict:
        all_features = torch.cat(self.events, dim=0)
        mean = all_features.mean(dim=0)
        std = all_features.std(dim=0)
        std = torch.clamp(std, min=1e-6)
        return {'mean': mean, 'std': std}
    
    def __len__(self) -> int:
        return len(self.events)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        particles = self.events[idx].clone()
        target = self.targets[idx]
        length = min(self.lengths[idx].item(), self.max_length)
        
        if self.normalize and self.feature_stats is not None:
            particles = (particles - self.feature_stats['mean']) / self.feature_stats['std']
        
        if len(particles) > self.max_length:
            particles = particles[:self.max_length]
            length = self.max_length
        
        feature_dim = particles.shape[1]
        if len(particles) < self.max_length:
            padding = torch.zeros(self.max_length - len(particles), feature_dim, dtype=particles.dtype)
            particles = torch.cat([particles, padding], dim=0)
        
        mask = torch.zeros(self.max_length, dtype=torch.bool)
        mask[:length] = True
        
        return particles, target, mask


def create_jet_tagging_dataloaders(
    n_train: int = 10000,
    n_val: int = 2000,
    n_test: int = 2000,
    n_particles_per_jet: int = 50,
    max_length: int = 128,
    batch_size: int = 32,
    normalize: bool = True,
    device: str = 'cpu',
    seed: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Create dataloaders for jet tagging (classification) task."""
    
    train_events, train_targets, train_lengths = generate_jet_tagging_dataset(
        n_train, n_particles_per_jet, device=device, seed=seed
    )
    val_events, val_targets, val_lengths = generate_jet_tagging_dataset(
        n_val, n_particles_per_jet, device=device, 
        seed=None if seed is None else seed + n_train
    )
    test_events, test_targets, test_lengths = generate_jet_tagging_dataset(
        n_test, n_particles_per_jet, device=device,
        seed=None if seed is None else seed + n_train + n_val
    )
    
    # Compute stats from training data
    train_temp = JetTaggingDataset(train_events, train_targets, train_lengths, max_length, normalize=False)
    feature_stats = train_temp._compute_stats() if normalize else None
    
    train_dataset = JetTaggingDataset(train_events, train_targets, train_lengths, max_length, normalize, feature_stats)
    val_dataset = JetTaggingDataset(val_events, val_targets, val_lengths, max_length, normalize, feature_stats)
    test_dataset = JetTaggingDataset(test_events, test_targets, test_lengths, max_length, normalize, feature_stats)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, {'feature_stats': feature_stats}


if __name__ == '__main__':
    print("Testing Resonance Reconstruction DataLoaders")
    print("=" * 50)
    
    train_loader, val_loader, test_loader, stats = create_resonance_dataloaders(
        n_train=100, n_val=20, n_test=20, batch_size=8, max_length=128, difficulty='medium'
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Feature stats: mean shape={stats['feature_stats']['mean'].shape}")
    print(f"Target stats: mean={stats['target_stats']['mean']}, std={stats['target_stats']['std']}")
    
    for features, targets, masks in train_loader:
        print(f"\nBatch shapes:")
        print(f"  Features: {features.shape}")  # (B, T, feature_dim)
        print(f"  Targets: {targets.shape}")    # (B, 4)
        print(f"  Masks: {masks.shape}")        # (B, T)
        print(f"  Features range: [{features.min():.2f}, {features.max():.2f}]")
        print(f"  Targets range: [{targets.min():.2f}, {targets.max():.2f}]")
        break
    
    print("\n" + "=" * 50)
    print("Testing Jet Tagging DataLoaders")
    print("=" * 50)
    
    train_loader, val_loader, test_loader, stats = create_jet_tagging_dataloaders(
        n_train=100, n_val=20, n_test=20, batch_size=8
    )
    
    for features, targets, masks in train_loader:
        print(f"Features: {features.shape}, Targets: {targets.shape}")
        print(f"Class distribution in batch: {targets.mean():.2f}")
        break

