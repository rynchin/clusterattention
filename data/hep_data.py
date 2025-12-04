"""
Data loading utilities for HEP event-level particle data.
Handles variable-length sequences, padding, and normalization.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List
import numpy as np

from data.simulate_hep_events import generate_hep_dataset


class HEPEventDataset(Dataset):
    """Dataset for HEP particle events with variable lengths."""
    
    def __init__(
        self,
        events: List[torch.Tensor],
        targets: torch.Tensor,
        lengths: torch.Tensor,
        max_length: int = 512,
        normalize: bool = True,
        feature_stats: Optional[dict] = None
    ):
        """
        Args:
            events: List of (n_particles_i, feature_dim) tensors
            targets: (n_events, target_dim) tensor with regression targets
            lengths: (n_events,) tensor with number of particles per event
            max_length: Maximum sequence length (pad/truncate to this)
            normalize: Whether to normalize features
            feature_stats: Dict with 'mean' and 'std' for normalization (computed if None)
        """
        self.events = events
        self.targets = targets
        self.lengths = lengths
        self.max_length = max_length
        self.normalize = normalize
        
        # Compute normalization stats if needed
        if normalize and feature_stats is None:
            self.feature_stats = self._compute_stats()
        elif normalize:
            self.feature_stats = feature_stats
        else:
            self.feature_stats = None
    
    def _compute_stats(self) -> dict:
        """Compute mean and std across all events."""
        all_features = torch.cat(self.events, dim=0)  # (total_particles, feature_dim)
        mean = all_features.mean(dim=0)
        std = all_features.std(dim=0)
        # Avoid division by zero
        std = torch.clamp(std, min=1e-6)
        return {'mean': mean, 'std': std}
    
    def __len__(self) -> int:
        return len(self.events)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            features: (max_length, feature_dim) padded/truncated particle features
            target: (target_dim,) regression target
            mask: (max_length,) boolean mask (True for real particles, False for padding)
        """
        particles = self.events[idx]  # (n_particles, feature_dim)
        target = self.targets[idx]  # (target_dim,)
        length = self.lengths[idx].item()
        
        # Normalize if requested
        if self.normalize and self.feature_stats is not None:
            particles = (particles - self.feature_stats['mean']) / self.feature_stats['std']
        
        # Truncate or pad
        if length > self.max_length:
            particles = particles[:self.max_length]
            length = self.max_length
        
        feature_dim = particles.shape[1]
        
        # Pad to max_length
        if length < self.max_length:
            padding = torch.zeros(self.max_length - length, feature_dim, dtype=particles.dtype, device=particles.device)
            particles = torch.cat([particles, padding], dim=0)
        
        # Create attention mask (True for real particles, False for padding)
        mask = torch.zeros(self.max_length, dtype=torch.bool, device=particles.device)
        mask[:length] = True
        
        return particles, target, mask


def create_hep_dataloaders(
    n_train: int = 10000,
    n_val: int = 2000,
    n_test: int = 2000,
    min_particles: int = 50,
    max_particles: int = 500,
    max_length: int = 512,
    batch_size: int = 32,
    normalize: bool = True,
    device: str = 'cpu',
    seed: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Create train/val/test dataloaders for HEP events.
    
    Returns:
        train_loader, val_loader, test_loader, feature_stats
    """
    # Generate datasets
    train_events, train_targets, train_lengths = generate_hep_dataset(
        n_train, min_particles, max_particles, device=device, seed=seed
    )
    val_events, val_targets, val_lengths = generate_hep_dataset(
        n_val, min_particles, max_particles, device=device, seed=None if seed is None else seed + n_train
    )
    test_events, test_targets, test_lengths = generate_hep_dataset(
        n_test, min_particles, max_particles, device=device, seed=None if seed is None else seed + n_train + n_val
    )
    
    # Compute normalization stats from training set only
    train_dataset = HEPEventDataset(
        train_events, train_targets, train_lengths, max_length, normalize=False
    )
    feature_stats = train_dataset._compute_stats() if normalize else None
    
    # Create datasets with normalization
    train_dataset = HEPEventDataset(
        train_events, train_targets, train_lengths, max_length, normalize=normalize, feature_stats=feature_stats
    )
    val_dataset = HEPEventDataset(
        val_events, val_targets, val_lengths, max_length, normalize=normalize, feature_stats=feature_stats
    )
    test_dataset = HEPEventDataset(
        test_events, test_targets, test_lengths, max_length, normalize=normalize, feature_stats=feature_stats
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, feature_stats or {}


if __name__ == '__main__':
    # Test data loading
    train_loader, val_loader, test_loader, stats = create_hep_dataloaders(
        n_train=100, n_val=20, n_test=20, batch_size=8, max_length=256
    )
    
    print(f"Feature stats: {stats}")
    
    # Test a batch
    for features, targets, masks in train_loader:
        print(f"Batch features shape: {features.shape}")  # (B, T, feature_dim)
        print(f"Batch targets shape: {targets.shape}")  # (B, target_dim)
        print(f"Batch masks shape: {masks.shape}")  # (B, T)
        print(f"Mask example: {masks[0][:20]}")
        break

