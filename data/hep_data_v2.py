"""
Data loading utilities for HEP jet tagging (classification) benchmark.
Handles variable-length sequences, padding, and normalization.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List
import numpy as np

from data.simulate_hep_events_v2 import generate_jet_tagging_dataset


class JetTaggingDataset(Dataset):
    """Dataset for HEP jet tagging with variable-length particle sequences."""
    
    def __init__(
        self,
        events: List[torch.Tensor],
        targets: torch.Tensor,
        lengths: torch.Tensor,
        max_length: int = 128,
        normalize: bool = True,
        feature_stats: Optional[dict] = None
    ):
        """
        Args:
            events: List of (n_particles_i, feature_dim) tensors
            targets: (n_events,) tensor with binary classification labels (0=quark, 1=gluon)
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
            target: (1,) tensor with binary label (0=quark, 1=gluon)
            mask: (max_length,) boolean mask (True for real particles, False for padding)
        """
        particles = self.events[idx]  # (n_particles, feature_dim)
        target = self.targets[idx]  # scalar tensor
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


def create_jet_tagging_dataloaders(
    n_train: int = 50000,
    n_val: int = 10000,
    n_test: int = 10000,
    n_particles_per_jet: int = 50,
    max_length: int = 128,
    batch_size: int = 32,
    normalize: bool = True,
    device: str = 'cpu',
    seed: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Create train/val/test dataloaders for jet tagging (quark vs gluon classification).
    
    Returns:
        train_loader, val_loader, test_loader, feature_stats
    """
    # Generate datasets
    train_events, train_targets, train_lengths = generate_jet_tagging_dataset(
        n_train, n_particles_per_jet=n_particles_per_jet, device=device, seed=seed
    )
    val_events, val_targets, val_lengths = generate_jet_tagging_dataset(
        n_val, n_particles_per_jet=n_particles_per_jet, device=device, 
        seed=None if seed is None else seed + n_train
    )
    test_events, test_targets, test_lengths = generate_jet_tagging_dataset(
        n_test, n_particles_per_jet=n_particles_per_jet, device=device,
        seed=None if seed is None else seed + n_train + n_val
    )
    
    # Compute normalization stats from training set only
    train_dataset = JetTaggingDataset(
        train_events, train_targets, train_lengths, max_length, normalize=False
    )
    feature_stats = train_dataset._compute_stats() if normalize else None
    
    # Create datasets with normalization
    train_dataset = JetTaggingDataset(
        train_events, train_targets, train_lengths, max_length, normalize=normalize, feature_stats=feature_stats
    )
    val_dataset = JetTaggingDataset(
        val_events, val_targets, val_lengths, max_length, normalize=normalize, feature_stats=feature_stats
    )
    test_dataset = JetTaggingDataset(
        test_events, test_targets, test_lengths, max_length, normalize=normalize, feature_stats=feature_stats
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, feature_stats or {}


if __name__ == '__main__':
    # Test data loading
    train_loader, val_loader, test_loader, stats = create_jet_tagging_dataloaders(
        n_train=100, n_val=20, n_test=20, batch_size=8, max_length=128
    )
    
    print(f"Feature stats: {stats}")
    
    # Test a batch
    for features, targets, masks in train_loader:
        print(f"Batch features shape: {features.shape}")  # (B, T, feature_dim)
        print(f"Batch targets shape: {targets.shape}")  # (B,)
        print(f"Batch masks shape: {masks.shape}")  # (B, T)
        print(f"Mask example: {masks[0][:20]}")
        print(f"Targets example: {targets[:10]}")
        break

