"""
Generate synthetic particle physics events for testing.
Creates events with variable numbers of particles and computes missing momentum.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def generate_particle_event(
    n_particles: int,
    device: str = 'cpu',
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a single particle event with realistic distributions.
    
    Args:
        n_particles: Number of particles in the event
        device: Device to create tensors on
        seed: Random seed for reproducibility
    
    Returns:
        particles: (n_particles, 6) tensor with [pT, eta, phi, mass, charge, pid]
        missing_momentum: (4,) tensor with [px, py, pz, E] of missing momentum
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Generate particle features with realistic distributions
    # pT: exponential distribution (typical for HEP)
    # Sample from exponential distribution using inverse transform
    uniform_samples = torch.rand(n_particles, device=device)
    pT = -torch.log(1 - uniform_samples) * 0.1  # Exponential with rate 0.1
    pT = torch.clamp(pT, 0.1, 100.0)  # GeV
    
    # eta: uniform in [-2.5, 2.5] (typical detector coverage)
    eta = torch.rand(n_particles, device=device) * 5.0 - 2.5
    
    # phi: uniform in [0, 2*pi]
    phi = torch.rand(n_particles, device=device) * 2 * np.pi
    
    # mass: mostly light particles (pions, kaons), some heavier
    mass_dist = torch.rand(n_particles, device=device)
    mass = torch.where(
        mass_dist < 0.7,
        torch.tensor(0.14, device=device),  # pion mass
        torch.where(
            mass_dist < 0.9,
            torch.tensor(0.5, device=device),  # kaon mass
            torch.tensor(1.0, device=device)  # proton-like
        )
    )
    
    # charge: mostly charged particles
    charge = torch.where(
        torch.rand(n_particles, device=device) < 0.8,
        torch.sign(torch.randn(n_particles, device=device)),  # +1 or -1
        torch.tensor(0.0, device=device)  # neutral
    )
    
    # PID: simple encoding (0=photon, 1=electron, 2=muon, 3=pion, 4=kaon, 5=proton)
    pid = torch.randint(0, 6, (n_particles,), device=device, dtype=torch.float)
    
    # Stack features: [pT, eta, phi, mass, charge, pid]
    particles = torch.stack([pT, eta, phi, mass, charge, pid], dim=1)
    
    # Compute 4-momentum for each particle
    px = pT * torch.cos(phi)
    py = pT * torch.sin(phi)
    pz = pT * torch.sinh(eta)
    E = torch.sqrt(px**2 + py**2 + pz**2 + mass**2)
    
    # Sum all visible momenta
    total_px = px.sum()
    total_py = py.sum()
    total_pz = pz.sum()
    total_E = E.sum()
    
    # Missing momentum (negative of visible sum, typical in HEP)
    # Clamp to reasonable range to avoid extreme values
    missing_px = torch.clamp(-total_px, -1000.0, 1000.0)
    missing_py = torch.clamp(-total_py, -1000.0, 1000.0)
    missing_pz = torch.clamp(-total_pz, -1000.0, 1000.0)
    missing_E = torch.clamp(-total_E, -1000.0, 1000.0)
    
    missing_momentum = torch.stack([missing_px, missing_py, missing_pz, missing_E])
    
    return particles, missing_momentum


def generate_hep_dataset(
    n_events: int,
    min_particles: int = 50,
    max_particles: int = 500,
    device: str = 'cpu',
    seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate a dataset of particle events.
    
    Args:
        n_events: Number of events to generate
        min_particles: Minimum particles per event
        max_particles: Maximum particles per event
        device: Device to create tensors on
        seed: Random seed for reproducibility
    
    Returns:
        events: List of (n_particles_i, 6) tensors
        targets: (n_events, 4) tensor with missing momentum [px, py, pz, E]
        lengths: (n_events,) tensor with number of particles per event
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    events = []
    targets = []
    lengths = []
    
    for i in range(n_events):
        # Variable number of particles per event
        n_particles = torch.randint(min_particles, max_particles + 1, (1,), device=device).item()
        
        particles, missing_momentum = generate_particle_event(n_particles, device=device)
        
        events.append(particles)
        targets.append(missing_momentum)
        lengths.append(n_particles)
    
    targets = torch.stack(targets)  # (n_events, 4)
    lengths = torch.tensor(lengths, device=device, dtype=torch.long)
    
    return events, targets, lengths


if __name__ == '__main__':
    # Test generation
    events, targets, lengths = generate_hep_dataset(10, min_particles=20, max_particles=100)
    print(f"Generated {len(events)} events")
    print(f"Particle counts: {lengths.tolist()}")
    print(f"Target shape: {targets.shape}")
    print(f"First event shape: {events[0].shape}")

