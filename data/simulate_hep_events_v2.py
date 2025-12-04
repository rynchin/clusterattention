"""
Improved HEP event simulation for attention benchmarking.

Key improvements over v1:
1. Events contain a hidden "resonance" decay that creates correlations
2. Background particles add noise the model must filter out  
3. Target requires learning particle relationships, not just sums
4. Multiple difficulty levels for scaling analysis

Task: Reconstruct the 4-momentum of an invisible particle from a resonance decay.
The resonance (e.g., Z boson) decays to one visible + one invisible particle.
Additional "pileup" particles add noise. The model must identify the decay products
and use momentum conservation + invariant mass constraints.
"""

import torch
import numpy as np
from typing import Tuple, Optional, List


def lorentz_boost(p4: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """
    Boost a 4-momentum by velocity beta.
    p4: (..., 4) tensor [E, px, py, pz]
    beta: (..., 3) tensor [bx, by, bz]
    Returns boosted 4-momentum.
    """
    E, px, py, pz = p4[..., 0], p4[..., 1], p4[..., 2], p4[..., 3]
    bx, by, bz = beta[..., 0], beta[..., 1], beta[..., 2]
    
    b2 = bx**2 + by**2 + bz**2
    gamma = 1.0 / torch.sqrt(1.0 - b2 + 1e-8)
    
    bp = bx * px + by * py + bz * pz
    gamma2 = (gamma - 1.0) / (b2 + 1e-8)
    
    E_new = gamma * (E - bp)
    px_new = px + gamma2 * bp * bx - gamma * bx * E
    py_new = py + gamma2 * bp * by - gamma * by * E
    pz_new = pz + gamma2 * bp * bz - gamma * bz * E
    
    return torch.stack([E_new, px_new, py_new, pz_new], dim=-1)


def generate_resonance_event(
    n_background: int,
    resonance_mass: float = 91.2,  # Z boson mass in GeV
    visible_mass: float = 0.105,   # muon mass
    invisible_mass: float = 0.0,   # neutrino mass
    resonance_pt_scale: float = 50.0,
    background_pt_scale: float = 5.0,
    device: str = 'cpu',
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Generate a single event with a resonance decay + background.
    
    The resonance decays: R -> visible + invisible
    The visible particle is marked with a special feature.
    Background particles are generated randomly.
    
    Returns:
        particles: (n_particles, 7) tensor with features:
            [pT, eta, phi, mass, charge, pid, is_signal]
            is_signal=1 for the visible decay product, 0 for background
        invisible_p4: (4,) tensor [px, py, pz, E] of invisible particle
        signal_idx: index of the signal particle
    """
    # Generate resonance kinematics in lab frame
    # Resonance has some pT and rapidity (clamped to avoid extreme boosts)
    R_pt = torch.abs(torch.randn(1, device=device)) * resonance_pt_scale
    R_pt = torch.clamp(R_pt, 10.0, 200.0)
    R_eta = torch.randn(1, device=device) * 1.5  # rapidity - reduced range
    R_eta = torch.clamp(R_eta, -3.0, 3.0)
    R_phi = torch.rand(1, device=device) * 2 * np.pi
    R_mass = torch.tensor([resonance_mass], device=device)
    
    # Resonance 4-momentum
    R_px = R_pt * torch.cos(R_phi)
    R_py = R_pt * torch.sin(R_phi)
    R_pz = R_pt * torch.sinh(R_eta)
    R_E = torch.sqrt(R_px**2 + R_py**2 + R_pz**2 + R_mass**2)
    
    # Decay in resonance rest frame
    # Two-body decay kinematics
    m1, m2 = visible_mass, invisible_mass
    M = resonance_mass
    
    # Momentum magnitude in rest frame (from two-body decay formula)
    p_star_sq = (M**2 - (m1 + m2)**2) * (M**2 - (m1 - m2)**2)
    p_star = np.sqrt(max(p_star_sq, 1e-8)) / (2 * M)
    
    # Random decay direction in rest frame
    cos_theta = 2 * torch.rand(1, device=device).item() - 1
    sin_theta = np.sqrt(1 - cos_theta**2)
    phi_decay = torch.rand(1, device=device).item() * 2 * np.pi
    
    # Visible particle in rest frame
    vis_px_rest = p_star * sin_theta * np.cos(phi_decay)
    vis_py_rest = p_star * sin_theta * np.sin(phi_decay)
    vis_pz_rest = p_star * cos_theta
    vis_E_rest = np.sqrt(p_star**2 + m1**2)
    
    # Invisible particle in rest frame (back-to-back)
    inv_px_rest = -vis_px_rest
    inv_py_rest = -vis_py_rest
    inv_pz_rest = -vis_pz_rest
    inv_E_rest = np.sqrt(p_star**2 + m2**2)
    
    # Boost to lab frame
    # Beta of resonance
    beta = torch.tensor([
        (R_px / R_E).item(), 
        (R_py / R_E).item(), 
        (R_pz / R_E).item()
    ], device=device)
    
    vis_p4_rest = torch.tensor([vis_E_rest, vis_px_rest, vis_py_rest, vis_pz_rest], device=device)
    inv_p4_rest = torch.tensor([inv_E_rest, inv_px_rest, inv_py_rest, inv_pz_rest], device=device)
    
    vis_p4_lab = lorentz_boost(vis_p4_rest, -beta)  # negative beta for boost from rest to lab
    inv_p4_lab = lorentz_boost(inv_p4_rest, -beta)
    
    # Extract visible particle kinematics
    vis_E = vis_p4_lab[0].item()
    vis_px = vis_p4_lab[1].item()
    vis_py = vis_p4_lab[2].item()
    vis_pz = vis_p4_lab[3].item()
    vis_pt = np.sqrt(vis_px**2 + vis_py**2)
    vis_eta = np.arcsinh(vis_pz / (vis_pt + 1e-8))
    vis_phi = np.arctan2(vis_py, vis_px)
    
    # Create visible particle features
    # [pT, eta, phi, mass, charge, pid, is_signal]
    signal_particle = torch.tensor([
        vis_pt, vis_eta, vis_phi,
        visible_mass,
        1.0,  # charge (muon-like)
        2.0,  # pid (muon)
        1.0,  # is_signal flag
    ], device=device)
    
    # Generate background particles
    background = []
    for _ in range(n_background):
        # Background pT: exponential distribution
        bg_pt = torch.abs(torch.randn(1, device=device)) * background_pt_scale
        bg_pt = torch.clamp(bg_pt, 0.5, 50.0)
        
        bg_eta = torch.randn(1, device=device) * 2.5
        bg_eta = torch.clamp(bg_eta, -4.0, 4.0)
        
        bg_phi = torch.rand(1, device=device) * 2 * np.pi
        
        # Random mass (pion-like mostly)
        bg_mass = torch.where(
            torch.rand(1, device=device) < 0.8,
            torch.tensor([0.14], device=device),
            torch.tensor([0.5], device=device)
        )
        
        bg_charge = torch.sign(torch.randn(1, device=device))
        bg_pid = torch.randint(0, 6, (1,), device=device, dtype=torch.float)
        bg_is_signal = torch.tensor([0.0], device=device)
        
        bg_particle = torch.cat([bg_pt, bg_eta, bg_phi, bg_mass, bg_charge, bg_pid, bg_is_signal])
        background.append(bg_particle)
    
    # Combine and shuffle
    all_particles = [signal_particle] + background
    
    # Random permutation
    perm = torch.randperm(len(all_particles))
    particles = torch.stack([all_particles[i] for i in perm])
    
    # Find where signal ended up
    signal_idx = (perm == 0).nonzero(as_tuple=True)[0].item()
    
    # Target: invisible particle 4-momentum [px, py, pz, E]
    invisible_p4 = torch.tensor([
        inv_p4_lab[1].item(), 
        inv_p4_lab[2].item(), 
        inv_p4_lab[3].item(), 
        inv_p4_lab[0].item()
    ], device=device)
    
    return particles, invisible_p4, signal_idx


def generate_resonance_dataset(
    n_events: int,
    min_background: int = 20,
    max_background: int = 200,
    resonance_mass: float = 91.2,
    difficulty: str = 'medium',
    device: str = 'cpu',
    seed: Optional[int] = None
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Generate a dataset of resonance events.
    
    Difficulty levels:
    - 'easy': Signal particle has is_signal=1 feature exposed
    - 'medium': is_signal feature is zeroed out (model must learn to identify signal)
    - 'hard': More background, lower signal pT
    
    Returns:
        events: List of (n_particles_i, feature_dim) tensors
        targets: (n_events, 4) tensor with invisible particle [px, py, pz, E]
        lengths: (n_events,) tensor with number of particles per event
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Difficulty settings
    if difficulty == 'easy':
        expose_signal = True
        resonance_pt_scale = 80.0
        background_pt_scale = 3.0
    elif difficulty == 'medium':
        expose_signal = False
        resonance_pt_scale = 50.0
        background_pt_scale = 5.0
    elif difficulty == 'hard':
        expose_signal = False
        resonance_pt_scale = 30.0
        background_pt_scale = 8.0
        min_background = min_background * 2
        max_background = max_background * 2
    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")
    
    events = []
    targets = []
    lengths = []
    
    for _ in range(n_events):
        n_bg = torch.randint(min_background, max_background + 1, (1,)).item()
        
        particles, invisible_p4, signal_idx = generate_resonance_event(
            n_background=n_bg,
            resonance_mass=resonance_mass,
            resonance_pt_scale=resonance_pt_scale,
            background_pt_scale=background_pt_scale,
            device=device,
        )
        
        # Remove is_signal feature for medium/hard difficulty
        if not expose_signal:
            particles = particles[:, :6]  # Remove is_signal column
        
        events.append(particles)
        targets.append(invisible_p4)
        lengths.append(len(particles))
    
    targets = torch.stack(targets)
    lengths = torch.tensor(lengths, device=device, dtype=torch.long)
    
    return events, targets, lengths


def generate_jet_tagging_dataset(
    n_events: int,
    n_particles_per_jet: int = 50,
    device: str = 'cpu',
    seed: Optional[int] = None
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Alternative task: Classify jets as quark-initiated vs gluon-initiated.
    
    Quark jets: Narrower, fewer particles, harder fragmentation
    Gluon jets: Wider, more particles, softer fragmentation
    
    Returns:
        events: List of (n_particles, 6) tensors
        targets: (n_events,) tensor with binary labels (0=quark, 1=gluon)
        lengths: (n_events,) tensor
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    events = []
    targets = []
    lengths = []
    
    for _ in range(n_events):
        is_gluon = torch.rand(1).item() > 0.5
        
        if is_gluon:
            # Gluon jet: more particles, wider, softer
            n_particles = int(n_particles_per_jet * (1.0 + 0.5 * torch.rand(1).item()))
            width = 0.4 + 0.2 * torch.rand(1).item()
            pt_scale = 3.0
        else:
            # Quark jet: fewer particles, narrower, harder
            n_particles = int(n_particles_per_jet * (0.6 + 0.4 * torch.rand(1).item()))
            width = 0.2 + 0.1 * torch.rand(1).item()
            pt_scale = 5.0
        
        # Jet axis
        jet_eta = torch.randn(1, device=device) * 2.0
        jet_phi = torch.rand(1, device=device) * 2 * np.pi
        
        particles = []
        for _ in range(n_particles):
            # Particle position relative to jet axis
            delta_eta = torch.randn(1, device=device) * width
            delta_phi = torch.randn(1, device=device) * width
            
            pt = torch.abs(torch.randn(1, device=device)) * pt_scale + 0.5
            eta = jet_eta + delta_eta
            phi = (jet_phi + delta_phi) % (2 * np.pi)
            
            mass = torch.tensor([0.14], device=device)  # pion mass
            charge = torch.sign(torch.randn(1, device=device))
            pid = torch.tensor([3.0], device=device)  # pion
            
            particle = torch.cat([pt, eta, phi, mass, charge, pid])
            particles.append(particle)
        
        events.append(torch.stack(particles))
        targets.append(torch.tensor([1.0 if is_gluon else 0.0], device=device))
        lengths.append(n_particles)
    
    targets = torch.cat(targets)
    lengths = torch.tensor(lengths, device=device, dtype=torch.long)
    
    return events, targets, lengths


if __name__ == '__main__':
    print("=== Testing Resonance Reconstruction Dataset ===\n")
    
    for difficulty in ['easy', 'medium', 'hard']:
        events, targets, lengths = generate_resonance_dataset(
            n_events=1000,
            min_background=20,
            max_background=100,
            difficulty=difficulty,
            seed=42
        )
        
        print(f"Difficulty: {difficulty}")
        print(f"  Events: {len(events)}")
        print(f"  Feature dim: {events[0].shape[1]}")
        print(f"  Particles per event: min={lengths.min()}, max={lengths.max()}, mean={lengths.float().mean():.1f}")
        print(f"  Target shape: {targets.shape}")
        
        # Target statistics
        for i, name in enumerate(['inv_px', 'inv_py', 'inv_pz', 'inv_E']):
            t = targets[:, i]
            print(f"  {name}: mean={t.mean():.2f}, std={t.std():.2f}, range=[{t.min():.2f}, {t.max():.2f}]")
        
        # Check if target correlates trivially with length
        corr_E = np.corrcoef(lengths.numpy(), targets[:, 3].numpy())[0, 1]
        print(f"  Correlation(length, E): {corr_E:.4f}")
        print()
    
    print("=== Testing Jet Tagging Dataset ===\n")
    events, targets, lengths = generate_jet_tagging_dataset(n_events=500, seed=42)
    print(f"Events: {len(events)}")
    print(f"Class balance: {targets.mean():.2f} (should be ~0.5)")
    print(f"Particles per event: min={lengths.min()}, max={lengths.max()}")

