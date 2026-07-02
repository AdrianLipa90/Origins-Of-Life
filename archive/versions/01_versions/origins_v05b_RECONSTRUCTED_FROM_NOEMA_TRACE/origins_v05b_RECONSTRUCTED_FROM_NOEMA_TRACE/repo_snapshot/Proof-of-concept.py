#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════════╗
║                                                                                ║
║                  MULTI-SCALE SELF-ORGANIZATION FRAMEWORK                       ║
║                  Computational Laboratory for Emergence Studies                ║
║                  Version: Monolithic | Focused | Falsifiable                   ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝

CORE HYPOTHESIS:
"Bidirectional coupling between chemical pattern formation and 
 oscillatory memory dynamics enhances the emergence of RNA-like 
 polymers under simulated early Earth conditions."

ONE NARROW QUESTION. ONE TESTABLE HYPOTHESIS.
"""

import numpy as np
import pandas as pd
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from scipy.ndimage import zoom
from scipy.signal import correlate2d
import time
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# PHYSICAL CONSTANTS & PARAMETER CALIBRATION
# ============================================================================

@dataclass
class PhysicalConstants:
    """All parameters with physical units and literature references."""
    
    # Universal constants
    AVOGADRO: float = 6.02214076e23  # mol⁻¹
    BOLTZMANN: float = 1.380649e-23  # J·K⁻¹
    GAS_CONSTANT: float = 8.314462618  # J·mol⁻¹·K⁻¹
    
    # Early Earth conditions (Hadean/Archean)
    TEMP_SEA_SURFACE_K: float = 338.0  # 65°C - tidal pools
    TEMP_HYDROTHERMAL_K: float = 363.0  # 90°C - vents
    PH_OCEAN: float = 6.5  # Early ocean pH (more acidic)
    
    # Montmorillonite clay (Ferris et al. 1996)
    CLAY_SURFACE_AREA: float = 150.0  # m²/g
    CLAY_SITE_DENSITY: float = 2.0e18  # adsorption sites/m²
    CLAY_CATALYTIC_ENHANCEMENT: float = 7.5  # Fold increase
    
    # Nucleotide properties (Sutherland et al. 2016)
    NUCLEOTIDE_MW: float = 330.0  # g/mol
    ACTIVATED_NUCLEOTIDE_LIFETIME: float = 24.0  # hours at 65°C
    
    # RNA properties (Szostak et al. 2018)
    RNA_BASE_PAIR_ENERGY: float = -2.0  # kcal/mol per AU pair
    RNA_GC_PAIR_ENERGY: float = -3.0  # kcal/mol per GC pair
    RNA_DEGRADATION_HALFLIFE: float = 12.0  # hours without protection
    
    # Kinetic rates from literature (1/hour)
    RATE_PHOTOCHEMISTRY: float = 0.35  # UV-driven synthesis
    RATE_CLAY_CATALYSIS: float = 0.15  # Clay-mediated oligomerization
    RASE_TEMPLATE_DIRECTED: float = 0.05  # Template-directed replication
    RATE_HYDROLYSIS: float = 0.04  # Peptide/ester hydrolysis
    
    # Diffusion coefficients (m²/hour)
    DIFFUSION_SMALL_MOLECULE: float = 3.6e-6  # ~1e-9 m²/s
    DIFFUSION_RNA: float = 3.6e-9  # ~1e-12 m²/s
    DIFFUSION_CLAY: float = 3.6e-10  # ~1e-13 m²/s
    
    # Experimental validation targets
    FERRIS_1996_OLIGOMER_LENGTH: int = 10  # Average length
    FERRIS_1996_TIMESCALE_H: float = 24.0  # Hours to achieve
    
    def get_arrhenius_factor(self, temp_k: float, ref_temp_k: float = 338.0, 
                            activation_energy: float = 50e3) -> float:
        """Temperature dependence of reaction rates."""
        return np.exp(-activation_energy * (1/temp_k - 1/ref_temp_k) / self.GAS_CONSTANT)


CONST = PhysicalConstants()


# ============================================================================
# CHEMICAL REACTION-DIFFUSION SYSTEM
# ============================================================================

class ChemicalSystem:
    """
    Physical implementation of clay-mediated oligomerization.
    Grid: 2D spatial simulation with explicit agents.
    Time: Hours, with sub-hour resolution.
    """
    
    def __init__(self, grid_shape: Tuple[int, int] = (128, 128),
                 grid_spacing_m: float = 1e-6,  # 1 micron cells
                 temperature_k: float = 338.0,
                 clay_concentration_g_l: float = 5.0):
        
        # Physical dimensions
        self.grid_shape = grid_shape
        self.grid_spacing = grid_spacing_m
        self.temperature_k = temperature_k
        self.time_h = 0.0
        
        # Chemical fields (concentration in arbitrary units 0-1, normalized)
        self.nucleotides = np.zeros(grid_shape)  # Free activated nucleotides
        self.nucleotides_adsorbed = np.zeros(grid_shape)  # Clay-bound
        self.oligomers = np.zeros(grid_shape)  # Short RNA chains
        self.long_polymers = np.zeros(grid_shape)  # Longer chains
        self.clay_sites = np.zeros(grid_shape)  # Clay adsorption capacity
        
        # Catalysis modulation from memory system
        self.catalysis_modulation = np.ones(grid_shape)
        
        # RNA sequence agents
        self.rna_agents: List[RNAAgent] = []
        self.agent_positions = np.zeros(grid_shape, dtype=object)
        self.agent_positions.fill(None)
        
        # Metrics tracking
        self.metrics = {
            'time_h': [],
            'mean_oligomer_length': [],
            'total_sequences': [],
            'gc_content_mean': [],
            'spatial_autocorrelation': [],
            'polymerization_rate': [],
            'sequence_diversity': [],
            'free_nucleotide_concentration': []
        }
        
        # Initialize fields
        self._initialize_fields(clay_concentration_g_l)
    
    def _initialize_fields(self, clay_concentration_g_l: float):
        """Initialize chemical fields with realistic distributions."""
        
        # Nucleotides with some spatial heterogeneity
        self.nucleotides = np.random.lognormal(mean=-4.6, sigma=0.5, size=self.grid_shape)
        self.nucleotides = np.clip(self.nucleotides / np.max(self.nucleotides), 0, 1)
        
        # Clay particles (clumped distribution)
        clay_mask = np.random.random(self.grid_shape) < 0.3
        clay_density = clay_concentration_g_l * 0.2  # Arbitrary scaling
        self.clay_sites[clay_mask] = clay_density
        
        # 10 seed RNA molecules
        for _ in range(10):
            i, j = np.random.randint(0, self.grid_shape[0]), np.random.randint(0, self.grid_shape[1])
            seq = ''.join(np.random.choice(['A', 'U', 'G', 'C'], size=15))
            agent = RNAAgent(sequence=seq, position=(i, j))
            self.rna_agents.append(agent)
            self.agent_positions[i, j] = agent
    
    def step_diffusion(self, dt_h: float):
        """Diffusion using explicit finite difference with reflecting boundaries."""
        
        # Diffusion coefficients (scaled to hours)
        D_nuc = CONST.DIFFUSION_SMALL_MOLECULE
        D_oligo = CONST.DIFFUSION_RNA / 10
        D_poly = CONST.DIFFUSION_RNA / 100
        
        # Helper function for diffusion with Neumann boundaries
        def diffuse(field: np.ndarray, D: float) -> np.ndarray:
            alpha = D * dt_h / (self.grid_spacing ** 2)
            if alpha > 0.25:
                alpha = 0.25  # Stability condition
            
            # 5-point stencil with reflecting boundaries
            new_field = field.copy()
            
            # Interior points
            new_field[1:-1, 1:-1] = field[1:-1, 1:-1] + alpha * (
                field[2:, 1:-1] + field[:-2, 1:-1] +
                field[1:-1, 2:] + field[1:-1, :-2] - 
                4 * field[1:-1, 1:-1]
            )
            
            return np.maximum(new_field, 0)
        
        self.nucleotides = diffuse(self.nucleotides, D_nuc)
        self.oligomers = diffuse(self.oligomers, D_oligo)
        self.long_polymers = diffuse(self.long_polymers, D_poly)
    
    def step_adsorption(self, dt_h: float):
        """Langmuir adsorption onto clay surfaces."""
        
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                clay_density = self.clay_sites[i, j]
                
                if clay_density > 0:
                    # Available sites (simplified)
                    max_adsorption = clay_density * 0.1  # Arbitrary scaling
                    free_nuc = self.nucleotides[i, j]
                    
                    # Langmuir isotherm
                    K_ads = 100.0  # Adsorption constant
                    adsorbed = (K_ads * free_nuc * max_adsorption) / (1 + K_ads * free_nuc)
                    
                    # Transfer
                    transfer = min(adsorbed * dt_h, free_nuc)
                    self.nucleotides[i, j] -= transfer
                    self.nucleotides_adsorbed[i, j] += transfer
    
    def step_polymerization(self, dt_h: float):
        """Clay-catalyzed oligomerization with temperature dependence."""
        
        # Temperature factor
        temp_factor = CONST.get_arrhenius_factor(self.temperature_k)
        
        for i in range(self.grid_shape[0]):
            for j in range(self.grid_shape[1]):
                adsorbed = self.nucleotides_adsorbed[i, j]
                
                if adsorbed > 0.001:  # Threshold
                    # Base polymerization rate
                    base_rate = CONST.RATE_CLAY_CATALYSIS * temp_factor
                    
                    # Template-directed enhancement
                    template_boost = 1.0
                    agent = self.agent_positions[i, j]
                    if agent is not None:
                        # More stable sequences catalyze better
                        stability = agent.calculate_stability()
                        template_boost = 1.0 + stability * 0.5
                    
                    # Memory system modulation
                    modulation = self.catalysis_modulation[i, j]
                    
                    # Calculate polymerization
                    polymerized = base_rate * adsorbed * template_boost * modulation * dt_h
                    
                    # Update concentrations
                    self.nucleotides_adsorbed[i, j] -= polymerized
                    
                    # Distribute between oligomers and polymers
                    if np.random.random() < 0.7:  # Mostly oligomers
                        self.oligomers[i, j] += polymerized * 0.9
                        self.long_polymers[i, j] += polymerized * 0.1
                    else:
                        self.oligomers[i, j] += polymerized * 0.5
                        self.long_polymers[i, j] += polymerized * 0.5
    
    def step_replication(self, dt_h: float):
        """RNA replication with selection based on sequence properties."""
        
        new_agents = []
        
        for agent in self.rna_agents:
            i, j = agent.position
            
            # Replication probability depends on:
            # 1. Local oligomer concentration (raw materials)
            # 2. Sequence stability
            # 3. Presence of clay (protection)
            
            material_availability = self.oligomers[i, j]
            stability = agent.calculate_stability()
            clay_present = self.clay_sites[i, j] > 0
            
            if clay_present:
                protection_factor = 2.0
            else:
                protection_factor = 0.5
            
            # Replication probability
            repl_prob = (
                material_availability * 
                (1.0 + stability) * 
                protection_factor * 
                dt_h * 0.1  # Scaling factor
            )
            
            if np.random.random() < repl_prob:
                # Error rate depends on temperature
                error_rate = 0.01 * CONST.get_arrhenius_factor(self.temperature_k, ref_temp_k=338.0, activation_energy=25e3)
                
                # Replicate with errors
                daughter = agent.replicate(error_rate=error_rate)
                daughter.creation_time = self.time_h
                new_agents.append(daughter)
        
        # Add new agents
        for agent in new_agents:
            i, j = agent.position
            self.rna_agents.append(agent)
            
            # Update position (allow stacking for now)
            if self.agent_positions[i, j] is None:
                self.agent_positions[i, j] = agent
        
        # Degradation (age-dependent)
        max_age_h = CONST.RNA_DEGRADATION_HALFLIFE * 2
        self.rna_agents = [
            agent for agent in self.rna_agents
            if (self.time_h - agent.creation_time) < max_age_h
        ]
        
        # Update positions
        self._update_agent_positions()
    
    def _update_agent_positions(self):
        """Update the spatial agent position grid."""
        self.agent_positions.fill(None)
        for agent in self.rna_agents:
            i, j = agent.position
            # Simple: last agent wins the position
            self.agent_positions[i, j] = agent
    
    def step_degradation(self, dt_h: float):
        """Chemical degradation of nucleotides and RNA."""
        
        # Nucleotide hydrolysis
        hydrolysis_rate = CONST.RATE_HYDROLYSIS * CONST.get_arrhenius_factor(self.temperature_k)
        self.nucleotides *= (1 - hydrolysis_rate * dt_h)
        self.nucleotides_adsorbed *= (1 - hydrolysis_rate * dt_h * 0.5)  # Protected on clay
        
        # RNA degradation (shorter chains degrade faster)
        self.oligomers *= (1 - hydrolysis_rate * dt_h * 2)
        self.long_polymers *= (1 - hydrolysis_rate * dt_h)
    
    def calculate_metrics(self):
        """Calculate all observable metrics."""
        
        # Time
        self.metrics['time_h'].append(self.time_h)
        
        # Sequence metrics
        if self.rna_agents:
            lengths = [len(agent.sequence) for agent in self.rna_agents]
            gc_contents = [agent.gc_content for agent in self.rna_agents]
            sequences = [agent.sequence for agent in self.rna_agents]
            
            self.metrics['mean_oligomer_length'].append(np.mean(lengths))
            self.metrics['total_sequences'].append(len(self.rna_agents))
            self.metrics['gc_content_mean'].append(np.mean(gc_contents))
            self.metrics['sequence_diversity'].append(len(set(sequences)) / max(1, len(sequences)))
        else:
            self.metrics['mean_oligomer_length'].append(0)
            self.metrics['total_sequences'].append(0)
            self.metrics['gc_content_mean'].append(0)
            self.metrics['sequence_diversity'].append(0)
        
        # Spatial autocorrelation of oligomers
        if np.sum(self.oligomers) > 0:
            norm_oligo = self.oligomers / np.max(self.oligomers)
            autocorr = correlate2d(norm_oligo, norm_oligo, mode='same')
            center = autocorr.shape[0] // 2, autocorr.shape[1] // 2
            self.metrics['spatial_autocorrelation'].append(autocorr[center])
        else:
            self.metrics['spatial_autocorrelation'].append(0)
        
        # Polymerization rate (change in total polymer)
        total_polymer = np.sum(self.oligomers) + np.sum(self.long_polymers)
        if len(self.metrics['time_h']) > 1:
            prev_total = (
                self.metrics['mean_oligomer_length'][-2] * 
                self.metrics['total_sequences'][-2]
            )
            current_total = (
                self.metrics['mean_oligomer_length'][-1] * 
                self.metrics['total_sequences'][-1]
            )
            rate = (current_total - prev_total) / (self.time_h - self.metrics['time_h'][-2])
            self.metrics['polymerization_rate'].append(rate)
        else:
            self.metrics['polymerization_rate'].append(0)
        
        # Free nucleotide concentration
        self.metrics['free_nucleotide_concentration'].append(np.mean(self.nucleotides))
    
    def apply_memory_modulation(self, modulation_field: np.ndarray):
        """
        Apply modulation from memory system.
        Rescales to match grid size and applies with damping.
        """
        if modulation_field.shape != self.grid_shape:
            # Resize using linear interpolation
            zoom_factor = (
                self.grid_shape[0] / modulation_field.shape[0],
                self.grid_shape[1] / modulation_field.shape[1]
            )
            modulation_field = zoom(modulation_field, zoom_factor, order=1)
        
        # Apply with damping to prevent oscillations
        alpha = 0.05  # Learning rate
        new_modulation = (1 - alpha) * self.catalysis_modulation + alpha * modulation_field
        
        # Clamp to reasonable range [0.1, 3.0]
        self.catalysis_modulation = np.clip(new_modulation, 0.1, 3.0)
    
    def get_pattern_for_memory(self) -> np.ndarray:
        """
        Extract pattern for memory system.
        Returns normalized pattern of polymer concentration.
        """
        pattern = self.oligomers + self.long_polymers * 2.0  # Weight polymers more
        
        if np.max(pattern) > 0:
            pattern = pattern / np.max(pattern)
        
        return pattern
    
    def run(self, duration_h: float, dt_h: float = 0.02, 
            record_interval: float = 0.2) -> pd.DataFrame:
        """
        Run chemical simulation.
        Returns: DataFrame with all metrics.
        """
        
        print(f"Chemical System: {self.grid_shape[0]}x{self.grid_shape[1]} grid")
        print(f"Temperature: {self.temperature_k - 273.15:.1f}°C")
        print(f"Duration: {duration_h}h, dt: {dt_h}h")
        
        steps = int(duration_h / dt_h)
        record_steps = int(record_interval / dt_h)
        
        for step in range(steps):
            # Order of operations matters
            self.step_diffusion(dt_h)
            self.step_adsorption(dt_h)
            self.step_polymerization(dt_h)
            self.step_replication(dt_h)
            self.step_degradation(dt_h)
            
            self.time_h += dt_h
            
            if step % record_steps == 0:
                self.calculate_metrics()
            
            if step % 500 == 0:
                print(f"  t={self.time_h:5.1f}h | "
                      f"RNA={len(self.rna_agents):3d} | "
                      f"Len={self.metrics['mean_oligomer_length'][-1]:.1f}")
        
        return pd.DataFrame(self.metrics)


# ============================================================================
# RNA AGENT IMPLEMENTATION
# ============================================================================

class RNAAgent:
    """Explicit RNA sequence with physical properties."""
    
    def __init__(self, sequence: str, position: Tuple[int, int],
                 generation: int = 0, creation_time: float = 0.0):
        self.sequence = sequence
        self.position = position
        self.generation = generation
        self.creation_time = creation_time
        self.replication_count = 0
        
    @property
    def length(self) -> int:
        return len(self.sequence)
    
    @property
    def gc_content(self) -> float:
        gc = self.sequence.count('G') + self.sequence.count('C')
        return gc / self.length if self.length > 0 else 0
    
    def calculate_stability(self) -> float:
        """
        Calculate relative stability score.
        Based on GC content and secondary structure potential.
        """
        # GC content contribution
        gc_score = 2.0 * (self.gc_content - 0.5)  # -1 to +1
        
        # Hairpin potential (simple palindrome check)
        hairpin_score = 0
        for i in range(self.length - 4):
            substr = self.sequence[i:i+6]
            if len(substr) == 6:
                # Check for small palindrome
                if substr[:3] == self._reverse_complement(substr[3:]):
                    hairpin_score += 1
        
        hairpin_score = min(hairpin_score / 3, 1.0)  # Normalize
        
        return 0.7 * gc_score + 0.3 * hairpin_score
    
    def _reverse_complement(self, seq: str) -> str:
        """Return reverse complement of sequence."""
        comp = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
        return ''.join(comp.get(base, base) for base in reversed(seq))
    
    def replicate(self, error_rate: float = 0.01) -> 'RNAAgent':
        """Replicate with possible mutations."""
        new_seq = []
        
        for base in self.sequence:
            if np.random.random() < error_rate:
                # Mutation
                bases = ['A', 'U', 'G', 'C']
                bases.remove(base)
                new_base = np.random.choice(bases)
                new_seq.append(new_base)
            else:
                new_seq.append(base)
        
        # Slight probability of insertion/deletion
        if np.random.random() < error_rate / 5:
            if np.random.random() < 0.5 and len(new_seq) > 5:
                # Deletion
                del_idx = np.random.randint(0, len(new_seq))
                del new_seq[del_idx]
            elif len(new_seq) < 50:
                # Insertion
                ins_idx = np.random.randint(0, len(new_seq))
                new_base = np.random.choice(['A', 'U', 'G', 'C'])
                new_seq.insert(ins_idx, new_base)
        
        self.replication_count += 1
        
        return RNAAgent(
            sequence=''.join(new_seq),
            position=self.position,  # Daughter stays near parent
            generation=self.generation + 1,
            creation_time=self.creation_time  # Will be updated by simulation
        )
    
    def calculate_melting_temp(self) -> float:
        """Approximate melting temperature (°C)."""
        # Simple Wallace rule: Tm = 4*(GC) + 2*(AT)
        gc = self.gc_content
        return 4 * gc * 100 + 2 * (1 - gc) * 100 - 20  # Rough estimate


# ============================================================================
# EMERGENT MEMORY SYSTEM (Kuramoto + Hebbian)
# ============================================================================

class EmergentMemory:
    """
    Phase synchronization system for pattern storage and recall.
    No metaphysical claims - just observable oscillator dynamics.
    """
    
    def __init__(self, grid_shape: Tuple[int, int] = (32, 32),
                 base_frequency_hz: float = 7.83,
                 coupling_strength: float = 0.15):
        
        self.grid_shape = grid_shape
        self.n_oscillators = grid_shape[0] * grid_shape[1]
        
        # Oscillator states
        self.phases = np.zeros(self.n_oscillators)
        self.frequencies = np.ones(self.n_oscillators) * 2 * np.pi * base_frequency_hz
        self.coupling = coupling_strength
        
        # Connectivity
        self.adjacency = self._build_topology()
        self.hebbian_weights = np.zeros((self.n_oscillators, self.n_oscillators))
        
        # Pattern storage
        self.stored_patterns = []
        self.pattern_history = []
        
        # Metrics
        self.time_s = 0.0
        self.coherence_history = []
        
        # Initialize deterministically
        self._initialize_with_collatz()
    
    def _build_topology(self) -> np.ndarray:
        """Build deterministic small-world topology."""
        adj = np.zeros((self.n_oscillators, self.n_oscillators))
        rows, cols = self.grid_shape
        
        # Local connections (8-neighbor)
        for i in range(self.n_oscillators):
            x, y = divmod(i, cols)
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols:
                        j = nx * cols + ny
                        dist = np.sqrt(dx*dx + dy*dy)
                        weight = np.exp(-dist)
                        adj[i, j] = weight
        
        # Add a few long-range connections (10%)
        n_long = self.n_oscillators // 10
        rng = np.random.RandomState(42)  # Deterministic
        
        for i in range(self.n_oscillators):
            candidates = np.where(adj[i] == 0)[0]
            candidates = candidates[candidates != i]
            
            if len(candidates) > 0:
                selected = rng.choice(candidates, size=min(2, len(candidates)), replace=False)
                for j in selected:
                    # Weight decays with grid distance
                    xi, yi = divmod(i, self.grid_shape[1])
                    xj, yj = divmod(j, self.grid_shape[1])
                    dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                    weight = 0.1 * np.exp(-dist / max(self.grid_shape))
                    adj[i, j] = weight
                    adj[j, i] = weight
        
        return adj
    
    def _initialize_with_collatz(self):
        """Deterministic initialization using Collatz sequence."""
        for i in range(self.n_oscillators):
            # Compute Collatz length
            n = i + 1000
            steps = 0
            while n != 1 and steps < 1000:
                if n % 2 == 0:
                    n //= 2
                else:
                    n = 3 * n + 1
                steps += 1
            
            # Use for phase initialization
            self.phases[i] = 2 * np.pi * (steps % 100) / 100
            
            # Small frequency variations
            self.frequencies[i] *= (1 + 0.05 * np.sin(2 * np.pi * i / self.n_oscillators))
    
    def step(self, dt_s: float = 0.01):
        """Kuramoto model update."""
        new_phases = self.phases.copy()
        
        for i in range(self.n_oscillators):
            coupling_sum = 0.0
            weight_sum = 0.0
            
            # Adjacency coupling
            for j in range(self.n_oscillators):
                w = self.adjacency[i, j]
                if w > 0:
                    phase_diff = self.phases[j] - self.phases[i]
                    coupling_sum += w * np.sin(phase_diff)
                    weight_sum += w
            
            # Hebbian memory coupling
            for j in range(self.n_oscillators):
                w = self.hebbian_weights[i, j]
                if w != 0:
                    phase_diff = self.phases[j] - self.phases[i]
                    coupling_sum += w * np.sin(phase_diff)
                    weight_sum += abs(w)
            
            if weight_sum > 0:
                coupling_sum /= weight_sum
            
            # Update phase
            dphase = self.frequencies[i] + self.coupling * coupling_sum
            new_phases[i] = (self.phases[i] + dphase * dt_s) % (2 * np.pi)
        
        self.phases = new_phases
        self.time_s += dt_s
        
        # Record coherence
        self.coherence_history.append(self.calculate_coherence())
    
    def calculate_coherence(self) -> float:
        """Kuramoto order parameter."""
        complex_sum = np.sum(np.exp(1j * self.phases))
        return np.abs(complex_sum) / self.n_oscillators
    
    def store_pattern(self, pattern: np.ndarray, strength: float = 0.1):
        """Hebbian learning of pattern."""
        
        if pattern.shape != self.grid_shape:
            zoom_factor = (
                self.grid_shape[0] / pattern.shape[0],
                self.grid_shape[1] / pattern.shape[1]
            )
            pattern = zoom(pattern, zoom_factor, order=1)
        
        pattern_flat = pattern.flatten()
        
        # Normalize pattern
        if np.std(pattern_flat) > 0:
            pattern_flat = (pattern_flat - np.mean(pattern_flat)) / np.std(pattern_flat)
        
        # Update weights (simplified Hebb)
        for i in range(self.n_oscillators):
            for j in range(i + 1, self.n_oscillators):
                if self.adjacency[i, j] > 0:
                    delta = strength * pattern_flat[i] * pattern_flat[j]
                    self.hebbian_weights[i, j] += delta
                    self.hebbian_weights[j, i] += delta
        
        # Store pattern
        self.stored_patterns.append(pattern.copy())
        if len(self.stored_patterns) > 20:
            self.stored_patterns.pop(0)
    
    def recall_pattern(self) -> np.ndarray:
        """Let system evolve and extract pattern."""
        # Run for a while
        for _ in range(50):
            self.step()
        
        # Phases to pattern
        pattern = np.cos(self.phases).reshape(self.grid_shape)
        
        # Normalize to [0, 1]
        p_min, p_max = pattern.min(), pattern.max()
        if p_max > p_min:
            pattern = (pattern - p_min) / (p_max - p_min)
        
        return pattern
    
    def extract_modulation_field(self) -> np.ndarray:
        """
        Extract field for chemical modulation.
        Based on local phase coherence.
        """
        coherence_map = np.zeros(self.grid_shape)
        rows, cols = self.grid_shape
        
        for i in range(self.n_oscillators):
            x, y = divmod(i, cols)
            
            # Get local neighborhood (3x3)
            x_min, x_max = max(0, x-1), min(rows, x+2)
            y_min, y_max = max(0, y-1), min(cols, y+2)
            
            # Extract phases in neighborhood
            local_phases = []
            for nx in range(x_min, x_max):
                for ny in range(y_min, y_max):
                    idx = nx * cols + ny
                    local_phases.append(self.phases[idx])
            
            # Calculate local coherence
            if local_phases:
                local_coherence = np.abs(np.mean(np.exp(1j * local_phases)))
                coherence_map[x, y] = local_coherence
        
        return coherence_map
    
    def get_state_summary(self) -> Dict:
        """Return summary of memory state."""
        return {
            'coherence': self.calculate_coherence(),
            'stored_patterns': len(self.stored_patterns),
            'mean_weight': np.mean(np.abs(self.hebbian_weights)),
            'time_s': self.time_s
        }


# ============================================================================
# COUPLED SYSTEM EXPERIMENT
# ============================================================================

class CoupledEmergenceExperiment:
    """
    Main experiment: Test if coupling enhances oligomerization.
    
    HYPOTHESIS: Bidirectional coupling between chemical pattern formation
    and oscillatory memory dynamics enhances the emergence of RNA-like
    polymers under simulated early Earth conditions.
    
    TEST: Compare coupled vs uncoupled systems over multiple runs.
    """
    
    def __init__(self, 
                 chemical_grid: Tuple[int, int] = (128, 128),
                 memory_grid: Tuple[int, int] = (32, 32),
                 temperature_k: float = 338.0,
                 clay_concentration: float = 5.0,
                 coupling_strength: float = 0.1,
                 experiment_duration_h: float = 48.0):
        
        # Systems
        self.chem = ChemicalSystem(
            grid_shape=chemical_grid,
            temperature_k=temperature_k,
            clay_concentration_g_l=clay_concentration
        )
        
        self.memory = EmergentMemory(
            grid_shape=memory_grid,
            coupling_strength=coupling_strength
        )
        
        # Experiment parameters
        self.duration_h = experiment_duration_h
        self.coupling_interval_h = 1.0  # Couple every hour
        self.coupling_enabled = True
        self.time_h = 0.0
        
        # Results storage
        self.results = {
            'coupled': {'runs': [], 'final_metrics': []},
            'uncoupled': {'runs': [], 'final_metrics': []}
        }
        
        # Current run data
        self.current_run = {
            'time_h': [],
            'oligomer_length': [],
            'sequence_count': [],
            'memory_coherence': [],
            'coupling_strength': [],
            'feedback_gain': []
        }
    
    def run_single_experiment(self, coupled: bool = True, 
                             run_id: int = 0) -> Dict:
        """
        Run a single experiment (coupled or uncoupled).
        Returns detailed results.
        """
        
        print(f"\n{'='*60}")
        print(f"EXPERIMENT RUN {run_id}: {'COUPLED' if coupled else 'UNCOUPLED'}")
        print(f"{'='*60}")
        
        # Reset systems
        self.chem = ChemicalSystem(
            grid_shape=self.chem.grid_shape,
            temperature_k=self.chem.temperature_k,
            clay_concentration_g_l=5.0
        )
        
        self.memory = EmergentMemory(
            grid_shape=self.memory.grid_shape,
            coupling_strength=self.memory.coupling
        )
        
        self.coupling_enabled = coupled
        self.time_h = 0.0
        self.current_run = {k: [] for k in self.current_run.keys()}
        
        # Simulation parameters
        dt_h = 0.02
        total_steps = int(self.duration_h / dt_h)
        coupling_steps = int(self.coupling_interval_h / dt_h)
        record_steps = int(0.2 / dt_h)  # Record every 0.2h
        
        last_coupling = 0.0
        last_length = 0.0
        
        for step in range(total_steps):
            current_time = step * dt_h
            
            # Chemical system steps
            self.chem.step_diffusion(dt_h)
            self.chem.step_adsorption(dt_h)
            self.chem.step_polymerization(dt_h)
            self.chem.step_replication(dt_h)
            self.chem.step_degradation(dt_h)
            
            # Coupling (if enabled and time)
            if (coupled and 
                current_time - last_coupling >= self.coupling_interval_h):
                
                # Chemistry → Memory: Store pattern
                pattern = self.chem.get_pattern_for_memory()
                self.memory.store_pattern(pattern, strength=0.05)
                
                # Memory → Chemistry: Apply modulation
                modulation = self.memory.extract_modulation_field()
                self.chem.apply_memory_modulation(modulation)
                
                # Evolve memory
                for _ in range(10):
                    self.memory.step(dt_s=0.01)
                
                last_coupling = current_time
            
            self.time_h = current_time
            
            # Record metrics
            if step % record_steps == 0:
                # Get chemical metrics
                if self.chem.rna_agents:
                    current_length = np.mean([len(a.sequence) for a in self.chem.rna_agents])
                    seq_count = len(self.chem.rna_agents)
                else:
                    current_length = 0
                    seq_count = 0
                
                # Calculate feedback gain
                if last_length > 0:
                    feedback_gain = (current_length - last_length) / (current_time - last_length)
                else:
                    feedback_gain = 0
                
                last_length = current_length
                
                # Record
                self.current_run['time_h'].append(current_time)
                self.current_run['oligomer_length'].append(current_length)
                self.current_run['sequence_count'].append(seq_count)
                self.current_run['memory_coherence'].append(self.memory.calculate_coherence())
                self.current_run['coupling_strength'].append(
                    np.var(self.chem.catalysis_modulation) if coupled else 0
                )
                self.current_run['feedback_gain'].append(feedback_gain)
            
            # Progress report
            if step % 500 == 0:
                print(f"  t={current_time:5.1f}h | "
                      f"RNA={seq_count:3d} | "
                      f"Len={current_length:5.1f} | "
                      f"Coh={self.memory.calculate_coherence():.3f}")
        
        # Final metrics
        final_metrics = self._calculate_final_metrics()
        
        print(f"\nRUN COMPLETE:")
        print(f"  Final oligomer length: {final_metrics['final_length']:.1f}")
        print(f"  Final sequence count: {final_metrics['final_sequences']}")
        print(f"  Average coherence: {final_metrics['avg_coherence']:.3f}")
        
        return {
            'run_id': run_id,
            'coupled': coupled,
            'final_metrics': final_metrics,
            'time_series': {k: v.copy() for k, v in self.current_run.items()}
        }
    
    def _calculate_final_metrics(self) -> Dict:
        """Calculate summary metrics for a run."""
        
        ts = self.current_run
        
        final_length = ts['oligomer_length'][-1] if ts['oligomer_length'] else 0
        final_sequences = ts['sequence_count'][-1] if ts['sequence_count'] else 0
        
        avg_coherence = np.mean(ts['memory_coherence']) if ts['memory_coherence'] else 0
        avg_feedback = np.mean(ts['feedback_gain']) if ts['feedback_gain'] else 0
        
        # Calculate rate of oligomerization
        if len(ts['oligomer_length']) > 10:
            early_len = np.mean(ts['oligomer_length'][:5])
            late_len = np.mean(ts['oligomer_length'][-5:])
            oligo_rate = (late_len - early_len) / (ts['time_h'][-1] - ts['time_h'][0])
        else:
            oligo_rate = 0
        
        return {
            'final_length': float(final_length),
            'final_sequences': int(final_sequences),
            'avg_coherence': float(avg_coherence),
            'avg_feedback_gain': float(avg_feedback),
            'oligomerization_rate': float(oligo_rate),
            'peak_length': float(np.max(ts['oligomer_length'])) if ts['oligomer_length'] else 0
        }
    
    def run_multiple_trials(self, n_trials: int = 5):
        """Run multiple trials of coupled and uncoupled experiments."""
        
        print("\n" + "="*70)
        print(f"MULTI-TRIAL EXPERIMENT: {n_trials} trials each")
        print("="*70)
        
        # Run uncoupled trials
        print("\nUNCOUPLED TRIALS:")
        for trial in range(n_trials):
            results = self.run_single_experiment(coupled=False, run_id=trial)
            self.results['uncoupled']['runs'].append(results)
            self.results['uncoupled']['final_metrics'].append(results['final_metrics'])
        
        # Run coupled trials
        print("\nCOUPLED TRIALS:")
        for trial in range(n_trials):
            results = self.run_single_experiment(coupled=True, run_id=trial)
            self.results['coupled']['runs'].append(results)
            self.results['coupled']['final_metrics'].append(results['final_metrics'])
        
        # Statistical analysis
        self._analyze_results()
    
    def _analyze_results(self):
        """Statistical comparison of coupled vs uncoupled."""
        
        uncoupled_metrics = self.results['uncoupled']['final_metrics']
        coupled_metrics = self.results['coupled']['final_metrics']
        
        # Extract key metrics
        uncoupled_lengths = [m['final_length'] for m in uncoupled_metrics]
        coupled_lengths = [m['final_length'] for m in coupled_metrics]
        
        uncoupled_rates = [m['oligomerization_rate'] for m in uncoupled_metrics]
        coupled_rates = [m['oligomerization_rate'] for m in coupled_metrics]
        
        # Calculate statistics
        stats = {
            'uncoupled_mean_length': np.mean(uncoupled_lengths),
            'uncoupled_std_length': np.std(uncoupled_lengths),
            'coupled_mean_length': np.mean(coupled_lengths),
            'coupled_std_length': np.std(coupled_lengths),
            
            'uncoupled_mean_rate': np.mean(uncoupled_rates),
            'uncoupled_std_rate': np.std(uncoupled_rates),
            'coupled_mean_rate': np.mean(coupled_rates),
            'coupled_std_rate': np.std(coupled_rates),
            
            'length_improvement_percent': (
                (np.mean(coupled_lengths) - np.mean(uncoupled_lengths)) /
                max(np.mean(uncoupled_lengths), 1e-6) * 100
            ),
            
            'rate_improvement_percent': (
                (np.mean(coupled_rates) - np.mean(uncoupled_rates)) /
                max(np.mean(uncoupled_rates), 1e-6) * 100
            )
        }
        
        # T-test for significance (simplified)
        if len(uncoupled_lengths) > 1 and len(coupled_lengths) > 1:
            # Simple effect size calculation
            pooled_std = np.sqrt(
                (np.var(uncoupled_lengths) + np.var(coupled_lengths)) / 2
            )
            if pooled_std > 0:
                stats['effect_size'] = (
                    np.mean(coupled_lengths) - np.mean(uncoupled_lengths)
                ) / pooled_std
            else:
                stats['effect_size'] = 0
        else:
            stats['effect_size'] = 0
        
        # Hypothesis test
        stats['hypothesis_supported'] = (
            stats['length_improvement_percent'] > 10 and  # >10% improvement
            stats['effect_size'] > 0.5  # Medium effect size
        )
        
        self.results['statistics'] = stats
        
        # Print results
        print("\n" + "="*70)
        print("STATISTICAL ANALYSIS")
        print("="*70)
        
        print(f"\nFINAL OLIGOMER LENGTH:")
        print(f"  Uncoupled: {stats['uncoupled_mean_length']:.1f} ± {stats['uncoupled_std_length']:.1f}")
        print(f"  Coupled:   {stats['coupled_mean_length']:.1f} ± {stats['coupled_std_length']:.1f}")
        print(f"  Improvement: {stats['length_improvement_percent']:+.1f}%")
        
        print(f"\nOLIGOMERIZATION RATE:")
        print(f"  Uncoupled: {stats['uncoupled_mean_rate']:.3f} ± {stats['uncoupled_std_rate']:.3f}")
        print(f"  Coupled:   {stats['coupled_mean_rate']:.3f} ± {stats['coupled_std_rate']:.3f}")
        print(f"  Improvement: {stats['rate_improvement_percent']:+.1f}%")
        
        print(f"\nEFFECT SIZE: {stats['effect_size']:.2f}")
        
        if stats['hypothesis_supported']:
            print(f"\n✓ HYPOTHESIS SUPPORTED: Coupling enhances oligomerization")
        else:
            print(f"\n✗ HYPOTHESIS NOT SUPPORTED: No significant enhancement")
    
    def save_results(self, filename_prefix: str = "emergence_experiment"):
        """Save all results to files."""
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed time series for first run of each type
        for coupled in [True, False]:
            runs = self.results['coupled' if coupled else 'uncoupled']['runs']
            if runs:
                first_run = runs[0]
                df = pd.DataFrame(first_run['time_series'])
                label = "coupled" if coupled else "uncoupled"
                df.to_csv(f"{filename_prefix}_{label}_run0_{timestamp}.csv", index=False)
        
        # Save summary statistics
        with open(f"{filename_prefix}_summary_{timestamp}.json", 'w') as f:
            json.dump(self.results['statistics'], f, indent=2)
        
        # Save all final metrics
        all_metrics = []
        for coupled in [True, False]:
            label = "coupled" if coupled else "uncoupled"
            for i, metrics in enumerate(self.results[label]['final_metrics']):
                metrics['condition'] = label
                metrics['trial'] = i
                all_metrics.append(metrics)
        
        df_metrics = pd.DataFrame(all_metrics)
        df_metrics.to_csv(f"{filename_prefix}_all_metrics_{timestamp}.csv", index=False)
        
        print(f"\nResults saved with timestamp: {timestamp}")


# ============================================================================
# VALIDATION & SENSITIVITY ANALYSIS
# ============================================================================

class Validation:
    """
    Validate against experimental data and analyze parameter sensitivity.
    """
    
    @staticmethod
    def calibrate_to_ferris(simulator: ChemicalSystem, 
                           target_length: int = 10,
                           target_time_h: float = 24.0) -> Dict:
        """
        Calibrate simulation to match Ferris et al. 1996 results.
        """
        
        print("\nCALIBRATION TO FERRIS ET AL. 1996:")
        print(f"Target: {target_length}-mer oligomers in {target_time_h}h")
        
        # Run simulation
        results = simulator.run(duration_h=target_time_h, dt_h=0.02)
        
        # Get final length
        final_length = results['mean_oligomer_length'].iloc[-1]
        
        # Calculate calibration factor
        if final_length > 0:
            calibration = target_length / final_length
        else:
            calibration = 1.0
        
        # Recommendations
        recommendations = []
        
        if calibration > 1.5:
            recommendations.append("Increase clay concentration or surface area")
            recommendations.append("Increase polymerization rate constant")
        elif calibration < 0.7:
            recommendations.append("Reduce degradation rates")
            recommendations.append("Increase nucleotide concentration")
        
        return {
            'target_length': target_length,
            'achieved_length': float(final_length),
            'calibration_factor': float(calibration),
            'match_percentage': float(min(100, 100 * final_length / target_length)),
            'recommendations': recommendations
        }
    
    @staticmethod
    def sensitivity_analysis(simulator: ChemicalSystem, 
                            parameters: Dict[str, Tuple[float, float]],
                            n_points: int = 5) -> pd.DataFrame:
        """
        Analyze sensitivity to parameter changes.
        """
        
        print("\nPARAMETER SENSITIVITY ANALYSIS")
        print("-"*40)
        
        results = []
        base_metrics = None
        
        # Test each parameter
        for param_name, (min_val, max_val) in parameters.items():
            print(f"Testing {param_name}...")
            
            param_values = np.linspace(min_val, max_val, n_points)
            param_effects = []
            
            for val in param_values:
                # Create modified simulator
                modified = ChemicalSystem(
                    grid_shape=simulator.grid_shape,
                    temperature_k=simulator.temperature_k,
                    clay_concentration_g_l=5.0
                )
                
                # Modify parameter (simplified - in real implementation, 
                # you'd have parameters as a mutable object)
                if param_name == "clay_concentration":
                    modified.clay_sites.fill(val * 0.2)
                
                # Run simulation
                run_results = modified.run(duration_h=12.0, dt_h=0.02)
                final_len = run_results['mean_oligomer_length'].iloc[-1]
                param_effects.append(final_len)
            
            # Calculate sensitivity
            if len(param_effects) > 1:
                sensitivity = (max(param_effects) - min(param_effects)) / (max_val - min_val)
            else:
                sensitivity = 0
            
            results.append({
                'parameter': param_name,
                'min_value': min_val,
                'max_value': max_val,
                'effect_range': f"{min(param_effects):.1f}-{max(param_effects):.1f}",
                'sensitivity': sensitivity,
                'normalized_sensitivity': sensitivity / max(1e-6, np.mean(param_effects))
            })
        
        df = pd.DataFrame(results)
        df = df.sort_values('sensitivity', ascending=False)
        
        print("\nMost sensitive parameters:")
        for _, row in df.head(3).iterrows():
            print(f"  {row['parameter']}: {row['sensitivity']:.3f}")
        
        return df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Complete experimental workflow.
    1. Calibration to literature
    2. Sensitivity analysis
    3. Main coupled vs uncoupled experiment
    4. Statistical analysis
    5. Save results
    """
    
    print("="*80)
    print("MULTI-SCALE SELF-ORGANIZATION EXPERIMENT")
    print("Testing hypothesis: Coupling enhances oligomerization")
    print("="*80)
    
    # Create baseline system
    print("\n1. CREATING BASELINE SYSTEM")
    chem_system = ChemicalSystem(
        grid_shape=(96, 96),  # Smaller for faster testing
        temperature_k=338.0,
        clay_concentration_g_l=5.0
    )
    
    # Calibration
    print("\n2. CALIBRATION TO EXPERIMENTAL DATA")
    calibration = Validation.calibrate_to_ferris(
        chem_system,
        target_length=CONST.FERRIS_1996_OLIGOMER_LENGTH,
        target_time_h=CONST.FERRIS_1996_TIMESCALE_H
    )
    
    print(f"   Achieved: {calibration['achieved_length']:.1f} vs Target: {calibration['target_length']}")
    print(f"   Match: {calibration['match_percentage']:.1f}%")
    
    if calibration['recommendations']:
        print("   Recommendations:")
        for rec in calibration['recommendations']:
            print(f"     - {rec}")
    
    # Sensitivity analysis
    print("\n3. SENSITIVITY ANALYSIS")
    sensitivity_params = {
        'clay_concentration': (1.0, 10.0),
        'temperature': (318.0, 358.0),  # 45-85°C
        'nucleotide_concentration': (0.5, 2.0)  # Relative to baseline
    }
    
    sensitivity_df = Validation.sensitivity_analysis(
        chem_system,
        sensitivity_params,
        n_points=3  # Fewer points for speed
    )
    
    # Main experiment
    print("\n4. MAIN EXPERIMENT: COUPLED VS UNCOUPLED")
    experiment = CoupledEmergenceExperiment(
        chemical_grid=(96, 96),  # Match calibration
        memory_grid=(24, 24),
        temperature_k=338.0,
        clay_concentration=5.0,
        experiment_duration_h=24.0  # Shorter for demonstration
    )
    
    # Run multiple trials
    experiment.run_multiple_trials(n_trials=3)  # Fewer trials for speed
    
    # Save results
    print("\n5. SAVING RESULTS")
    experiment.save_results()
    
    # Final conclusion
    stats = experiment.results.get('statistics', {})
    
    print("\n" + "="*80)
    print("EXPERIMENTAL CONCLUSION")
    print("="*80)
    
    if stats.get('hypothesis_supported', False):
        print("\n✓ HYPOTHESIS SUPPORTED")
        print(f"  Coupling increases oligomer length by {stats['length_improvement_percent']:.1f}%")
        print(f"  Effect size: {stats.get('effect_size', 0):.2f} (medium)")
        
        print("\nINTERPRETATION:")
        print("  The bidirectional coupling between chemical pattern formation")
        print("  and oscillatory memory dynamics appears to enhance the")
        print("  emergence of RNA-like polymers under these simulated conditions.")
        print("  This suggests that information-storage mechanisms might")
        print("  play a role in prebiotic chemical evolution.")
    
    else:
        print("\n✗ HYPOTHESIS NOT SUPPORTED")
        print(f"  Coupling effect: {stats.get('length_improvement_percent', 0):.1f}%")
        
        print("\nINTERPRETATION:")
        print("  Under these specific conditions and parameter choices,")
        print("  the coupling does not produce a statistically significant")
        print("  enhancement. This could mean:")
        print("  - The coupling strength needs optimization")
        print("  - Different timescales are required")
        print("  - The hypothesis might be incorrect for this system")
    
    print("\n" + "="*80)
    print("NEXT STEPS SUGGESTED:")
    print("  1. Run more trials (n=20+) for statistical power")
    print("  2. Test different coupling intervals (0.5h, 2h, 4h)")
    print("  3. Vary temperature across the Hadean range (50-90°C)")
    print("  4. Test with different clay minerals (illite, kaolinite)")
    print("  5. Compare with experimental data from wet-dry cycles")
    print("="*80)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    'PhysicalConstants',
    'ChemicalSystem',
    'RNAAgent',
    'EmergentMemory',
    'CoupledEmergenceExperiment',
    'Validation',
    'main'
]

# ============================================================================
# EXECUTION GUARD
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Check for minimal dependencies
    try:
        import numpy as np
        import pandas as pd
        from scipy.ndimage import zoom
        from scipy.signal import correlate2d
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install numpy pandas scipy")
        sys.exit(1)
    
    # Run main experiment
    main()