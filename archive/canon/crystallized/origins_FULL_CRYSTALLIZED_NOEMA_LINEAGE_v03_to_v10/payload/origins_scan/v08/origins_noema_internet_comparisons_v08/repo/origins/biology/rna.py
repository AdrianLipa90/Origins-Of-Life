"""
RNA molecule models.

Two representations are provided:

RNASequence – explicit AUGC sequence object (per-molecule, v2.0 style).
RNAPopulation – vectorised array-of-structs for large populations (v3+/holonomy style).
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np

from ..constants import RNA_FIDELITY


# ============================================================================
# RNASequence — explicit genetics
# ============================================================================

class RNASequence:
    """
    Single RNA molecule with explicit nucleotide sequence (AUGC).

    Fitness is a product of:
    - GC-content score (optimal ~50%)
    - Length score (optimal ~70 nt)
    - Sequence complexity (low-complexity runs penalised)
    """

    _id_counter = 0

    def __init__(
        self,
        sequence: str,
        position: Tuple[int, int],
        generation: int = 0,
        parent_id: Optional[int] = None,
    ):
        RNASequence._id_counter += 1
        self.id             = RNASequence._id_counter
        self.sequence       = sequence
        self.position       = position
        self.length         = len(sequence)
        self.generation     = generation
        self.parent_id      = parent_id
        self.fitness        = self._calculate_fitness()
        self.age_h          = 0.0
        self.replication_count = 0

    # ------------------------------------------------------------------
    def _calculate_fitness(self) -> float:
        seq = self.sequence
        n = len(seq)

        # GC content: optimum at 50%
        gc = (seq.count('G') + seq.count('C')) / n
        gc_fitness = np.exp(-((gc - 0.5) ** 2) / 0.05)

        # Length: optimum at 70 nt
        len_fitness = np.exp(-((n - 70) ** 2) / 500.0)

        # Complexity: penalise long homopolymer runs
        max_run = max(
            len(max((seq + 'X').split(b), key=len))
            for b in ('A', 'U', 'G', 'C')
        )
        complexity_fitness = np.exp(-(max_run ** 2) / 50.0)

        return float(np.clip(gc_fitness * len_fitness * complexity_fitness, 0.0, 1.0))

    # ------------------------------------------------------------------
    def replicate(
        self,
        fidelity: float = RNA_FIDELITY,
        uv_damage: float = 0.0,
    ) -> "RNASequence":
        """
        Produce a daughter RNA with stochastic mutations.

        Parameters
        ----------
        fidelity   : per-base copying fidelity (default from constants)
        uv_damage  : additional per-base UV damage rate
        """
        bases = ['A', 'U', 'G', 'C']
        new_seq = []
        for base in self.sequence:
            if np.random.random() < fidelity:
                new_seq.append(base)
            else:
                alts = [b for b in bases if b != base]
                new_seq.append(np.random.choice(alts))
            if uv_damage > 0.0 and np.random.random() < uv_damage:
                new_seq[-1] = np.random.choice(bases)

        daughter = RNASequence(
            sequence=''.join(new_seq),
            position=self.position,
            generation=self.generation + 1,
            parent_id=self.id,
        )
        self.replication_count += 1
        return daughter

    def __repr__(self) -> str:
        return (
            f"RNASequence(id={self.id}, len={self.length}, "
            f"gen={self.generation}, fitness={self.fitness:.3f})"
        )


# ============================================================================
# RNAPopulation — vectorised arrays
# ============================================================================

class RNAPopulation:
    """
    Vectorised RNA population for efficient large-scale simulation.

    Internally stores parallel NumPy arrays (struct-of-arrays layout)
    instead of a list of objects, enabling vectorised replication,
    fragmentation, and selection.
    """

    def __init__(self):
        self._data: dict[str, np.ndarray] = {
            'pos_x':   np.empty(0, dtype=np.int32),
            'pos_y':   np.empty(0, dtype=np.int32),
            'length':  np.empty(0, dtype=np.int32),
            'fitness': np.empty(0, dtype=np.float64),
            'age':     np.empty(0, dtype=np.float64),
        }

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._data['pos_x'])

    @property
    def pos_x(self) -> np.ndarray:
        return self._data['pos_x']

    @property
    def pos_y(self) -> np.ndarray:
        return self._data['pos_y']

    @property
    def fitness(self) -> np.ndarray:
        return self._data['fitness']

    @property
    def length(self) -> np.ndarray:
        return self._data['length']

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @classmethod
    def seed(
        cls,
        n: int,
        Nx: int,
        Ny: int,
        rng: np.random.Generator,
        len_range: Tuple[int, int] = (20, 50),
        fitness_range: Tuple[float, float] = (0.3, 0.6),
    ) -> "RNAPopulation":
        """Create a random seed population."""
        pop = cls()
        pop._data = {
            'pos_x':   rng.integers(0, Nx, n).astype(np.int32),
            'pos_y':   rng.integers(0, Ny, n).astype(np.int32),
            'length':  rng.integers(*len_range, n).astype(np.int32),
            'fitness': rng.uniform(*fitness_range, n),
            'age':     np.zeros(n, dtype=np.float64),
        }
        return pop

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def replicate_and_select(
        self,
        R_field: np.ndarray,
        topo_field: np.ndarray,
        topo_curvature: np.ndarray,
        dt: float,
        rng: np.random.Generator,
        base_rep_rate: float = 0.02,
    ) -> np.ndarray:
        """
        Vectorised replication and selection step.

        Offspring are appended to the population in-place.
        Returns an array of (x, y) positions where R field should be seeded.
        """
        if self.size == 0:
            return np.empty((0, 2), dtype=np.int32)

        px, py = self._data['pos_x'], self._data['pos_y']
        fit    = self._data['fitness']

        local_R     = R_field[px, py]
        local_topo  = topo_field[px, py]
        local_curv  = topo_curvature[px, py]
        topo_mod    = np.clip(1.0 + 0.5 * local_topo + 0.3 * local_curv, 0.1, 3.0)

        rep_probs = np.clip(
            base_rep_rate * (1.0 + fit) * (local_R + 1e-6) * topo_mod * dt,
            0.0, 0.8,
        )
        mask = rng.random(self.size) < rep_probs
        n_new = int(mask.sum())

        seed_positions = np.empty((0, 2), dtype=np.int32)
        if n_new > 0:
            idx = np.nonzero(mask)[0]
            Nx, Ny = R_field.shape
            offs_x = (px[idx] + rng.integers(-1, 2, n_new)) % Nx
            offs_y = (py[idx] + rng.integers(-1, 2, n_new)) % Ny
            offs_len = np.clip(
                self._data['length'][idx] + rng.integers(-2, 3, n_new),
                5, 200,
            ).astype(np.int32)
            offs_fit = np.clip(
                fit[idx] + rng.normal(0.0, 0.02, n_new),
                0.0, 1.0,
            )
            self._append(offs_x, offs_y, offs_len, offs_fit)
            seed_positions = np.column_stack([offs_x, offs_y])

        return seed_positions

    def fragment(
        self,
        rng: np.random.Generator,
        length_threshold: int = 80,
        frag_prob_per_dt: float = 0.005,
        dt: float = 0.05,
        Nx: int = 96,
        Ny: int = 96,
    ) -> np.ndarray:
        """
        Stochastic RNA fragmentation for long molecules.

        Returns seed positions for R field updates.
        """
        if self.size == 0:
            return np.empty((0, 2), dtype=np.int32)

        frag_mask = self._data['length'] > length_threshold
        if not frag_mask.any():
            return np.empty((0, 2), dtype=np.int32)

        draws = rng.random(self.size)
        do_frag = frag_mask & (draws < frag_prob_per_dt * dt)
        seed_pos = []
        for idx in np.nonzero(do_frag)[0]:
            L0 = int(self._data['length'][idx])
            if L0 <= 10:
                continue
            cut = int(rng.integers(5, max(6, L0 - 4)))
            self._data['length'][idx] = cut
            new_x = (int(self._data['pos_x'][idx]) + int(rng.integers(-1, 2))) % Nx
            new_y = (int(self._data['pos_y'][idx]) + int(rng.integers(-1, 2))) % Ny
            new_fit = float(np.clip(self._data['fitness'][idx] + rng.normal(0.0, 0.01), 0.0, 1.0))
            self._append(
                np.array([new_x], dtype=np.int32),
                np.array([new_y], dtype=np.int32),
                np.array([max(5, L0 - cut)], dtype=np.int32),
                np.array([new_fit]),
            )
            seed_pos.append([new_x, new_y])
        return np.array(seed_pos, dtype=np.int32) if seed_pos else np.empty((0, 2), dtype=np.int32)

    def degrade(
        self,
        k_deg: float,
        temp_factor: float,
        topo_field: np.ndarray,
        topo_curvature: np.ndarray,
        dt: float,
        rng: np.random.Generator,
    ) -> None:
        """Remove molecules that degrade this time step."""
        if self.size == 0:
            return
        px, py = self._data['pos_x'], self._data['pos_y']
        local_topo = topo_field[px, py]
        local_curv = topo_curvature[px, py]
        mod = np.clip(1.0 + 0.5 * (-local_topo) + 0.3 * local_curv, 0.05, 4.0)
        probs = k_deg * temp_factor * dt * mod
        keep  = rng.random(self.size) >= probs

        self._data['age'] = (self._data['age'] + dt)[keep]
        for key in ('pos_x', 'pos_y', 'length', 'fitness'):
            self._data[key] = self._data[key][keep]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append(
        self,
        pos_x: np.ndarray,
        pos_y: np.ndarray,
        lengths: np.ndarray,
        fitness: np.ndarray,
    ) -> None:
        self._data['pos_x']   = np.concatenate([self._data['pos_x'],   pos_x])
        self._data['pos_y']   = np.concatenate([self._data['pos_y'],   pos_y])
        self._data['length']  = np.concatenate([self._data['length'],  lengths])
        self._data['fitness'] = np.concatenate([self._data['fitness'], fitness])
        self._data['age']     = np.concatenate([self._data['age'],     np.zeros(len(pos_x))])

    def mean_fitness(self) -> float:
        if self.size == 0:
            return 0.0
        return float(np.mean(self._data['fitness']))

    def __len__(self) -> int:
        return self.size
