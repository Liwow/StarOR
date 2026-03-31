"""
Cluster management for reward computation.

Provides:
- SemanticCluster: tracks objective value clusters with relative tolerance matching
- StructuralCluster: tracks model structure feature tuples with time-decayed counting
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SemanticSample:
    """A single sample in the semantic cluster."""
    obj_value: float
    cluster_leader: float  # The cluster leader this sample belongs to
    iteration: int


@dataclass(slots=True)
class StructuralSample:
    """A single sample in the structural cluster."""
    feature_tuple: tuple[int, int, int, int]  # (ModelSense, NumVars, NumBinVars, NumIntVars)
    iteration: int


class SemanticCluster:
    """
    Manages semantic clusters based on objective values.
    
    Uses relative tolerance (default 0.5%) to group similar objective values.
    Tracks cluster leaders and their counts for computing r1 reward.
    """
    
    def __init__(self, rel_tol: float = 0.005) -> None:
        self.rel_tol = rel_tol
        # {cluster_leader: count} - valid samples only
        self._clusters: dict[float, int] = {}
        # Total count of valid samples
        self._total_valid: int = 0
        # History of all samples for debugging
        self._samples: list[SemanticSample] = []
    
    def add_sample(self, obj_value: float, iteration: int) -> float:
        """
        Add a new sample to the cluster.
        
        Returns the cluster leader this sample belongs to.
        """
        # Find matching cluster leader
        leader = self._find_or_create_cluster(obj_value)
        
        # Update counts
        self._clusters[leader] = self._clusters.get(leader, 0) + 1
        self._total_valid += 1
        
        # Record sample
        self._samples.append(SemanticSample(
            obj_value=obj_value,
            cluster_leader=leader,
            iteration=iteration,
        ))
        
        return leader
    
    def get_cluster_count(self, leader: float) -> int:
        """Get the count for a specific cluster leader."""
        return self._clusters.get(leader, 0)
    
    def get_total_valid(self) -> int:
        """Get total number of valid samples."""
        return self._total_valid
    
    def get_num_clusters(self) -> int:
        """Get number of discovered clusters."""
        return len(self._clusters)
    
    def compute_r1(self, obj_value: float, alpha: float, min_k: int) -> tuple[float, dict[str, Any]]:
        """
        Compute r1 reward for a given objective value.
        
        Formula: r1 = (count_i + alpha) / (N + alpha * K)
        where K = max(num_clusters + 1, min_k)
        
        Returns (r1, debug_info)
        """
        leader = self._find_matching_leader(obj_value)
        if leader is None:
            # This shouldn't happen if add_sample was called first
            leader = obj_value
        
        count_i = self._clusters.get(leader, 0)
        n = self._total_valid
        k = max(len(self._clusters) + 1, min_k)
        
        if n == 0:
            r1 = alpha / (alpha * k)
        else:
            r1 = (count_i + alpha) / (n + alpha * k)
        
        debug_info = {
            "cluster_leader": leader,
            "cluster_count": count_i,
            "total_valid": n,
            "num_clusters": len(self._clusters),
            "k": k,
            "alpha": alpha,
            "r1": r1,
        }
        
        return r1, debug_info
    
    def _find_or_create_cluster(self, obj_value: float) -> float:
        """Find matching cluster leader or create new cluster."""
        leader = self._find_matching_leader(obj_value)
        if leader is not None:
            return leader
        # Create new cluster with this value as leader
        self._clusters[obj_value] = 0
        return obj_value
    
    def _find_matching_leader(self, obj_value: float) -> float | None:
        """Find a cluster leader within relative tolerance."""
        for leader in self._clusters:
            if self._within_rel_tol(obj_value, leader):
                return leader
        return None
    
    def _within_rel_tol(self, value: float, ref: float) -> bool:
        """Check if value is within relative tolerance of ref."""
        base = max(abs(ref), 1e-12)
        return abs(value - ref) <= self.rel_tol * base
    
    def get_state(self) -> dict[str, Any]:
        """Get cluster state for debugging/logging."""
        return {
            "clusters": dict(self._clusters),
            "total_valid": self._total_valid,
            "num_clusters": len(self._clusters),
            "rel_tol": self.rel_tol,
        }


class StructuralCluster:
    """
    Manages structural clusters based on model feature tuples.
    
    Features: (ModelSense, NumVars, NumBinVars, NumIntVars)
    Uses time-decay weighting for historical counts.
    """
    
    def __init__(self, decay: float = 0.95) -> None:
        self.decay = decay
        # All samples with their iteration number
        self._samples: list[StructuralSample] = []
        # Total sample count (includes failed executions)
        self._total_count: int = 0
    
    def add_sample(
        self,
        feature_tuple: tuple[int, int, int, int] | None,
        iteration: int,
    ) -> None:
        """
        Add a new sample to the cluster.
        
        feature_tuple can be None if extraction failed.
        """
        self._total_count += 1
        if feature_tuple is not None:
            self._samples.append(StructuralSample(
                feature_tuple=feature_tuple,
                iteration=iteration,
            ))
    
    def compute_r4(
        self,
        feature_tuple: tuple[int, int, int, int] | None,
        current_iteration: int,
        alpha: float,
        k: int,
    ) -> tuple[float, dict[str, Any]]:
        """
        Compute r4 reward for a given feature tuple.
        
        Uses time-decayed consensus counting for each feature dimension.
        
        Formula:
        - For each feature X, compute ψ(X) = Σ(decay^(current_iter - sample_iter)) for matching samples
        - S = √ψ(D) + √ψ(NV) + √ψ(NB) + √ψ(NI)
        - S_max = 4 * √(N + alpha * K)
        - r4 = S / S_max
        
        Returns (r4, debug_info)
        """
        n = self._total_count
        
        if n == 0 or feature_tuple is None:
            return 0.0, {
                "r4": 0.0,
                "reason": "no_samples" if n == 0 else "no_feature_tuple",
                "total_count": n,
            }
        
        # Compute decayed consensus for each feature dimension
        model_sense, num_vars, num_bin_vars, num_int_vars = feature_tuple
        
        psi_d = 0.0  # ModelSense (Direction)
        psi_nv = 0.0  # NumVars
        psi_nb = 0.0  # NumBinVars
        psi_ni = 0.0  # NumIntVars
        
        for sample in self._samples:
            delta_iter = current_iteration - sample.iteration
            weight = self.decay ** max(0, delta_iter)
            
            s_sense, s_nv, s_nb, s_ni = sample.feature_tuple
            
            if s_sense == model_sense:
                psi_d += weight
            if s_nv == num_vars:
                psi_nv += weight
            if s_nb == num_bin_vars:
                psi_nb += weight
            if s_ni == num_int_vars:
                psi_ni += weight
        
        # Add current sample contribution (weight = 1.0)
        psi_d += 1.0
        psi_nv += 1.0
        psi_nb += 1.0
        psi_ni += 1.0
        
        # Compute raw score
        s = math.sqrt(psi_d) + math.sqrt(psi_nv) + math.sqrt(psi_nb) + math.sqrt(psi_ni)
        
        # Compute max score
        s_max = 4.0 * math.sqrt(n + alpha * k)
        
        # Normalize
        r4 = s / s_max if s_max > 0 else 0.0
        r4 = min(1.0, max(0.0, r4))  # Clamp to [0, 1]
        
        debug_info = {
            "r4": r4,
            "feature_tuple": feature_tuple,
            "psi_direction": psi_d,
            "psi_num_vars": psi_nv,
            "psi_num_bin_vars": psi_nb,
            "psi_num_int_vars": psi_ni,
            "raw_score": s,
            "max_score": s_max,
            "total_count": n,
            "num_structural_samples": len(self._samples),
            "decay": self.decay,
            "alpha": alpha,
            "k": k,
        }
        
        return r4, debug_info
    
    def get_total_count(self) -> int:
        """Get total sample count (all samples, including failed)."""
        return self._total_count
    
    def get_state(self) -> dict[str, Any]:
        """Get cluster state for debugging/logging."""
        # Count feature tuple frequencies
        feature_counts: dict[tuple[int, int, int, int], int] = {}
        for sample in self._samples:
            ft = sample.feature_tuple
            feature_counts[ft] = feature_counts.get(ft, 0) + 1
        
        return {
            "total_count": self._total_count,
            "num_structural_samples": len(self._samples),
            "decay": self.decay,
            "feature_counts": {str(k): v for k, v in feature_counts.items()},
        }
