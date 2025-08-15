#!/usr/bin/env python3
"""GPU Proof of Concept for Monty - Clean Implementation

Clean implementation that tests GPU kernels against CPU implementations.
Each Monty operation has exactly one GPU and one CPU implementation.

Features:
- Always runs per-step analysis
- Always benchmarks GPU vs CPU performance
- Always verifies GPU outputs against CPU outputs
- Chains operations when possible, falls back to trace data when needed
- Clear separation between GPU and CPU implementations

Usage:
    # Test all functions (default)
    python gpu_monty_poc_clean.py --output-dir /path/to/experiment

    # Test specific function
    python gpu_monty_poc_clean.py --output-dir /path/to/experiment --function displacement
"""

import argparse
import csv
import gc
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.spatial import cKDTree

# Add gpu_kernels to path
gpu_kernels_dir = os.path.join(os.path.dirname(__file__), "../gpu_kernels")
sys.path.insert(0, gpu_kernels_dir)

try:
    import monty_cuda
except ImportError:
    print("Warning: monty_cuda not available, GPU functions will fail")
    monty_cuda = None


def load_profiling_data(output_dir: str) -> List[Dict]:
    """Load saved profiling traces from experiment output."""
    output_path = Path(output_dir)
    total_traces = []
    # Hard coded to 5 LM max for now TODO handle arbitrary LM count max
    for i in range(5):
        trace_file = output_path / f"hypothesis_computation_trace_LMlearning_module_{i}.pkl"
        print(trace_file)
        if trace_file.exists():
            with open(trace_file, "rb") as f:
                traces = pickle.load(f)
            print(f"Loaded {len(traces)} computation traces")

            complete_traces = [t for t in traces if "evidence_intermediates" in t]
            print(f"Found {len(complete_traces)} traces with complete evidence data")
            total_traces += complete_traces
    if len(total_traces) == 0:
        print("No profiling data found!")
        return []
    return total_traces




class UnifiedStepData:
    """Unified data structure for processing a step with both per-trace and batched approaches."""

    def __init__(self, step_traces: List[Dict], device: torch.device):
        self.device = device
        self.step_traces = step_traces
        self.num_traces = len(step_traces)

        self.valid_traces = [t for t in step_traces if self._is_valid_trace(t)]
        self.num_valid_traces = len(self.valid_traces)

        # Per-trace data storage
        self.per_trace_data = []

        # Stacked data for batched GPU processing
        self.stacked_data = None


        self.cpu_times = {}
        self.gpu_per_trace_times = {}
        self.gpu_batched_times = {}

        self.verification_results = {}

        self._prepare_data()

    def _is_valid_trace(self, trace: Dict) -> bool:
        """Check if trace has all required data."""
        return ("inputs" in trace and
                "initial_hypotheses" in trace["inputs"] and
                "channel_displacement" in trace["inputs"])

    def _prepare_data(self):
        """Prepare both per-trace and stacked data structures."""
        # Prepare per-trace data
        for trace in self.valid_traces:
            inputs = trace["inputs"]
            per_trace_item = {
                "poses": inputs["initial_hypotheses"]["poses"],
                "locations": inputs["initial_hypotheses"]["locations"],
                "evidence": inputs["initial_hypotheses"]["evidence"],
                "displacement": inputs["channel_displacement"],
                "trace": trace  # Keep reference to full trace
            }

            if ("evidence_intermediates" in trace):

                # Add pose transformation data
                if ("pose_transformation" in trace["evidence_intermediates"]):
                    pose_data = trace["evidence_intermediates"]["pose_transformation"]
                    per_trace_item["channel_possible_poses"] = pose_data["inputs"]["channel_possible_poses"]
                    per_trace_item["channel_features"] = pose_data["inputs"]["channel_features"]
                    per_trace_item["pose_vectors"] = pose_data["inputs"]["channel_features"]["pose_vectors"]

                # Add KNN search data
                if ("nearest_neighbor_search" in trace["evidence_intermediates"]):
                    nn_data = trace["evidence_intermediates"]["nearest_neighbor_search"]
                    per_trace_item["graph_locations"] = nn_data["inputs"]["graph_locations"]
                    per_trace_item["search_locations"] = nn_data["inputs"]["search_locations"]
                    per_trace_item["nearest_node_ids"] = nn_data["outputs"]["nearest_node_ids"]

                # Add distance calculation data
                if ("distance_calculation" in trace["evidence_intermediates"]):
                    dist_data = trace["evidence_intermediates"]["distance_calculation"]
                    per_trace_item["search_locations"] = dist_data["inputs"]["search_locations"]
                    per_trace_item["nearest_node_locs"] = dist_data["inputs"]["nearest_node_locs"]
                    per_trace_item["pose_normals"] = dist_data["inputs"]["pose_normals"]
                    per_trace_item["max_abs_curvature"] = dist_data["inputs"]["max_abs_curvature"]
                    per_trace_item["custom_distances"] = dist_data["outputs"]["custom_nearest_node_dists"]

                # Add pose evidence data
                if ("pose_evidence_matrix" in trace["evidence_intermediates"]):
                    pe_data = trace["evidence_intermediates"]["pose_evidence_matrix"]
                    per_trace_item["query_pose_vectors"] = pe_data["inputs"]["query_features"]["pose_vectors"]
                    per_trace_item["node_pose_vectors"] = pe_data["inputs"]["node_features"]["pose_vectors"]
                    per_trace_item["pn_angles"] = pe_data["angle_calculation_outputs"]["pn_angles"]
                    per_trace_item["pn_evidence"] = pe_data["intermediates"]["pn_evidence"]
                    per_trace_item["cd1_evidence"] = pe_data["intermediates"]["cd1_evidence"]
                    per_trace_item["pn_weight"] = pe_data["intermediates"]["pn_weight"]
                    per_trace_item["cd1_weight"] = pe_data["intermediates"]["cd1_weight"]
                    per_trace_item["query_poses_fully_defined"] = pe_data["inputs"]["query_features"]["pose_fully_defined"]
                    per_trace_item["node_poses_fully_defined"] = pe_data["inputs"]["node_features"]["pose_fully_defined"]
                    per_trace_item["use_cd"] = per_trace_item["node_poses_fully_defined"][:, :, 0] * per_trace_item["query_poses_fully_defined"]
                    per_trace_item["pose_evidence_weighted"] = pe_data["outputs"]["pose_evidence_weighted"]

                # Add radius evidence max data
                if ("radius_evidence_max" in trace["evidence_intermediates"]):
                    final_data = trace["evidence_intermediates"]["radius_evidence_max"]
                    per_trace_item["radius_evidence"] = final_data["inputs"]["radius_evidence"]
                    per_trace_item["location_evidence"] = final_data["outputs"]["location_evidence"]

                # Add evidence aggregation data
                if ("evidence_aggregation" in trace["evidence_intermediates"]):
                    evidence_data = trace["evidence_intermediates"]["evidence_aggregation"]
                    per_trace_item["old_evidence"] = evidence_data["inputs"]["old_evidence"]
                    per_trace_item["new_evidence"] = evidence_data["inputs"]["new_evidence"]
                    per_trace_item["current_evidence"] = evidence_data["inputs"]["evidence_to_add"]
                    per_trace_item["hyp_ids_to_test"] = evidence_data["inputs"]["hyp_ids_to_test"]
                    per_trace_item["evidence_update_threshold"] = evidence_data["inputs"]["evidence_update_threshold"]
                    per_trace_item["min_update"] = evidence_data["inputs"]["min_update"]
                    per_trace_item["past_weight"] = evidence_data["inputs"]["past_weight"]
                    per_trace_item["present_weight"] = evidence_data["inputs"]["present_weight"]

            self.per_trace_data.append(per_trace_item)

        # Prepare stacked data if we have valid traces
        if self.valid_traces:
            self._prepare_stacked_data()

    def _prepare_stacked_data(self):
        """Prepare stacked data for batched GPU processing."""
        # Collect all data and compute offsets

        # Pose transformation stacked data
        all_poses = []
        all_locations = []
        all_evidence = []
        all_displacements = []
        all_channel_poses = []
        hyp_offsets = [0]

        # KNN stacked data
        all_search_locs = []
        all_nearest_locs = []
        all_pose_normals = []
        all_curvatures = []
        distance_offsets = [0]
        all_channel_features = []
        all_pose_vectors = []
        pose_offsets = [0]

        # Evidence update stacked data
        stacked_evidence_update_threshold = []
        stacked_hyp_test_ids = []
        stacked_old_evidence = []
        stacked_new_evidence = []
        stacked_current_evidence = []
        stacked_min_update = []
        evidence_update_offsets = [0]

        # Pose evidence stacked data
        all_query_pose_vectors = []
        all_node_pose_vectors = []
        all_use_cd_masks = []
        all_radius_evidence = []
        pose_evidence_offsets = [0]
        final_agg_offsets = [0]

        # Radius evidence max


        for item in self.per_trace_data:
            all_poses.append(item["poses"])
            all_locations.append(item["locations"])
            all_evidence.append(item["evidence"])
            all_displacements.append(item["displacement"])
            hyp_offsets.append(hyp_offsets[-1] + len(item["poses"]))

            all_channel_poses.append(item["channel_possible_poses"])
            all_channel_features.append(item["channel_features"])
            # all_pose_vectors.append(item["pose_vectors"].unsqueeze(0).repeat(item["channel_possible_poses"].shape[0], 1, 1))
            # all_pose_vectors.append(item["pose_vectors"].unsqueeze(0).repeat(item["channel_possible_poses"].shape[0], 1, 1))
            all_pose_vectors.append(np.broadcast_to(item["pose_vectors"][None, :, :], (item["channel_possible_poses"].shape[0], 3, 3)))
            pose_offsets.append(pose_offsets[-1] + len(all_channel_poses[-1]))
            # print(f"channel poses shape {all_channel_poses[-1].shape}")
            # print(f"pose vectors shape {all_pose_vectors[-1].shape}")
            if "search_locations" in item:
                all_search_locs.append(item["search_locations"])
                all_nearest_locs.append(item["nearest_node_locs"])
                all_pose_normals.append(item["pose_normals"])
                # print(type(item["pose_normals"]))
                # print(type(item["max_abs_curvature"]))
                # print(item["max_abs_curvature"])
                all_curvatures.append(item["max_abs_curvature"])
                distance_offsets.append(distance_offsets[-1] + len(item["search_locations"]))
            if "old_evidence" in item:
                stacked_old_evidence.append(item["old_evidence"])
                stacked_new_evidence.append(item["new_evidence"])
                stacked_current_evidence.append(item["current_evidence"])
                evidence_length = item["current_evidence"].shape[0]
                stacked_evidence_update_threshold.append(np.repeat(item["evidence_update_threshold"], evidence_length))
                stacked_min_update.append(np.repeat(item["min_update"], evidence_length))
                stacked_hyp_test_ids.append(item["hyp_ids_to_test"])
                evidence_update_offsets.append(evidence_update_offsets[-1] + evidence_length)

            # Add pose evidence data
            if "query_pose_vectors" in item and "node_pose_vectors" in item:
                all_query_pose_vectors.append(item["query_pose_vectors"])
                all_node_pose_vectors.append(item["node_pose_vectors"])
                all_use_cd_masks.append(item["use_cd"])
                pose_evidence_offsets.append(pose_evidence_offsets[-1] + len(item["query_pose_vectors"]))

            # Add radius evidence max data
            if "radius_evidence" in item:
                all_radius_evidence.append(item["radius_evidence"])
                final_agg_offsets.append(final_agg_offsets[-1] + item["radius_evidence"].shape[0])

        # Stack into GPU tensors
        self.stacked_data = {
            "poses": torch.from_numpy(np.concatenate(all_poses)).float().to(self.device),
            "locations": torch.from_numpy(np.concatenate(all_locations)).float().to(self.device),
            "evidence": torch.from_numpy(np.concatenate(all_evidence)).float().to(self.device),
            "displacements": torch.from_numpy(np.stack(all_displacements)).float().to(self.device),
            "hyp_offsets": torch.tensor(hyp_offsets, dtype=torch.int32, device=self.device),
            "distance_offsets": torch.tensor(distance_offsets, dtype=torch.int32, device=self.device),
            "search_locations": torch.from_numpy(np.concatenate(all_search_locs)).float().to(self.device),
            "nearest_locations": torch.from_numpy(np.concatenate(all_nearest_locs)).float().to(self.device),
            "pose_normals": torch.from_numpy(np.concatenate(all_pose_normals)).float().to(self.device),
            "curvatures": torch.tensor(all_curvatures, device=self.device).float(),
            "hyp_counts": torch.tensor([len(poses) for poses in all_poses], dtype=torch.int32, device=self.device),
            "total_hypotheses": sum(len(poses) for poses in all_poses),
            "num_traces": len(all_poses),
            "channel_poses": torch.from_numpy(np.concatenate(all_channel_poses)).float().to(self.device),
            "pose_offsets": torch.tensor(pose_offsets, dtype=torch.int32, device=self.device),
            "pose_vectors": torch.from_numpy(np.concatenate(all_pose_vectors)).float().to(self.device),
            "old_evidence": torch.from_numpy(np.concatenate(stacked_old_evidence)).float().to(self.device),
            "new_evidence": torch.from_numpy(np.concatenate(stacked_new_evidence)).float().to(self.device),
            "current_evidence": torch.from_numpy(np.concatenate(stacked_current_evidence)).float().to(self.device),
            "evidence_update_thresholds": torch.from_numpy(np.concatenate(stacked_evidence_update_threshold)).float().to(self.device),
            "hyp_test_ids": torch.from_numpy(np.concatenate(stacked_hyp_test_ids)).float().to(self.device),
            "min_updates": torch.from_numpy(np.concatenate(stacked_min_update)).float().to(self.device),
            "update_offsets": torch.tensor(evidence_update_offsets, dtype=torch.int32, device=self.device),
            "query_pose_vectors": torch.from_numpy(np.concatenate(all_query_pose_vectors)).float().to(self.device),
            "node_pose_vectors": torch.from_numpy(np.concatenate(all_node_pose_vectors)).float().to(self.device),
            "use_cd_masks": torch.from_numpy(np.concatenate(all_use_cd_masks)).float().to(self.device),
            "pose_evidence_offsets": torch.tensor(pose_evidence_offsets, dtype=torch.int32, device=self.device),
            "radius_evidence": torch.from_numpy(np.concatenate(all_radius_evidence)).float().to(self.device),
            "final_agg_offsets": torch.tensor(final_agg_offsets, dtype=torch.int32, device=self.device),
        }

        print(f"Prepared unified data: {self.num_valid_traces} valid traces, {self.stacked_data['total_hypotheses']} total hypotheses")

    def get_per_trace_item(self, idx: int) -> Dict:
        """Get per-trace data for a specific trace."""
        return self.per_trace_data[idx]

    def get_stacked_tensors(self) -> Dict:
        """Get stacked tensors for batched processing."""
        return self.stacked_data

    def cleanup(self):
        """Cleanup GPU memory."""
        try:
            # Cleanup stacked tensors
            if self.stacked_data:
                for key, tensor in self.stacked_data.items():
                    if torch.is_tensor(tensor):
                        del tensor
                self.stacked_data = None

            # Cleanup intermediate results
            for attr in ["search_locations", "nearest_indices", "nearest_locations",
                        "custom_distances", "pose_evidence", "final_evidence"]:
                if hasattr(self, attr) and getattr(self, attr) is not None:
                    val = getattr(self, attr)
                    if isinstance(val, list):
                        for item in val:
                            if torch.is_tensor(item):
                                del item
                    elif torch.is_tensor(val):
                        del val
                    setattr(self, attr, None)

            # Force GPU memory cleanup
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")

    def __del__(self):
        """Cleanup when object is garbage collected."""
        self.cleanup()


class MontyOperations:
    """Implementation of Monty operations with GPU and CPU implementations."""

    def __init__(self, device: torch.device):
        self.device = device

    # =================================================================
    # DISPLACEMENT OPERATIONS
    # =================================================================

    def displacement_gpu(self, poses: torch.Tensor, locations: torch.Tensor,
                        displacement: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """GPU implementation of displacement calculation using CUDA kernels."""
        if monty_cuda is None:
            raise RuntimeError("CUDA kernels not available")

        start_time = time.perf_counter()

        expanded_displacement = displacement.repeat(poses.shape[0], 1)

        search_locations = monty_cuda.displacement_stacked(
            poses, expanded_displacement, locations
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        return search_locations, gpu_time

    def displacement_cpu(self, poses: np.ndarray, locations: np.ndarray,
                        displacement: np.ndarray) -> Tuple[np.ndarray, float]:
        """CPU implementation matching Monty codebase exactly."""
        start_time = time.perf_counter()

        rotated_displacements = np.dot(poses, displacement)  # (N, 3)
        search_locations = locations + rotated_displacements  # (N, 3)

        cpu_time = time.perf_counter() - start_time

        return search_locations, cpu_time

    def displacement_gpu_batched(self, unified_data: UnifiedStepData) -> Tuple[torch.Tensor, float]:
        """Batched GPU displacement - single kernel call for all traces."""
        if monty_cuda is None:
            raise RuntimeError("CUDA kernels not available")

        stacked = unified_data.get_stacked_tensors()
        if not stacked:
            return None, 0.0

        expanded_displacements = []
        for i, count in enumerate(stacked["hyp_counts"]):
            expanded_displacements.append(stacked["displacements"][i].repeat(count, 1))
        stacked_displacements = torch.cat(expanded_displacements, dim=0)

        start_time = time.perf_counter()
        search_locations = monty_cuda.displacement_stacked(
            stacked["poses"], stacked_displacements, stacked["locations"]
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        for tensor in expanded_displacements:
            del tensor
        del stacked_displacements

        return search_locations, stacked["hyp_offsets"], gpu_time

    # =================================================================
    # NEAREST NEIGHBOR SEARCH OPERATIONS
    # =================================================================

    def knn_search_gpu(self, search_locations: torch.Tensor,
                      graph_locations: torch.Tensor,
                      k: int = 3) -> Tuple[torch.Tensor, float]:
        """GPU implementation of KNN search using CUDA kernels."""
        if monty_cuda is None:
            raise RuntimeError("CUDA kernels not available")

        start_time = time.perf_counter()

        graph_offsets = torch.tensor([0], dtype=torch.int32, device=self.device)
        query_offsets = torch.tensor([0], dtype=torch.int32, device=self.device)

        nearest_indices = monty_cuda.knn_search_stacked(
            graph_locations, search_locations, graph_offsets, query_offsets, k
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        return nearest_indices, gpu_time

    def knn_search_cpu(self, search_locations: np.ndarray,
                      graph_locations: np.ndarray,
                      k: int = 3) -> Tuple[np.ndarray, float]:
        """CPU implementation using cKDTree (matches Monty)."""
        tree = cKDTree(graph_locations)
        start_time = time.perf_counter()
        _, nearest_indices = tree.query(search_locations, k=k)

        if k == 1:
            nearest_indices = nearest_indices.reshape(-1, 1)

        cpu_time = time.perf_counter() - start_time

        return nearest_indices, cpu_time

    def knn_search_gpu_batched(self, unified_data: UnifiedStepData,
                              search_locations: torch.Tensor,
                              graph_locations: torch.Tensor,
                              k: int = 3) -> Tuple[torch.Tensor, float]:
        """Batched GPU KNN search - single kernel call for all traces."""
        if monty_cuda is None:
            raise RuntimeError("CUDA kernels not available")

        stacked = unified_data.get_stacked_tensors()
        if not stacked:
            return None, 0.0

        start_time = time.perf_counter()

        # Create graph and query offsets for all traces
        graph_offsets = torch.zeros(stacked["num_traces"], dtype=torch.int32, device=self.device)
        query_offsets = stacked["hyp_offsets"]

        # Single stacked kernel call for all traces
        nearest_indices = monty_cuda.knn_search_stacked(
            graph_locations, search_locations, graph_offsets, query_offsets, k
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        return nearest_indices, stacked["hyp_offsets"], gpu_time

    # =================================================================
    # DISTANCE CALCULATION OPERATIONS
    # =================================================================

    def distance_calculation_gpu(self, search_locations: torch.Tensor,
                                nearest_locations: torch.Tensor,
                                pose_normals: torch.Tensor,
                                curvature: float) -> Tuple[torch.Tensor, float]:
        """GPU implementation of custom distance calculation using CUDA kernels."""
        if monty_cuda is None:
            raise RuntimeError("CUDA kernels not available")


        trace_offsets = torch.tensor([0], dtype=torch.int32, device=self.device)
        curvatures_tensor = torch.tensor([curvature], dtype=torch.float32, device=self.device)

        start_time = time.perf_counter()

        custom_distances = monty_cuda.custom_distance_stacked(
            nearest_locations, search_locations, pose_normals, curvatures_tensor, trace_offsets
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        return custom_distances, gpu_time

    def distance_calculation_cpu(self, search_locations: np.ndarray,
                                nearest_locations: np.ndarray,
                                pose_normals: np.ndarray,
                                curvature: float) -> Tuple[np.ndarray, float]:
        """CPU implementation matching Monty's get_custom_distances."""
        start_time = time.perf_counter()

        # Exact CPU implementation from original script
        query_locs_expanded = search_locations[:, np.newaxis, :]
        differences = nearest_locations - query_locs_expanded
        euclidean_dists = np.linalg.norm(differences, axis=2)

        dot_products = np.einsum("ijk,ik->ij", differences, pose_normals)
        curvature_factor = 1.0 / (abs(curvature) + 0.5)
        custom_distances = euclidean_dists + np.abs(dot_products) * curvature_factor

        cpu_time = time.perf_counter() - start_time

        return custom_distances, cpu_time

    def distance_calculation_gpu_batched(self, unified_data: UnifiedStepData,
                                    #    search_locations: torch.Tensor,
                                    #    nearest_locations: torch.Tensor,
                                    #    pose_normals: torch.Tensor,
                                    #    curvatures: List[float]
                                       ) -> Tuple[torch.Tensor, float]:
        """Batched GPU distance calculation - single kernel call for all traces."""
        if monty_cuda is None:
            raise RuntimeError("CUDA kernels not available")

        stacked = unified_data.get_stacked_tensors()
        if not stacked:
            return None, 0.0
        search_locations = stacked["search_locations"]
        nearest_locations = stacked["nearest_locations"]
        pose_normals = stacked["pose_normals"]
        curvatures = stacked["curvatures"]
        trace_offsets = stacked["distance_offsets"]

        start_time = time.perf_counter()

        custom_distances = monty_cuda.custom_distance_stacked(
            nearest_locations, search_locations, pose_normals, curvatures, trace_offsets
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        return custom_distances, trace_offsets, gpu_time

    # =================================================================
    # POSE EVIDENCE CALCULATION OPERATIONS
    # =================================================================

    def pose_evidence_gpu(self, query_poses: torch.Tensor,
                         node_poses: torch.Tensor,
                         use_cd_mask: torch.Tensor,
                         weights: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """GPU implementation of pose evidence calculation using CUDA kernels."""
        if monty_cuda is None:
            raise RuntimeError("CUDA kernels not available")

        query_normals = query_poses[:, 0]  # (N, 3)
        node_normals = node_poses[:, :, :3]  # (N, K, 3)

        query_normals = query_normals.contiguous()
        node_normals = node_normals.contiguous()

        pn_angles = monty_cuda.angle_calculation(node_normals, query_normals)
        cd1_angles = monty_cuda.angle_calculation(node_poses[:, :, 3:6].contiguous(), query_poses[:, 1].contiguous())

        pose_offsets = torch.tensor([0], dtype=torch.int32, device=self.device).contiguous()
        pn_weights = torch.tensor([weights[0]], dtype=torch.float32, device=self.device).expand(pn_angles.shape).contiguous()
        cd1_weights = torch.tensor([weights[1]], dtype=torch.float32, device=self.device).expand(pn_angles.shape).contiguous()

        start_time = time.perf_counter()

        pose_evidence = monty_cuda.pose_evidence_stacked(
            pn_angles, cd1_angles, use_cd_mask, pn_weights, cd1_weights, pose_offsets
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        return pose_evidence, gpu_time

    def pose_evidence_cpu(self, query_poses: np.ndarray,
                         node_poses: np.ndarray,
                         use_cd: np.ndarray,
                         query_poses_full_defined: bool,
                         weights: np.ndarray) -> Tuple[np.ndarray, float]:
        """CPU implementation matching Monty's pose evidence calculation."""
        query_normals = query_poses[:, 0]  # Take first pose vector
        node_normals = node_poses[:, :, :3]

        pn_weight = np.array([weights[0]])
        cd1_weight = np.array([weights[1]])

        start_time = time.perf_counter()

        dot_product = np.einsum("ijk,ik->ij", node_normals, query_normals)
        pn_angles = np.arccos(np.clip(dot_product, -1, 1))
        pn_evidence = -(np.sin(pn_angles / 2) - 0.5)
        if not query_poses_full_defined:
            cd1_weight = 0
            cd1_evidence = np.zeros(pn_evidence.shape)
        else:
            dot_product = np.einsum("ijk,ik->ij", node_poses[:, :, 3:6], query_poses[:, 1])
            cd1_angles = np.arccos(np.clip(dot_product, -1, 1))
            cd1_error = np.pi / 2 - np.abs(cd1_angles - np.pi / 2)
            cd1_evidence = -(np.sin(cd1_error) - 0.5)
            cd1_evidence = cd1_evidence * use_cd
            pn_evidence[np.logical_not(use_cd)] *= 2
        pose_evidence_weighted = pn_evidence * pn_weight + cd1_evidence * cd1_weight

        cpu_time = time.perf_counter() - start_time

        return pose_evidence_weighted, cpu_time

    def pose_evidence_gpu_batched(self, unified_data: UnifiedStepData,
                                weights: np.ndarray) -> Tuple[torch.Tensor, float]:
        """Batched GPU pose evidence - single kernel call for all traces."""
        if monty_cuda is None:
            raise RuntimeError("CUDA kernels not available")

        stacked = unified_data.get_stacked_tensors()
        if not stacked or "query_pose_vectors" not in stacked:
            return None, 0.0

        query_poses = stacked["query_pose_vectors"]
        node_poses = stacked["node_pose_vectors"]
        use_cd_masks = stacked["use_cd_masks"].contiguous()
        poses_offsets = stacked["pose_evidence_offsets"]

        query_normals = query_poses[:, 0].contiguous()  # (N, 3)
        node_normals = node_poses[:, :, :3].contiguous()  # (N, K, 3)

        pn_angles = monty_cuda.angle_calculation(node_normals, query_normals)
        cd1_angles = monty_cuda.angle_calculation(node_poses[:, :, 3:6].contiguous(), query_poses[:, 1].contiguous())
        pn_weights = torch.full_like(pn_angles, weights[0], dtype=torch.float32, device=self.device).contiguous()
        cd1_weights = torch.full_like(cd1_angles, weights[1], dtype=torch.float32, device=self.device).contiguous()

        start_time = time.perf_counter()

        pose_evidence = monty_cuda.pose_evidence_stacked(
            pn_angles,
            cd1_angles,
            use_cd_masks,
            pn_weights,
            cd1_weights,
            poses_offsets
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        return pose_evidence, gpu_time

    # =================================================================
    # Radius evidence max OPERATIONS
    # =================================================================

    def radius_evidence_max_gpu(self, evidence_matrix: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """GPU implementation of radius evidence max using CUDA kernels."""
        if monty_cuda is None:
            raise RuntimeError("CUDA kernels not available")

        start_time = time.perf_counter()

        final_evidence = monty_cuda.radius_evidence_max_stacked(evidence_matrix)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        return final_evidence, gpu_time

    def radius_evidence_max_cpu(self, evidence_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
        """CPU implementation of radius evidence max."""
        start_time = time.perf_counter()

        final_evidence = np.max(evidence_matrix, axis=1)

        cpu_time = time.perf_counter() - start_time

        return final_evidence, cpu_time

    def radius_evidence_max_gpu_batched(self, unified_data: UnifiedStepData) -> Tuple[torch.Tensor, float]:
        """Batched GPU radius evidence max - single kernel call for all traces."""
        if monty_cuda is None:
            raise RuntimeError("CUDA kernels not available")

        stacked = unified_data.get_stacked_tensors()
        if not stacked or "radius_evidence" not in stacked:
            return None, 0.0

        evidence_matrix = stacked["radius_evidence"]

        start_time = time.perf_counter()

        final_evidence = monty_cuda.radius_evidence_max_stacked(evidence_matrix)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        return final_evidence, gpu_time

    # =================================================================
    # POSE TRANSFORMATION OPERATIONS
    # =================================================================

    def pose_transformation_gpu(self, pose_vectors: torch.Tensor,
                               reference_poses: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """GPU implementation of pose transformation using CUDA kernels."""
        if monty_cuda is None:
            raise RuntimeError("CUDA kernels not available")

        start_time = time.perf_counter()

        transformed_vectors = monty_cuda.pose_transformation(
            pose_vectors, reference_poses
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        return transformed_vectors, gpu_time

    def pose_transformation_cpu(self, pose_vectors: np.ndarray,
                               reference_poses: np.ndarray) -> Tuple[np.ndarray, float]:
        """CPU implementation matching rotate_pose_dependent_features exactly."""
        start_time = time.perf_counter()

        pose_vectors_T = pose_vectors.T
        result = np.matmul(reference_poses, pose_vectors_T)
        transformed_vectors = result.transpose(0, 2, 1)

        cpu_time = time.perf_counter() - start_time

        return transformed_vectors, cpu_time

    def pose_transformation_gpu_batched(self, unified_data: UnifiedStepData) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Batched GPU pose transformation - single kernel call for all traces."""
        if monty_cuda is None:
            raise RuntimeError("CUDA kernels not available")

        stacked = unified_data.get_stacked_tensors()
        pose_vectors_gpu = stacked["pose_vectors"].contiguous()
        channel_poses_gpu = stacked["channel_poses"].contiguous()
        pose_offsets_gpu = stacked["pose_offsets"].contiguous()

        start_time = time.perf_counter()

        transformed_vectors = monty_cuda.pose_transformation_stacked(
            pose_vectors_gpu, channel_poses_gpu, pose_offsets_gpu
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        gpu_time = time.perf_counter() - start_time

        return transformed_vectors, pose_offsets_gpu, gpu_time

    # =================================================================
    # EVIDENCE AGGREGATION OPERATIONS
    # =================================================================

    def evidence_aggregation_gpu(self, old_evidence: torch.Tensor,
                                new_evidence: torch.Tensor, test_indices: torch.Tensor,
                                min_update: float, past_weight: float,
                                present_weight: float) -> Tuple[torch.Tensor, float]:
        """GPU implementation of evidence aggregation using CUDA kernels."""
        if monty_cuda is None:
            raise RuntimeError("CUDA kernels not available")

        start_time = time.perf_counter()

        aggregated_evidence = monty_cuda.evidence_aggregation(
            old_evidence, new_evidence, test_indices,
            min_update, past_weight, present_weight
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        return aggregated_evidence, gpu_time

    def evidence_aggregation_cpu(self, old_evidence: np.ndarray,
                                new_evidence: np.ndarray, test_indices: np.ndarray,
                                min_update: float, past_weight: float,
                                present_weight: float) -> Tuple[np.ndarray, float]:
        """CPU implementation matching Monty evidence aggregation exactly."""
        start_time = time.perf_counter()

        evidence_to_add = np.ones_like(old_evidence) * min_update
        evidence_to_add[test_indices] = new_evidence

        aggregated_evidence = old_evidence * past_weight + evidence_to_add * present_weight

        cpu_time = time.perf_counter() - start_time

        return aggregated_evidence, cpu_time

    def evidence_aggregation_gpu_batched(self, unified_data: UnifiedStepData,
                                        past_weight: float,
                                       present_weight: float) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Batched GPU evidence aggregation - single kernel call for all traces."""
        if monty_cuda is None:
            raise RuntimeError("CUDA kernels not available")

        stacked = unified_data.get_stacked_tensors()
        if not stacked:
            return None, None, 0.0

        start_time = time.perf_counter()

        stacked_old_evidence = stacked["old_evidence"]
        stacked_current_evidence = stacked["current_evidence"]
        evidence_update_thresholds = stacked["evidence_update_thresholds"]
        min_updates = stacked["min_updates"]
        # Call stacked kernel
        aggregated_evidence = monty_cuda.evidence_aggregation_stacked(
            stacked_old_evidence, stacked_current_evidence, evidence_update_thresholds,
            min_updates, past_weight, present_weight
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        return aggregated_evidence, stacked["update_offsets"], gpu_time

    # =================================================================
    # MEMORY CLEANUP OPERATIONS
    # =================================================================


    def _memory_cleanup(self):
        """Aggressive GPU memory cleanup to prevent accumulation."""
        if self.device.type != "cuda":
            return

        torch.cuda.empty_cache()

        # Force garbage collection
        import gc
        gc.collect()

        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats(self.device)

        allocated = torch.cuda.memory_allocated(self.device) / (1024**2)
        cached = torch.cuda.memory_reserved(self.device) / (1024**2)
        print(f"    Post-cleanup memory: {allocated:.1f}MB allocated, {cached:.1f}MB cached")

    def _get_gpu_memory_info(self) -> Dict[str, Any]:
        """Get current GPU memory information."""
        if self.device.type != "cuda":
            return {"type": "cpu", "note": "CPU execution, no GPU memory"}

        try:
            allocated = torch.cuda.memory_allocated(self.device)
            cached = torch.cuda.memory_reserved(self.device)
            max_allocated = torch.cuda.max_memory_allocated(self.device)
            max_cached = torch.cuda.max_memory_reserved(self.device)

            return {
                "type": "cuda",
                "allocated_bytes": allocated,
                "allocated_mb": allocated / (1024**2),
                "cached_bytes": cached,
                "cached_mb": cached / (1024**2),
                "max_allocated_mb": max_allocated / (1024**2),
                "max_cached_mb": max_cached / (1024**2),
                "free_cached_mb": (cached - allocated) / (1024**2),
            }
        except Exception as e:
            return {"type": "cuda", "error": str(e)}

def run_step_analysis(traces: List[Dict], operations: MontyOperations,
                     target_function: Optional[str] = None) -> Dict[str, Any]:
    """Run per-step analysis comparing CPU, GPU Per-Trace, and GPU Batched."""
    # Group traces by step for all hypotheses that can run in parallel
    traces_by_step = {}
    for trace in traces:
        step = trace.get("step", 0)
        if step not in traces_by_step:
            traces_by_step[step] = []
        traces_by_step[step].append(trace)
    print(f"Processing {len(traces_by_step)} steps with {len(traces)} total traces")

    # Function mapping
    function_map = {
        "displacement": ["displacement"],
        "pose_transformation": ["pose_transformation"],
        "evidence_aggregation": ["evidence_aggregation"],
        "knn_search": ["displacement", "knn_search"],
        "distance": ["displacement", "knn_search", "distance_calculation"],
        "pose_evidence": ["displacement", "knn_search", "distance_calculation", "pose_evidence"],
        "aggregation": ["displacement", "knn_search", "distance_calculation", "pose_evidence", "radius_evidence_max"],
        None: ["displacement", "pose_transformation", "evidence_aggregation", "knn_search", "distance_calculation", "pose_evidence", "radius_evidence_max"]
    }

    functions_to_test = function_map.get(target_function, function_map[None])

    step_results = []
    total_cpu_time = 0
    total_gpu_per_trace_time = 0
    total_gpu_batched_time = 0

    for step, step_traces in sorted(traces_by_step.items()):
        print(f"\n=== Step {step}: {len(step_traces)} traces ===")

        # Create unified data structure to track all trace data for this step
        unified_data = UnifiedStepData(step_traces, operations.device)

        if unified_data.num_valid_traces == 0:
            print("  No valid traces in this step")
            continue

        # Process the cpu, gpu per-trace, and gpu stacked implementations for each function
        for func_name in functions_to_test:
            success = process_function_all_approaches(func_name, unified_data, operations)
            if not success and target_function == func_name:
                print(f"  Failed to test target function {func_name}")
                break

        step_cpu_time = sum(unified_data.cpu_times.values())
        step_gpu_per_trace_time = sum(unified_data.gpu_per_trace_times.values())
        step_gpu_batched_time = sum(unified_data.gpu_batched_times.values())

        total_cpu_time += step_cpu_time
        total_gpu_per_trace_time += step_gpu_per_trace_time
        total_gpu_batched_time += step_gpu_batched_time

        print(f"  CPU:             {step_cpu_time * 1000:.3f}ms")
        print(f"  GPU Per-Trace:   {step_gpu_per_trace_time * 1000:.3f}ms")
        print(f"  GPU Batched:     {step_gpu_batched_time * 1000:.3f}ms")

        if step_cpu_time > 0:
            per_trace_speedup = step_cpu_time / step_gpu_per_trace_time if step_gpu_per_trace_time > 0 else 0
            batched_speedup = step_cpu_time / step_gpu_batched_time if step_gpu_batched_time > 0 else 0
            batch_vs_pertrace = step_gpu_per_trace_time / step_gpu_batched_time if step_gpu_batched_time > 0 else 0

            print(f"  Per-Trace vs CPU: {per_trace_speedup:.2f}x")
            print(f"  Batched vs CPU:   {batched_speedup:.2f}x")
            print(f"  Batched vs Per-Trace: {batch_vs_pertrace:.2f}x")

        step_results.append({
            "step": step,
            "total_hypotheses": unified_data.get_stacked_tensors()["total_hypotheses"],
            "num_traces": len(step_traces),
            "cpu_times": unified_data.cpu_times.copy(),
            "gpu_per_trace_times": unified_data.gpu_per_trace_times.copy(),
            "gpu_batched_times": unified_data.gpu_batched_times.copy(),
            "verification_results": unified_data.verification_results.copy(),
            "total_cpu_time": step_cpu_time,
            "total_gpu_per_trace_time": step_gpu_per_trace_time,
            "total_gpu_batched_time": step_gpu_batched_time
        })

        unified_data.cleanup()
        operations._memory_cleanup()
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "num_steps": len(traces_by_step),
        "total_traces": len(traces),
        "functions_tested": functions_to_test,
        "total_cpu_time": total_cpu_time,
        "total_gpu_per_trace_time": total_gpu_per_trace_time,
        "total_gpu_batched_time": total_gpu_batched_time,
        "step_results": step_results
    }


def process_function_all_approaches(func_name: str, unified_data: UnifiedStepData,
                                   operations: MontyOperations) -> bool:
    """Process a single function with CPU, GPU per-trace, and GPU batched implementations."""
    print(f"  Testing {func_name}...")

    try:
        if func_name == "displacement":
            return process_displacement_all_approaches(unified_data, operations)
        elif func_name == "pose_transformation":
            return process_pose_transformation_all_approaches(unified_data, operations)
        elif func_name == "evidence_aggregation":
            return process_evidence_aggregation_all_approaches(unified_data, operations)
        elif func_name == "knn_search":
            return process_knn_search_all_approaches(unified_data, operations)
        elif func_name == "distance_calculation":
            return process_distance_calculation_all_approaches(unified_data, operations)
        elif func_name == "pose_evidence":
            return process_pose_evidence_all_approaches(unified_data, operations)
        elif func_name == "radius_evidence_max":
            return process_radius_evidence_max_all_approaches(unified_data, operations)
        else:
            print(f"    Unknown function: {func_name}")
            return False

    except Exception as e:
        print(f"    Error in {func_name}: {str(e)}")
        return False


def process_displacement_all_approaches(unified_data: UnifiedStepData,
                                       operations: MontyOperations) -> bool:
    """Process displacement with CPU, GPU per-trace, and GPU batched approaches."""
    all_search_locations_cpu = []
    all_search_locations_gpu_per_trace = []
    total_cpu_time = 0
    total_gpu_per_trace_time = 0

    for i in range(unified_data.num_valid_traces):
        trace_data = unified_data.get_per_trace_item(i)
        poses = trace_data["poses"]
        locations = trace_data["locations"]
        displacement = trace_data["displacement"]

        # CPU implementation
        search_locations_cpu, cpu_time = operations.displacement_cpu(
            poses, locations, displacement
        )
        total_cpu_time += cpu_time
        all_search_locations_cpu.append(search_locations_cpu)

        # GPU Per-Trace version
        poses_gpu = torch.from_numpy(poses.astype(np.float32)).to(operations.device)
        locations_gpu = torch.from_numpy(locations.astype(np.float32)).to(operations.device)
        displacement_gpu = torch.from_numpy(displacement.astype(np.float32)).to(operations.device)

        search_locations_gpu, gpu_time = operations.displacement_gpu(
            poses_gpu, locations_gpu, displacement_gpu
        )
        total_gpu_per_trace_time += gpu_time
        all_search_locations_gpu_per_trace.append(search_locations_gpu)

    # Stacked GPU implementation
    total_gpu_batched_time = 0
    batched_search_locations = None
    hypothesis_offsets = None
    if unified_data.num_valid_traces > 0:
        try:
            batched_search_locations, hypothesis_offsets, total_gpu_batched_time = operations.displacement_gpu_batched(unified_data)
        except Exception as e:
            print(f"    Batched GPU failed: {e}")
            total_gpu_batched_time = 0

    if not all_search_locations_cpu:
        print("    No displacement data found")
        return False

    # Store timing results
    unified_data.cpu_times["displacement"] = total_cpu_time
    unified_data.gpu_per_trace_times["displacement"] = total_gpu_per_trace_time
    unified_data.gpu_batched_times["displacement"] = total_gpu_batched_time

    # Verify results
    max_diff = 0
    max_batched_diff = 0
    total_per_trace_passed = 0
    total_stacked_passed = 0
    for i in range(len(all_search_locations_cpu)):
        cpu_result = all_search_locations_cpu[i]
        gpu_result = all_search_locations_gpu_per_trace[i].cpu().numpy()

        # Per-trace verification
        trace_diff = np.max(np.abs(cpu_result - gpu_result))
        max_diff = max(max_diff, trace_diff)

        # Batched verification if available
        gpu_batched_result = batched_search_locations[hypothesis_offsets[i]:hypothesis_offsets[i+1]]
        gpu_batched_result_np = gpu_batched_result.cpu().numpy()
        batched_trace_diff = np.max(np.abs(cpu_result - gpu_batched_result_np))
        max_batched_diff = max(max_batched_diff, batched_trace_diff)

        if max_diff < 1e-5:
            total_per_trace_passed += 1
        if max_batched_diff < 1e-5:
            total_stacked_passed += 1

    unified_data.verification_results["displacement"] = {
        "max_diff": max_diff,
        "max_batched_diff": max_batched_diff,
        "percent_per_trace_passed": 100 * total_per_trace_passed / len(all_search_locations_cpu),
        "percent_stacked_passed": 100 * total_stacked_passed / len(all_search_locations_cpu),
        "passed": max_diff < 1e-5
    }
    # print(f"passed {}")

    # Print timing comparison
    print(f"    CPU: {total_cpu_time * 1000:.3f}ms")
    print(f"    GPU Per-Trace: {total_gpu_per_trace_time * 1000:.3f}ms")
    print(f"    GPU Batched: {total_gpu_batched_time * 1000:.3f}ms")
    if total_cpu_time > 0:
        per_trace_speedup = total_cpu_time / total_gpu_per_trace_time if total_gpu_per_trace_time > 0 else 0
        batched_speedup = total_cpu_time / total_gpu_batched_time if total_gpu_batched_time > 0 else 0
        print(f"    Per-Trace Speedup: {per_trace_speedup:.2f}x, Batched Speedup: {batched_speedup:.2f}x")
    print(f"    Per-trace max diff: {max_diff:.2e}, Batched max diff: {max_batched_diff:.2e}")

    return True


def process_knn_search_all_approaches(unified_data: UnifiedStepData,
                                     operations: MontyOperations) -> bool:
    """Process KNN search with CPU, GPU per-trace, and GPU batched approaches."""
    # CPU and GPU Per-Trace processing
    all_nearest_indices_cpu = []
    all_nearest_indices_gpu_per_trace = []
    total_cpu_time = 0
    total_gpu_per_trace_time = 0

    # Use trace data directly
    for i in range(unified_data.num_valid_traces):
        trace_data = unified_data.get_per_trace_item(i)

        graph_locations = trace_data["graph_locations"]
        search_locations = trace_data["search_locations"]

        # CPU implementation
        nearest_indices_cpu, cpu_time = operations.knn_search_cpu(
            search_locations, graph_locations, k=3
        )
        total_cpu_time += cpu_time
        all_nearest_indices_cpu.append(nearest_indices_cpu)

        # GPU Per-Trace version
        graph_locations_gpu = torch.from_numpy(graph_locations.astype(np.float32)).to(operations.device)
        search_locations_gpu = torch.from_numpy(search_locations.astype(np.float32)).to(operations.device)

        nearest_indices_gpu, gpu_time = operations.knn_search_gpu(
            search_locations_gpu, graph_locations_gpu, k=3
        )
        total_gpu_per_trace_time += gpu_time
        all_nearest_indices_gpu_per_trace.append(nearest_indices_gpu)

    # Stacked GPU implementation
    total_gpu_batched_time = 0
    batched_nearest_indices = None
    hypothesis_offsets = None

    if unified_data.num_valid_traces > 0:
        try:
            print("    KNN batched processing not implemented for trace data only")
            total_gpu_batched_time = 0
        except Exception as e:
            print(f"    Batched GPU failed: {e}")
            total_gpu_batched_time = 0

    if not all_nearest_indices_cpu:
        print("    No KNN search data processed")
        return False

    # Store timing results
    unified_data.cpu_times["knn_search"] = total_cpu_time
    unified_data.gpu_per_trace_times["knn_search"] = total_gpu_per_trace_time
    unified_data.gpu_batched_times["knn_search"] = total_gpu_batched_time

    # Verify results
    max_diff = 0
    max_batched_diff = 0
    total_per_trace_passed = 0
    total_stacked_passed = 0
    for i in range(len(all_nearest_indices_cpu)):
        cpu_result = all_nearest_indices_cpu[i]
        gpu_result = all_nearest_indices_gpu_per_trace[i].cpu().numpy()

        # Per-trace verification
        trace_diff = np.max(np.abs(cpu_result - gpu_result))
        max_diff = max(max_diff, trace_diff)

        # Batched verification
        gpu_batched_result = batched_nearest_indices[hypothesis_offsets[i]:hypothesis_offsets[i+1]]
        gpu_batched_result_np = gpu_batched_result.cpu().numpy()
        batched_trace_diff = np.max(np.abs(cpu_result - gpu_batched_result_np))
        max_batched_diff = max(max_batched_diff, batched_trace_diff)
        if max_diff < 1e-5:
            total_per_trace_passed += 1
        if max_batched_diff < 1e-5:
            total_stacked_passed += 1

    unified_data.verification_results["knn_search"] = {
        "max_diff": max_diff,
        "max_batched_diff": max_batched_diff,
        "percent_per_trace_passed": 100 * total_per_trace_passed / len(all_nearest_indices_cpu),
        "percent_stacked_passed": 100 * total_stacked_passed / len(all_nearest_indices_cpu),
        "passed": max_diff < 1e-5
    }


    # Print timing comparison
    print(f"    CPU: {total_cpu_time * 1000:.3f}ms")
    print(f"    GPU Per-Trace: {total_gpu_per_trace_time * 1000:.3f}ms")
    print(f"    GPU Batched: {total_gpu_batched_time * 1000:.3f}ms")
    if total_cpu_time > 0:
        per_trace_speedup = total_cpu_time / total_gpu_per_trace_time if total_gpu_per_trace_time > 0 else 0
        batched_speedup = total_cpu_time / total_gpu_batched_time if total_gpu_batched_time > 0 else 0
        print(f"    Per-Trace Speedup: {per_trace_speedup:.2f}x, Batched Speedup: {batched_speedup:.2f}x")
    print(f"    Per-trace max diff: {max_diff:.2e}, Batched max diff: {max_batched_diff:.2e}")

    return True


def process_distance_calculation_all_approaches(unified_data: UnifiedStepData,
                                              operations: MontyOperations) -> bool:
    """Process distance calculation with CPU, GPU per-trace, and GPU batched approaches."""
    # CPU and GPU Per-Trace processing
    all_distances_cpu = []
    all_distances_gpu_per_trace = []
    total_cpu_time = 0
    total_gpu_per_trace_time = 0

    # Use trace data directly
    for i in range(unified_data.num_valid_traces):
        trace_data = unified_data.get_per_trace_item(i)
        if ("search_locations" not in trace_data or "nearest_node_locs" not in trace_data or
            "pose_normals" not in trace_data or "max_abs_curvature" not in trace_data):
            continue

        search_locs_cpu = trace_data["search_locations"]
        nearest_locs_cpu = trace_data["nearest_node_locs"]
        pose_normals_cpu = trace_data["pose_normals"]
        curvature = trace_data["max_abs_curvature"]

        # CPU implementation
        custom_distances_cpu, cpu_time = operations.distance_calculation_cpu(
            search_locs_cpu, nearest_locs_cpu, pose_normals_cpu, curvature
        )
        total_cpu_time += cpu_time
        all_distances_cpu.append(custom_distances_cpu)

        # GPU Per-Trace version
        search_locs_gpu = torch.from_numpy(search_locs_cpu.astype(np.float32)).to(operations.device).contiguous()
        nearest_locs_gpu = torch.from_numpy(nearest_locs_cpu.astype(np.float32)).to(operations.device).contiguous()
        pose_normals_gpu = torch.from_numpy(pose_normals_cpu.astype(np.float32)).to(operations.device).contiguous()

        custom_distances_gpu, gpu_time = operations.distance_calculation_gpu(
            search_locs_gpu, nearest_locs_gpu, pose_normals_gpu, curvature
        )
        total_gpu_per_trace_time += gpu_time
        all_distances_gpu_per_trace.append(custom_distances_gpu)

    # Stacked GPU implementation
    total_gpu_batched_time = 0
    batched_custom_distances = None
    hypothesis_offsets = None
    if unified_data.num_valid_traces > 0:
        try:
            batched_custom_distances, hypothesis_offsets, total_gpu_batched_time = operations.distance_calculation_gpu_batched(unified_data)
        except Exception as e:
            print(f"    Batched GPU failed: {e}")
            total_gpu_batched_time = 0

    if not all_distances_cpu:
        print("    No distance calculation data processed")
        return False

    # Verify results
    max_diff = 0
    max_batched_diff = 0
    total_per_trace_passed = 0
    total_stacked_passed = 0
    for i in range(len(all_distances_cpu)):
        cpu_result = all_distances_cpu[i]
        gpu_result = all_distances_gpu_per_trace[i].cpu().numpy()

        trace_diff = np.max(np.abs(cpu_result - gpu_result))
        max_diff = max(max_diff, trace_diff)

        gpu_batched_result = batched_custom_distances[hypothesis_offsets[i]:hypothesis_offsets[i+1]]
        gpu_batched_result_np = gpu_batched_result.cpu().numpy()
        batched_trace_diff = np.max(np.abs(cpu_result - gpu_batched_result_np))
        max_batched_diff = max(max_batched_diff, batched_trace_diff)
        if max_diff < 1e-5:
            total_per_trace_passed += 1
        if max_batched_diff < 1e-5:
            total_stacked_passed += 1
    unified_data.cpu_times["distance_calculation"] = total_cpu_time
    unified_data.gpu_per_trace_times["distance_calculation"] = total_gpu_per_trace_time
    unified_data.gpu_batched_times["distance_calculation"] = total_gpu_batched_time


    unified_data.verification_results["distance_calculation"] = {
        "max_diff": max_diff,
        "max_batched_diff": max_batched_diff,
        "percent_per_trace_passed": 100 * total_per_trace_passed / len(all_distances_cpu),
        "percent_stacked_passed": 100 * total_stacked_passed / len(all_distances_cpu),
        "passed": max_diff < 1e-5
    }

    print(f"    CPU: {total_cpu_time * 1000:.3f}ms")
    print(f"    GPU Per-Trace: {total_gpu_per_trace_time * 1000:.3f}ms")
    print(f"    GPU Batched: {total_gpu_batched_time * 1000:.3f}ms")
    if total_cpu_time > 0:
        per_trace_speedup = total_cpu_time / total_gpu_per_trace_time if total_gpu_per_trace_time > 0 else 0
        batched_speedup = total_cpu_time / total_gpu_batched_time if total_gpu_batched_time > 0 else 0
        print(f"    Per-Trace Speedup: {per_trace_speedup:.2f}x, Batched Speedup: {batched_speedup:.2f}x")
    if max_batched_diff > 0:
        print(f"    Per-trace max diff: {max_diff:.2e}, Batched max diff: {max_batched_diff:.2e}")
    else:
        print(f"    Per-trace max diff: {max_diff:.2e}, Batched: Not implemented/No data")

    return True


def process_pose_evidence_all_approaches(unified_data: UnifiedStepData,
                                       operations: MontyOperations) -> bool:
    """Process pose evidence with CPU, GPU per-trace, and GPU batched approaches."""
    # CPU and GPU Per-Trace processing
    all_pose_evidence_cpu = []
    all_pose_evidence_gpu_per_trace = []
    total_cpu_time = 0
    total_gpu_per_trace_time = 0

    weights_cpu = np.array([1.0, 0.5])  # Default weights

    for i in range(unified_data.num_valid_traces):
        trace_data = unified_data.get_per_trace_item(i)
        if ("query_pose_vectors" not in trace_data or "node_pose_vectors" not in trace_data):
            continue

        query_pose_vectors = trace_data["query_pose_vectors"]
        node_pose_vectors = trace_data["node_pose_vectors"]
        query_poses_fully_defined = trace_data["query_poses_fully_defined"]
        # Get additional data if available
        use_cd = trace_data["use_cd"]

        # CPU implementation
        pose_evidence_cpu, cpu_time = operations.pose_evidence_cpu(
            query_pose_vectors, node_pose_vectors, use_cd, query_poses_fully_defined, weights_cpu
        )
        total_cpu_time += cpu_time
        all_pose_evidence_cpu.append(pose_evidence_cpu)

        # GPU Per-Trace version
        query_poses_gpu = torch.from_numpy(query_pose_vectors.astype(np.float32)).to(operations.device).contiguous()
        node_poses_gpu = torch.from_numpy(node_pose_vectors.astype(np.float32)).to(operations.device).contiguous()
        weights_gpu = torch.from_numpy(weights_cpu.astype(np.float32)).to(operations.device).contiguous()
        use_cd_gpu = torch.from_numpy(use_cd.astype(np.float32)).to(operations.device).contiguous()

        pose_evidence_gpu, gpu_time = operations.pose_evidence_gpu(
            query_poses_gpu, node_poses_gpu, use_cd_gpu, weights_gpu
        )
        total_gpu_per_trace_time += gpu_time
        all_pose_evidence_gpu_per_trace.append(pose_evidence_gpu)

    # Stacked GPU implementation - now uses UnifiedStepData
    total_gpu_batched_time = 0
    batched_pose_evidence = None

    if unified_data.num_valid_traces > 0:
        try:
            # Use UnifiedStepData for batched processing
            batched_pose_evidence, total_gpu_batched_time = operations.pose_evidence_gpu_batched(
                unified_data, weights_cpu
            )
        except Exception as e:
            print(f"    Batched GPU failed: {e}")
            total_gpu_batched_time = 0

    # Store results and timing
    unified_data.cpu_times["pose_evidence"] = total_cpu_time
    unified_data.gpu_per_trace_times["pose_evidence"] = total_gpu_per_trace_time
    unified_data.gpu_batched_times["pose_evidence"] = total_gpu_batched_time

    # Verify results
    max_diff = 0
    max_batched_diff = 0
    total_per_trace_passed = 0
    total_stacked_passed = 0
    for i in range(len(all_pose_evidence_cpu)):
        cpu_result = all_pose_evidence_cpu[i]
        gpu_result = all_pose_evidence_gpu_per_trace[i].cpu().numpy()

        # Per-trace verification
        trace_diff = np.max(np.abs(cpu_result - gpu_result))
        max_diff = max(max_diff, trace_diff)

        # Batched verification if available
        if batched_pose_evidence is not None:
            stacked = unified_data.get_stacked_tensors()
            if "pose_evidence_offsets" in stacked:
                offsets = stacked["pose_evidence_offsets"]
                gpu_batched_result = batched_pose_evidence[offsets[i]:offsets[i+1]]
                gpu_batched_result_np = gpu_batched_result.cpu().numpy()
                batched_trace_diff = np.max(np.abs(cpu_result - gpu_batched_result_np))
                max_batched_diff = max(max_batched_diff, batched_trace_diff)
        if max_diff < 1e-5:
            total_per_trace_passed += 1
        if max_batched_diff < 1e-5:
            total_stacked_passed += 1


    unified_data.verification_results["pose_evidence"] = {
        "max_diff": max_diff,
        "max_batched_diff": max_batched_diff,
        "percent_per_trace_passed": 100 * total_per_trace_passed / len(all_pose_evidence_cpu),
        "percent_stacked_passed": 100 * total_stacked_passed / len(all_pose_evidence_cpu),
        "passed": max_diff < 1e-5
    }


    # Print timing comparison
    print(f"    CPU: {total_cpu_time * 1000:.3f}ms")
    print(f"    GPU Per-Trace: {total_gpu_per_trace_time * 1000:.3f}ms")
    print(f"    GPU Batched: {total_gpu_batched_time * 1000:.3f}ms")
    if total_cpu_time > 0:
        per_trace_speedup = total_cpu_time / total_gpu_per_trace_time if total_gpu_per_trace_time > 0 else 0
        batched_speedup = total_cpu_time / total_gpu_batched_time if total_gpu_batched_time > 0 else 0
        print(f"    Per-Trace Speedup: {per_trace_speedup:.2f}x, Batched Speedup: {batched_speedup:.2f}x")
    print(f"    Per-trace max diff: {max_diff:.2e}, Batched max diff: {max_batched_diff:.2e}")

    return True


def process_radius_evidence_max_all_approaches(unified_data: UnifiedStepData,
                                           operations: MontyOperations) -> bool:
    """Process radius evidence max with CPU, GPU per-trace, and GPU batched approaches."""
    # CPU and GPU Per-Trace processing
    all_final_evidence_cpu = []
    all_final_evidence_gpu_per_trace = []
    total_cpu_time = 0
    total_gpu_per_trace_time = 0

    for i in range(unified_data.num_valid_traces):
        trace_data = unified_data.get_per_trace_item(i)

        radius_evidence = trace_data["radius_evidence"]

        # CPU implementation
        final_evidence_cpu, cpu_time = operations.radius_evidence_max_cpu(radius_evidence)
        total_cpu_time += cpu_time
        all_final_evidence_cpu.append(final_evidence_cpu)

        # GPU Per-Trace version
        radius_evidence_gpu = torch.from_numpy(radius_evidence.astype(np.float32)).to(operations.device)
        final_evidence_gpu, gpu_time = operations.radius_evidence_max_gpu(radius_evidence_gpu)
        total_gpu_per_trace_time += gpu_time
        all_final_evidence_gpu_per_trace.append(final_evidence_gpu)

    # Stacked GPU implementation - now uses UnifiedStepData
    total_gpu_batched_time = 0
    batched_final_evidence = None

    if unified_data.num_valid_traces > 0:
        try:
            # Use UnifiedStepData for batched processing
            batched_final_evidence, total_gpu_batched_time = operations.radius_evidence_max_gpu_batched(unified_data)
        except Exception as e:
            print(f"    Batched GPU failed: {e}")
            total_gpu_batched_time = 0

    if not all_final_evidence_cpu:
        print("    No radius evidence max data processed")
        return False

    # Store results and timing
    unified_data.cpu_times["radius_evidence_max"] = total_cpu_time
    unified_data.gpu_per_trace_times["radius_evidence_max"] = total_gpu_per_trace_time
    unified_data.gpu_batched_times["radius_evidence_max"] = total_gpu_batched_time

    # Verify results
    max_diff = 0
    max_batched_diff = 0
    total_per_trace_passed = 0
    total_stacked_passed = 0
    for i in range(len(all_final_evidence_cpu)):
        cpu_result = all_final_evidence_cpu[i]
        gpu_result = all_final_evidence_gpu_per_trace[i].cpu().numpy()

        # Per-trace verification
        trace_diff = np.max(np.abs(cpu_result - gpu_result))
        max_diff = max(max_diff, trace_diff)

        # Batched verification if available
        if batched_final_evidence is not None:
            stacked = unified_data.get_stacked_tensors()
            if "final_agg_offsets" in stacked:
                offsets = stacked["final_agg_offsets"]
                batched_result = batched_final_evidence[offsets[i]:offsets[i+1]].cpu().numpy()
                batched_trace_diff = np.max(np.abs(cpu_result - batched_result))
                max_batched_diff = max(max_batched_diff, batched_trace_diff)
        if max_diff < 1e-5:
            total_per_trace_passed += 1
        if max_batched_diff < 1e-5:
            total_stacked_passed += 1

    unified_data.verification_results["radius_evidence_max"] = {
        "max_diff": max_diff,
        "max_batched_diff": max_batched_diff,
        "percent_per_trace_passed": 100 * total_per_trace_passed / len(all_final_evidence_cpu),
        "percent_stacked_passed": 100 * total_stacked_passed / len(all_final_evidence_cpu),
        "passed": max_diff < 1e-5
    }

    print(f"    CPU: {total_cpu_time * 1000:.3f}ms")
    print(f"    GPU Per-Trace: {total_gpu_per_trace_time * 1000:.3f}ms")
    print(f"    GPU Batched: {total_gpu_batched_time * 1000:.3f}ms")
    if total_cpu_time > 0:
        per_trace_speedup = total_cpu_time / total_gpu_per_trace_time if total_gpu_per_trace_time > 0 else 0
        batched_speedup = total_cpu_time / total_gpu_batched_time if total_gpu_batched_time > 0 else 0
        print(f"    Per-Trace Speedup: {per_trace_speedup:.2f}x, Batched Speedup: {batched_speedup:.2f}x")
    print(f"    Per-trace max diff: {max_diff:.2e}, Batched max diff: {max_batched_diff:.2e}")

    return True


def process_pose_transformation_all_approaches(unified_data: UnifiedStepData,
                                              operations: MontyOperations) -> bool:
    """Process pose transformation with CPU, GPU per-trace, and GPU batched approaches."""
    # CPU and GPU Per-Trace processing
    all_transformed_cpu = []
    all_transformed_gpu_per_trace = []

    total_cpu_time = 0
    total_gpu_per_trace_time = 0
    # for trace in unified_data.step_traces:
    for i in range(unified_data.num_valid_traces):
        trace_data = unified_data.get_per_trace_item(i)
        channel_poses = trace_data["channel_possible_poses"]
        pose_vectors = trace_data["pose_vectors"]

        # CPU implementation
        transformed_cpu, cpu_time = operations.pose_transformation_cpu(
            pose_vectors, channel_poses
        )
        total_cpu_time += cpu_time
        all_transformed_cpu.append(transformed_cpu)

        # GPU Per-Trace version
        pose_vectors_gpu = torch.from_numpy(pose_vectors).to(operations.device)
        channel_poses_gpu = torch.from_numpy(channel_poses).to(operations.device)

        transformed_gpu, gpu_time = operations.pose_transformation_gpu(
            pose_vectors_gpu, channel_poses_gpu
        )
        total_gpu_per_trace_time += gpu_time
        all_transformed_gpu_per_trace.append(transformed_gpu)

    # Stacked GPU implementation using wrapper function
    total_gpu_batched_time = 0
    batched_transformed = None
    try:
        batched_transformed, pose_offsets, total_gpu_batched_time = operations.pose_transformation_gpu_batched(
            unified_data
        )
    except Exception as e:
        print(f"    Batched GPU failed: {e}")
        total_gpu_batched_time = 0

    # Store results and timing
    unified_data.cpu_times["pose_transformation"] = total_cpu_time
    unified_data.gpu_per_trace_times["pose_transformation"] = total_gpu_per_trace_time
    unified_data.gpu_batched_times["pose_transformation"] = total_gpu_batched_time

    # Verify results
    max_diff = 0
    max_batched_diff = 0
    total_per_trace_passed = 0
    total_stacked_passed = 0
    for i in range(len(all_transformed_cpu)):
        cpu_result = all_transformed_cpu[i]
        gpu_result = all_transformed_gpu_per_trace[i].cpu().numpy()

        # Per-trace verification
        trace_diff = np.max(np.abs(cpu_result - gpu_result))
        max_diff = max(max_diff, trace_diff)

        gpu_batched_result = batched_transformed[pose_offsets[i]:pose_offsets[i+1]]
        gpu_batched_result_np = gpu_batched_result.cpu().numpy()
        batched_trace_diff = np.max(np.abs(cpu_result - gpu_batched_result_np))
        max_batched_diff = max(max_batched_diff, batched_trace_diff)
        if max_diff < 1e-5:
            total_per_trace_passed += 1
        if max_batched_diff < 1e-5:
            total_stacked_passed += 1

    unified_data.verification_results["pose_transformation"] = {
        "max_diff": max_diff,
        "max_batched_diff": max_batched_diff,
        "percent_per_trace_passed": 100 * total_per_trace_passed / len(all_transformed_cpu),
        "percent_stacked_passed": 100 * total_stacked_passed / len(all_transformed_cpu),
        "passed": max_diff < 1e-5
    }

    # Print timing comparison
    print(f"    CPU: {total_cpu_time * 1000:.3f}ms")
    print(f"    GPU Per-Trace: {total_gpu_per_trace_time * 1000:.3f}ms")
    print(f"    GPU Batched: {total_gpu_batched_time * 1000:.3f}ms")
    if total_cpu_time > 0:
        per_trace_speedup = total_cpu_time / total_gpu_per_trace_time if total_gpu_per_trace_time > 0 else 0
        batched_speedup = total_cpu_time / total_gpu_batched_time if total_gpu_batched_time > 0 else 0
        print(f"    Per-Trace Speedup: {per_trace_speedup:.2f}x, Batched Speedup: {batched_speedup:.2f}x")
    print(f"    Per-trace max diff: {max_diff:.2e}, Batched max diff: {max_batched_diff:.2e}")

    return True


def process_evidence_aggregation_all_approaches(unified_data: UnifiedStepData,
                                               operations: MontyOperations) -> bool:
    """Process evidence aggregation with CPU, GPU per-trace, and GPU batched approaches."""
    # CPU and GPU Per-Trace processing
    all_aggregated_cpu = []
    all_aggregated_gpu_per_trace = []
    total_cpu_time = 0
    total_gpu_per_trace_time = 0

    past_weight = 1
    present_weight = 1

    for i in range(unified_data.num_valid_traces):
        trace_data = unified_data.get_per_trace_item(i)
        old_evidence = trace_data["old_evidence"]
        new_evidence = trace_data["new_evidence"]
        current_evidence = trace_data["current_evidence"]
        min_update = trace_data["min_update"]
        hyp_ids_to_test = trace_data["hyp_ids_to_test"]
        evidence_update_threshold = trace_data["evidence_update_threshold"]

        # CPU implementation
        aggregated_cpu, cpu_time = operations.evidence_aggregation_cpu(
            old_evidence, new_evidence, hyp_ids_to_test,
            min_update, past_weight, present_weight
        )

        total_cpu_time += cpu_time
        all_aggregated_cpu.append(aggregated_cpu)

        # GPU Per-Trace version
        old_evidence_gpu = torch.from_numpy(old_evidence).to(operations.device)
        current_evidence_gpu = torch.from_numpy(current_evidence).to(operations.device)

        aggregated_gpu, gpu_time = operations.evidence_aggregation_gpu(
            old_evidence_gpu, current_evidence_gpu, evidence_update_threshold,
            min_update, past_weight, present_weight
        )
        total_gpu_per_trace_time += gpu_time
        all_aggregated_gpu_per_trace.append(aggregated_gpu)

    # Stacked GPU implementation (stacked approach)
    total_gpu_batched_time = 0
    batched_aggregated = None
    if unified_data.num_valid_traces > 0:
        try:
            batched_aggregated, update_offsets, total_gpu_batched_time = operations.evidence_aggregation_gpu_batched(
                unified_data, past_weight, present_weight
            )
        except Exception as e:
            print(f"    Batched GPU failed: {e}")
            total_gpu_batched_time = 0

    if not all_aggregated_cpu:
        print("    No evidence aggregation data found")
        return False

    unified_data.cpu_times["evidence_aggregation"] = total_cpu_time
    unified_data.gpu_per_trace_times["evidence_aggregation"] = total_gpu_per_trace_time
    unified_data.gpu_batched_times["evidence_aggregation"] = total_gpu_batched_time

    # Verify results
    max_diff = 0
    max_batched_diff = 0
    total_per_trace_passed = 0
    total_stacked_passed = 0
    for i in range(len(all_aggregated_cpu)):
        cpu_result = all_aggregated_cpu[i]
        gpu_result = all_aggregated_gpu_per_trace[i].cpu().numpy()

        # Per-trace verification
        trace_diff = np.max(np.abs(cpu_result - gpu_result))
        max_diff = max(max_diff, trace_diff)

        # Batched verification
        gpu_batched_result = batched_aggregated[update_offsets[i]:update_offsets[i+1]]
        gpu_batched_result_np = gpu_batched_result.cpu().numpy()
        batched_trace_diff = np.max(np.abs(cpu_result - gpu_batched_result_np))
        max_batched_diff = max(max_batched_diff, batched_trace_diff)
        if max_diff < 1e-5:
            total_per_trace_passed += 1
        if max_batched_diff < 1e-5:
            total_stacked_passed += 1

    unified_data.verification_results["evidence_aggregation"] = {
        "max_diff": max_diff,
        "max_batched_diff": max_batched_diff,
        "percent_per_trace_passed": 100 * total_per_trace_passed / len(all_aggregated_cpu),
        "percent_stacked_passed": 100 * total_stacked_passed / len(all_aggregated_cpu),
        "passed": max_diff < 1e-5
    }

    # Print timing comparison
    print(f"    CPU: {total_cpu_time * 1000:.3f}ms")
    print(f"    GPU Per-Trace: {total_gpu_per_trace_time * 1000:.3f}ms")
    print(f"    GPU Batched: {total_gpu_batched_time * 1000:.3f}ms")
    if total_cpu_time > 0:
        per_trace_speedup = total_cpu_time / total_gpu_per_trace_time if total_gpu_per_trace_time > 0 else 0
        batched_speedup = total_cpu_time / total_gpu_batched_time if total_gpu_batched_time > 0 else 0
        print(f"    Per-Trace Speedup: {per_trace_speedup:.2f}x, Batched Speedup: {batched_speedup:.2f}x")
    print(f"    Per-trace max diff: {max_diff:.2e}, Batched max diff: {max_batched_diff:.2e}")

    return True

def extract_experiment_name(output_dir: str) -> str:
    """Extract experiment name from output directory path."""
    path = Path(output_dir)
    return path.name


def log_performance_to_csv(results: Dict[str, Any], experiment_name: str, csv_file: str = "gpu_performance_data.csv"):
    """Log performance results to CSV for plotting and analysis."""

    # Define CSV columns
    fieldnames = [
        'experiment_name', 'step', 'function_name', 'num_traces', 'total_hypotheses',
        'cpu_time_ms', 'gpu_per_trace_time_ms', 'gpu_batched_time_ms',
        'per_trace_speedup', 'batched_speedup', 'batch_vs_pertrace_speedup',
        'verification_percent_per_trace_passed', 'verification_percent_stacked_passed', 'max_diff', 'max_batched_diff', 'timestamp'
    ]

    # Check if CSV exists, create with headers if not
    csv_path = Path(csv_file)
    write_header = not csv_path.exists()

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()
            print(f"Created performance log: {csv_path.absolute()}")

        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

        # Log data for each step and function
        for step_result in results['step_results']:
            step = step_result['step']
            num_traces = step_result['num_traces']
            total_hypotheses = step_result['total_hypotheses']

            # Log per-function timing data
            for func_name in results['functions_tested']:
                cpu_time = step_result['cpu_times'].get(func_name, 0) * 1000  # Convert to ms
                gpu_per_trace_time = step_result['gpu_per_trace_times'].get(func_name, 0) * 1000
                gpu_batched_time = step_result['gpu_batched_times'].get(func_name, 0) * 1000

                # Calculate speedups
                per_trace_speedup = cpu_time / gpu_per_trace_time if gpu_per_trace_time > 0 else 0
                batched_speedup = cpu_time / gpu_batched_time if gpu_batched_time > 0 else 0
                batch_vs_pertrace = gpu_per_trace_time / gpu_batched_time if gpu_batched_time > 0 else 0

                # Get verification results
                verification = step_result['verification_results'].get(func_name, {})
                verification_percent_per_trace_passed = verification.get('percent_per_trace_passed', 0)
                verification_percent_stacked_passed = verification.get('percent_stacked_passed', 0)
                max_diff = verification.get('max_diff', verification.get('per_trace_max_diff', 0))
                max_batched_diff = verification.get('max_batched_diff', 0)

                # Write row to CSV
                writer.writerow({
                    'experiment_name': experiment_name,
                    'step': step,
                    'function_name': func_name,
                    'num_traces': num_traces,
                    'total_hypotheses': total_hypotheses,
                    'cpu_time_ms': f"{cpu_time:.3f}",
                    'gpu_per_trace_time_ms': f"{gpu_per_trace_time:.3f}",
                    'gpu_batched_time_ms': f"{gpu_batched_time:.3f}",
                    'per_trace_speedup': f"{per_trace_speedup:.2f}",
                    'batched_speedup': f"{batched_speedup:.2f}",
                    'batch_vs_pertrace_speedup': f"{batch_vs_pertrace:.2f}",
                    'verification_percent_per_trace_passed': verification_percent_per_trace_passed,
                    'verification_percent_stacked_passed': verification_percent_stacked_passed,
                    'max_diff': f"{max_diff:.2e}",
                    'max_batched_diff': f"{max_batched_diff:.2e}",
                    'timestamp': timestamp
                })

    print(f"Performance data logged to: {csv_path.absolute()}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="GPU vs CPU Monty Operations Test")
    parser.add_argument("--output-dir", required=True, help="Experiment output directory")
    parser.add_argument("--function", choices=["displacement", "knn_search", "distance",
                       "pose_evidence", "aggregation"],
                       help="Test specific function (default: test all)")
    parser.add_argument("--csv-file", default="gpu_performance_data.csv",
                       help="CSV file to log performance data (default: gpu_performance_data.csv)")

    args = parser.parse_args()

    # Extract experiment name for logging
    experiment_name = extract_experiment_name(args.output_dir)

    # Load profiling data
    traces = load_profiling_data(args.output_dir)
    if not traces:
        return

    # Set up GPU device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # Initialize operations
    operations = MontyOperations(device)

    # Run analysis
    print(f"\n=== Running Analysis ===")
    if args.function:
        print(f"Testing function: {args.function}")
    else:
        print("Testing all functions")

    # results = run_step_analysis(traces, operations, args.function)
    # results = run_step_analysis(traces, operations, args.function)
    results = run_step_analysis(traces, operations, args.function)

    # Print comprehensive final summary
    print(f"\n=== ANALYSIS RESULTS ===")
    print(f"Steps processed: {results['num_steps']}")
    print(f"Total traces: {results['total_traces']}")
    print(f"Functions tested: {', '.join(results['functions_tested'])}")

    print(f"\n--- TIMING COMPARISON ---")
    print(f"CPU Time:             {results['total_cpu_time'] * 1000:.3f}ms")
    print(f"GPU Per-Trace Time:   {results['total_gpu_per_trace_time'] * 1000:.3f}ms")
    print(f"GPU Batched Time:     {results['total_gpu_batched_time'] * 1000:.3f}ms")

    # Calculate and display speedups
    if results["total_cpu_time"] > 0:
        per_trace_speedup = results["total_cpu_time"] / results["total_gpu_per_trace_time"] if results["total_gpu_per_trace_time"] > 0 else 0
        batched_speedup = results["total_cpu_time"] / results["total_gpu_batched_time"] if results["total_gpu_batched_time"] > 0 else 0
        batch_vs_pertrace = results["total_gpu_per_trace_time"] / results["total_gpu_batched_time"] if results["total_gpu_batched_time"] > 0 else 0

        print(f"\n--- SPEEDUP ANALYSIS ---")
        print(f"GPU Per-Trace vs CPU:     {per_trace_speedup:.2f}x")
        print(f"GPU Batched vs CPU:       {batched_speedup:.2f}x")
        print(f"GPU Batched vs Per-Trace: {batch_vs_pertrace:.2f}x")

    # Print verification summary
    print(f"\n=== VERIFICATION SUMMARY ===")
    all_passed = True
    for step_result in results["step_results"]:
        for func_name, verification in step_result["verification_results"].items():
            if not verification.get("passed", False):
                all_passed = False
                print(f"{func_name}: FAILED verification")

    if all_passed:
        print("All functions passed verification")

    # Log performance data to CSV for analysis and plotting
    print(f"\n=== LOGGING PERFORMANCE DATA ===")
    try:
        log_performance_to_csv(results, experiment_name, args.csv_file)
        print(f"Performance data successfully logged")
    except Exception as e:
        print(f"Error logging performance data: {e}")


if __name__ == "__main__":
    main()
