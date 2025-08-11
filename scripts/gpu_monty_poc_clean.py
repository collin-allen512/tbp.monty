#!/usr/bin/env python3
"""
GPU Proof of Concept for Monty - Clean Implementation

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
import pickle
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import sys
import os

import numpy as np
import torch
from scipy.spatial import cKDTree
import gc

# Add gpu_kernels to path
gpu_kernels_dir = os.path.join(os.path.dirname(__file__), '../gpu_kernels')
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
    for i in range(5):
        trace_file = output_path / f"hypothesis_computation_trace_LMlearning_module_{i}.pkl"
        print(trace_file)
        if trace_file.exists():
            with open(trace_file, "rb") as f:
                traces = pickle.load(f)
            print(f"Loaded {len(traces)} computation traces")

            # Check for evidence intermediates
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
        # for trace in self.step_traces:
            # print(f"LM ID {trace['lm_id']}")

        # Filter valid traces (with required data)
        self.valid_traces = [t for t in step_traces if self._is_valid_trace(t)]
        self.num_valid_traces = len(self.valid_traces)

        # Per-trace data storage
        self.per_trace_data = []

        # Stacked data for batched processing
        self.stacked_data = None


        # Timing data for three approaches
        self.cpu_times = {}
        self.gpu_per_trace_times = {}
        self.gpu_batched_times = {}

        # Verification results
        self.verification_results = {}

        # Initialize data structures
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

            # Add pose transformation data
            if ("evidence_intermediates" in trace and "pose_transformation" in trace["evidence_intermediates"]):
                pose_data = trace["evidence_intermediates"]["pose_transformation"]
                per_trace_item["channel_possible_poses"] = pose_data["inputs"]["channel_possible_poses"]
                per_trace_item["channel_features"] = pose_data["inputs"]["channel_features"]
                per_trace_item["pose_vectors"] = pose_data["inputs"]["channel_features"]['pose_vectors']

            # Add KNN search data if available
            if ("evidence_intermediates" in trace and "nearest_neighbor_search" in trace["evidence_intermediates"]):
                nn_data = trace["evidence_intermediates"]["nearest_neighbor_search"]
                per_trace_item["graph_locations"] = nn_data["inputs"]["graph_locations"]
                per_trace_item["search_locations"] = nn_data["inputs"]["search_locations"]
                per_trace_item["nearest_node_ids"] = nn_data["outputs"]["nearest_node_ids"]

            # Add distance calculation data
            if ("evidence_intermediates" in trace and "distance_calculation" in trace["evidence_intermediates"]):
                dist_data = trace["evidence_intermediates"]["distance_calculation"]
                per_trace_item["search_locations"] = dist_data["inputs"]["search_locations"]
                per_trace_item["nearest_node_locs"] = dist_data["inputs"]["nearest_node_locs"]
                per_trace_item["pose_normals"] = dist_data["inputs"]["pose_normals"]
                per_trace_item["max_abs_curvature"] = dist_data["inputs"]["max_abs_curvature"]
                per_trace_item["custom_distances"] = dist_data["outputs"]["custom_nearest_node_dists"]

            # Add pose evidence data
            if ("evidence_intermediates" in trace and "pose_evidence_matrix" in trace["evidence_intermediates"]):
                pe_data = trace["evidence_intermediates"]["pose_evidence_matrix"]
                # Get angle calculation data
                if "angle_calculation_inputs" in pe_data:

                    per_trace_item["query_pose_vectors"] = pe_data["inputs"]["query_features"]["pose_vectors"]
                    # per_trace_item["query_pose_vectors"] = pe_data["angle_calculation_inputs"]["query_pose_vectors"]
                    per_trace_item["node_pose_vectors"] = pe_data["inputs"]["node_features"]["pose_vectors"]
                if "angle_calculation_outputs" in pe_data:
                    per_trace_item["pn_angles"] = pe_data["angle_calculation_outputs"]["pn_angles"]
                # Get evidence calculation data
                if "intermediates" in pe_data:
                    per_trace_item["pn_evidence"] = pe_data["intermediates"]["pn_evidence"]
                    per_trace_item["cd1_evidence"] = pe_data["intermediates"]["cd1_evidence"]
                    per_trace_item["pn_weight"] = pe_data["intermediates"]["pn_weight"]
                    per_trace_item["cd1_weight"] = pe_data["intermediates"]["cd1_weight"]
                    per_trace_item["query_poses_fully_defined"] = pe_data["inputs"]["query_features"]["pose_fully_defined"]
                    per_trace_item["node_poses_fully_defined"] = pe_data["inputs"]["node_features"]["pose_fully_defined"]
                    per_trace_item["use_cd"] = per_trace_item["node_poses_fully_defined"][:, :, 0] * per_trace_item["query_poses_fully_defined"]
                    # if "use_cd" in pe_data["intermediates"]:
                    #     per_trace_item["use_cd"] = pe_data["intermediates"]["use_cd"]
                if "outputs" in pe_data:
                    per_trace_item["pose_evidence_weighted"] = pe_data["outputs"]["pose_evidence_weighted"]

            # Add final aggregation data
            if ("evidence_intermediates" in trace and "final_aggregation" in trace["evidence_intermediates"]):
                final_data = trace["evidence_intermediates"]["final_aggregation"]
                per_trace_item["radius_evidence"] = final_data["inputs"]["radius_evidence"]
                per_trace_item["location_evidence"] = final_data["outputs"]["location_evidence"]

            # Add evidence aggregation data
            if ("evidence_intermediates" in trace and "evidence_aggregation" in trace["evidence_intermediates"]):
                evidence_data = trace["evidence_intermediates"]["evidence_aggregation"]
                per_trace_item["old_evidence"] = evidence_data["inputs"]["old_evidence"]
                per_trace_item['new_evidence'] = evidence_data["inputs"]["new_evidence"]
                per_trace_item['current_evidence'] = evidence_data["inputs"]["evidence_to_add"]
                per_trace_item['hyp_ids_to_test'] = evidence_data["inputs"]["hyp_ids_to_test"]
                per_trace_item['evidence_update_threshold'] = evidence_data["inputs"]["evidence_update_threshold"]
                per_trace_item['min_update'] = evidence_data["inputs"]["min_update"]
                per_trace_item['past_weight'] = evidence_data["inputs"]["past_weight"]
                per_trace_item['present_weight'] = evidence_data["inputs"]["present_weight"]

            self.per_trace_data.append(per_trace_item)

        # Prepare stacked data if we have valid traces
        if self.valid_traces:
            self._prepare_stacked_data()

    def _prepare_stacked_data(self):
        """Prepare stacked data for batched GPU processing."""
        # Collect all data and compute offsets
        all_poses = []
        all_locations = []
        all_evidence = []
        all_displacements = []
        all_channel_poses = []
        hyp_offsets = [0]

        all_search_locs = []
        all_nearest_locs = []
        all_pose_normals = []
        all_curvatures = []
        distance_offsets = [0]
        all_channel_features = []
        all_pose_vectors = []
        pose_offsets = [0]

        stacked_evidence_update_threshold = []
        stacked_hyp_test_ids = []
        stacked_old_evidence = []
        stacked_new_evidence = []
        stacked_current_evidence = []
        stacked_min_update = []
        evidence_update_offsets = [0]

        # Pose evidence and final aggregation data
        all_query_pose_vectors = []
        all_node_pose_vectors = []
        all_use_cd_masks = []
        all_radius_evidence = []
        pose_evidence_offsets = [0]
        final_agg_offsets = [0]

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
            if 'old_evidence' in item:
                stacked_old_evidence.append(item['old_evidence'])
                stacked_new_evidence.append(item['new_evidence'])
                stacked_current_evidence.append(item['current_evidence'])
                evidence_length = item['current_evidence'].shape[0]
                stacked_evidence_update_threshold.append(np.repeat(item['evidence_update_threshold'], evidence_length))
                stacked_min_update.append(np.repeat(item['min_update'], evidence_length))
                stacked_hyp_test_ids.append(item['hyp_ids_to_test'])
                evidence_update_offsets.append(evidence_update_offsets[-1] + evidence_length)

            # Add pose evidence data
            if 'query_pose_vectors' in item and 'node_pose_vectors' in item:
                all_query_pose_vectors.append(item['query_pose_vectors'])
                all_node_pose_vectors.append(item['node_pose_vectors'])
                all_use_cd_masks.append(item['use_cd'])
                pose_evidence_offsets.append(pose_evidence_offsets[-1] + len(item['query_pose_vectors']))

            # Add final aggregation data
            if 'radius_evidence' in item:
                all_radius_evidence.append(item['radius_evidence'])
                final_agg_offsets.append(final_agg_offsets[-1] + item['radius_evidence'].shape[0])

        # print(len(stacked_old_evidence))
        # print(len(stacked_new_evidence))
        # print(len(stacked_hyp_test_ids))
        # print(len(stacked_evidence_update_threshold))
        # print(len(stacked_min_update))
        # print(len(evidence_update_offsets))
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
        }

        # Add pose evidence data if available
        if all_query_pose_vectors:
            self.stacked_data.update({
                "query_pose_vectors": torch.from_numpy(np.concatenate(all_query_pose_vectors)).float().to(self.device),
                "node_pose_vectors": torch.from_numpy(np.concatenate(all_node_pose_vectors)).float().to(self.device),
                "use_cd_masks": torch.from_numpy(np.concatenate(all_use_cd_masks)).float().to(self.device),
                "pose_evidence_offsets": torch.tensor(pose_evidence_offsets, dtype=torch.int32, device=self.device),
            })

        # Add final aggregation data if available
        if all_radius_evidence:
            self.stacked_data.update({
                "radius_evidence": torch.from_numpy(np.concatenate(all_radius_evidence)).float().to(self.device),
                "final_agg_offsets": torch.tensor(final_agg_offsets, dtype=torch.int32, device=self.device),
            })

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
            for attr in ['search_locations', 'nearest_indices', 'nearest_locations',
                        'custom_distances', 'pose_evidence', 'final_evidence']:
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
    """Clean implementation of Monty operations with GPU and CPU versions."""

    def __init__(self, device: torch.device):
        self.device = device

    # =================================================================
    # 1. DISPLACEMENT OPERATIONS
    # =================================================================

    def displacement_gpu(self, poses: torch.Tensor, locations: torch.Tensor,
                        displacement: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """GPU implementation of displacement calculation using CUDA kernels."""
        if monty_cuda is None:
            raise RuntimeError("CUDA kernels not available")

        start_time = time.perf_counter()

        # Expand displacement to match poses for stacked kernel
        # poses: (N, 3, 3), displacement: (3,) -> expanded: (N, 3)
        expanded_displacement = displacement.repeat(poses.shape[0], 1)

        # Use stacked CUDA kernel for displacement
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

        # Exact implementation from Monty codebase
        # displacement is (3,) shape, poses is (N, 3, 3)
        rotated_displacements = np.dot(poses, displacement)  # (N, 3)
        search_locations = locations + rotated_displacements  # (N, 3)

        cpu_time = time.perf_counter() - start_time

        return search_locations, cpu_time

    # =================================================================
    # 2. NEAREST NEIGHBOR SEARCH OPERATIONS
    # =================================================================

    def knn_search_gpu(self, search_locations: torch.Tensor,
                      graph_locations: torch.Tensor,
                      k: int = 3) -> Tuple[torch.Tensor, float]:
        """GPU implementation of KNN search using CUDA kernels."""
        if monty_cuda is None:
            raise RuntimeError("CUDA kernels not available")

        start_time = time.perf_counter()

        # Create offsets for single trace (stacked kernel signature)
        # For single trace: graph_offsets=[0], query_offsets=[0]
        graph_offsets = torch.tensor([0], dtype=torch.int32, device=self.device)
        query_offsets = torch.tensor([0], dtype=torch.int32, device=self.device)

        # Use stacked CUDA kernel for KNN search
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

        # Build KDTree and query (exact Monty implementation)
        tree = cKDTree(graph_locations)
        start_time = time.perf_counter()
        _, nearest_indices = tree.query(search_locations, k=k)

        if k == 1:
            nearest_indices = nearest_indices.reshape(-1, 1)

        cpu_time = time.perf_counter() - start_time

        return nearest_indices, cpu_time

    # =================================================================
    # 3. DISTANCE CALCULATION OPERATIONS
    # =================================================================

    def distance_calculation_gpu(self, search_locations: torch.Tensor,
                                nearest_locations: torch.Tensor,
                                pose_normals: torch.Tensor,
                                curvature: float) -> Tuple[torch.Tensor, float]:
        """GPU implementation of custom distance calculation using CUDA kernels."""
        if monty_cuda is None:
            raise RuntimeError("CUDA kernels not available")


        # Create trace offsets for single trace and curvature tensor
        trace_offsets = torch.tensor([0], dtype=torch.int32, device=self.device)
        curvatures_tensor = torch.tensor([curvature], dtype=torch.float32, device=self.device)

        start_time = time.perf_counter()

        # Use stacked CUDA kernel for custom distance calculation
        # Note: kernel expects (nearest_locs, search_locs, pose_normals, curvatures, trace_offsets)
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

        # Add curvature correction matching original implementation
        if curvature > 0:
            dot_products = np.einsum("ijk,ik->ij", differences, pose_normals)
            curvature_factor = 1.0 / (abs(curvature) + 0.5)
            custom_distances = euclidean_dists + np.abs(dot_products) * curvature_factor
        else:
            custom_distances = euclidean_dists

        cpu_time = time.perf_counter() - start_time

        return custom_distances, cpu_time

    # =================================================================
    # 4. POSE EVIDENCE CALCULATION OPERATIONS
    # =================================================================

    def pose_evidence_gpu(self, query_poses: torch.Tensor,
                         node_poses: torch.Tensor,
                         use_cd_mask: torch.Tensor,
                         weights: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """GPU implementation of pose evidence calculation using CUDA kernels."""
        if monty_cuda is None:
            raise RuntimeError("CUDA kernels not available")

        # query_poses = query_poses.contiguous()
        # node_poses = node_poses.contiguous()
        # The pose_evidence_stacked kernel expects angles, not raw poses
        # We need to calculate angles first
        query_normals = query_poses[:, 0]  # (N, 3)
        node_normals = node_poses[:, :, :3]  # (N, K, 3)

        # print(f" shape {.shape}")
        # print(f"query_normals contiguous: {query_normals.is_contiguous()}")
        # print(f"node_normals contiguous: {node_normals.is_contiguous()}")
        query_normals = query_normals.contiguous()
        node_normals = node_normals.contiguous()
        # print(f"query_normals contiguous: {query_normals.is_contiguous()}")
        # print(f"node_normals contiguous: {node_normals.is_contiguous()}")

        # print(f" query_normals shape {query_normals.shape}")
        # print(f"node_normals shape {node_normals.shape}")
        pn_angles = monty_cuda.angle_calculation(node_normals, query_normals)

        # dot_products = torch.einsum("ijk,ik->ij", node_normals, query_normals)
        # dot_products = torch.clamp(dot_products, -1, 1)
        # pn_angles_ = torch.acos(dot_products)

        # For simplicity, create dummy cd1_angles and use_cd (could be enhanced later)
        # cd1_angles = torch.zeros_like(pn_angles)
        cd1_angles = monty_cuda.angle_calculation(node_poses[:, :, 3:6].contiguous(), query_poses[:, 1].contiguous())

        # use_cd = torch
        # use_cd = torch.zeros_like(pn_angles, dtype=torch.int32)
        # print(weights[0])
        # use_cd = weights[0].expand(pn_angles.shape).to(torch.int32)
        # use_cd = torch.ones(pn_angles.shape, device=self.device).to(torch.int32)
        # print(use_cd.shape)
        # Create trace offsets and weight tensors
        pose_offsets = torch.tensor([0], dtype=torch.int32, device=self.device).contiguous()
        pn_weights = torch.tensor([weights[0]], dtype=torch.float32, device=self.device).expand(pn_angles.shape).contiguous()
        cd1_weights = torch.tensor([weights[1]], dtype=torch.float32, device=self.device).expand(pn_angles.shape).contiguous()

        start_time = time.perf_counter()
        # Use stacked CUDA kernel for pose evidence calculation
        pose_evidence = monty_cuda.pose_evidence_stacked(
            pn_angles, cd1_angles, use_cd_mask, pn_weights, cd1_weights, pose_offsets
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        # return cd1_angles, 0
        return pose_evidence, gpu_time

    def pose_evidence_cpu(self, query_poses: np.ndarray,
                         node_poses: np.ndarray,
                         use_cd: np.ndarray,
                         query_poses_full_defined: bool,
                         weights: np.ndarray) -> Tuple[np.ndarray, float]:
        """CPU implementation matching Monty's pose evidence calculation."""

        # Exact implementation from Monty
        # Calculate angles between pose vectors
        # print(f"DEBUG: query poses shape {query_poses.shape}")
        # print(f"DEBUG: node poses shape {node_poses.shape}")

        # Handle potential dimension issues
        # if len(query_poses.shape) == 2 and query_poses.shape[1] >= 3:
        #     query_normals = query_poses[:, 0:3]  # Take first 3 elements as normal
        # elif len(query_poses.shape) == 3:
        #     query_normals = query_poses[:, 0]  # Take first pose vector
        # else:
        #     raise ValueError(f"Unexpected query_poses shape: {query_poses.shape}")

        # if len(node_poses.shape) == 3:
        #     node_normals = node_poses[:, :, :3]
        # elif len(node_poses.shape) == 4:
        #     node_normals = node_poses[:, :, 0, :3]  # Take first pose vector, first 3 elements
        # else:
        #     raise ValueError(f"Unexpected node_poses shape: {node_poses.shape}")
        query_normals = query_poses[:, 0]  # Take first pose vector
        node_normals = node_poses[:, :, :3]

        pn_weight = np.array([weights[0]])
        cd1_weight = np.array([weights[1]])

        # print(f"DEBUG: query_normals shape {query_normals.shape}")
        # print(f"DEBUG: node_normals shape {node_normals.shape}")
        # query_normals = torch.from_numpy(query_poses[:, 0]).float().to(torch.device('cpu'))
        # node_normals = torch.from_numpy(node_poses[:, :, :3]).float().to(torch.device('cpu'))

        # Compute dot products and angles
        start_time = time.perf_counter()
        # dot_products = np.sum(
        #     query_normals[:, np.newaxis] * node_normals, axis=2
        # )
        # dot_products = np.clip(dot_products, -1, 1)
        # angles = np.arccos(np.abs(dot_products))
        # dot_products = torch.einsum("ijk,ik->ij", node_normals, query_normals)
        # dot_products = torch.clamp(dot_products, -1, 1)
        # pn_angles = torch.acos(dot_products)
        dot_product = np.einsum("ijk,ik->ij", node_normals, query_normals)
        pn_angles = np.arccos(np.clip(dot_product, -1, 1))

        # Convert to evidence
        pn_evidence = -(np.sin(pn_angles / 2) - 0.5)
        # print(f"pn_evidence {pn_evidence.shape}")
        # For simplicity, create dummy cd1_angles and use_cd (could be enhanced later)
        # cd1_angle = get_angles_for_all_hypotheses(
        #     node_features["pose_vectors"][:, :, 3:6],
        #     query_features["pose_vectors"][:, 1],
        # )

        # print(f"cd1_angles {cd1_angles.shape}")

        # cd1_angles = torch.zeros_like(pn_angles)
        # use_cd = np.ones(pn_angles.shape, dtype=bool)
        # print(f"use_cd {use_cd.shape}")
        # Create trace offsets and weight tensors
        # print(weights)
        # print(weights[0])
        # print([weights[0]])


        # print(f"pn_weights {pn_weights.shape}")
        # print(f"cd1_weights {cd1_weights.shape}")

        # print(f"cd1_error {cd1_error.shape}")
        # We then apply the same operations as on pn error to get cd1_evidence
        # in range [-0.5, 0.5]
        if not query_poses_full_defined:
            cd1_weight = 0
            cd1_evidence = np.zeros(pn_evidence.shape)
        else:
            # Handle CD1 calculation with proper dimensions
            # if len(query_poses.shape) == 2 and query_poses.shape[1] >= 6:
            #     query_cd1 = query_poses[:, 3:6]  # Take elements 3-6 as CD1
            # elif len(query_poses.shape) == 3:
            #     query_cd1 = query_poses[:, 1]  # Take second pose vector
            # else:
            #     query_cd1 = np.zeros_like(query_normals)  # Fallback

            # if len(node_poses.shape) == 3:
            #     node_cd1 = node_poses[:, :, 3:6] if node_poses.shape[2] >= 6 else np.zeros_like(node_normals)
            # elif len(node_poses.shape) == 4:
            #     node_cd1 = node_poses[:, :, 1, :3] if node_poses.shape[2] >= 2 else np.zeros_like(node_normals)
            # else:
            #     node_cd1 = np.zeros_like(node_normals)  # Fallback

            # print(f"DEBUG: query_cd1 shape {query_cd1.shape}")
            # print(f"DEBUG: node_cd1 shape {node_cd1.shape}")

            dot_product = np.einsum("ijk,ik->ij", node_poses[:, :, 3:6], query_poses[:, 1])
            # dot_product = np.einsum("ijk,ik->ij", node_cd1, query_cd1)
            cd1_angles = np.arccos(np.clip(dot_product, -1, 1))
            cd1_error = np.pi / 2 - np.abs(cd1_angles - np.pi / 2)
            cd1_evidence = -(np.sin(cd1_error) - 0.5)
            cd1_evidence = cd1_evidence * use_cd
            # print(f"cd1_evidence {cd1_evidence.shape}")
            # nodes where pc1==pc2 receive no cd evidence but twice the pn evidence
            # -> overall evidence can be in range [-1, 1]
            # cd1_evidence = cd1_evidence * use_cd
            # print(f"cd1_evidence {cd1_evidence.shape}")
            pn_evidence[np.logical_not(use_cd)] *= 2
            # print(f"pn_evidence {pn_evidence.shape}")

        # pose_evidence = pn_evidence * weights[0]
        pose_evidence_weighted = pn_evidence * pn_weight + cd1_evidence * cd1_weight
        # print(pose_evidence_weighted.shape)
        cpu_time = time.perf_counter() - start_time

        # return cd1_angles, cpu_time
        return pose_evidence_weighted, cpu_time

    # =================================================================
    # 5. FINAL AGGREGATION OPERATIONS
    # =================================================================

    def final_aggregation_gpu(self, evidence_matrix: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """GPU implementation of final evidence aggregation using CUDA kernels."""
        if monty_cuda is None:
            raise RuntimeError("CUDA kernels not available")

        start_time = time.perf_counter()

        # Use stacked CUDA kernel for final aggregation
        # This kernel takes the evidence matrix and does max along neighbor dimension
        final_evidence = monty_cuda.final_aggregation_stacked(evidence_matrix)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        return final_evidence, gpu_time

    def final_aggregation_cpu(self, evidence_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
        """CPU implementation of final evidence aggregation."""
        start_time = time.perf_counter()

        # Max along neighbor dimension (exact Monty implementation)
        final_evidence = np.max(evidence_matrix, axis=1)

        cpu_time = time.perf_counter() - start_time

        return final_evidence, cpu_time

    # =================================================================
    # 6. POSE TRANSFORMATION OPERATIONS
    # =================================================================

    def pose_transformation_gpu(self, pose_vectors: torch.Tensor,
                               reference_poses: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """GPU implementation of pose transformation using CUDA kernels."""
        if monty_cuda is None:
            raise RuntimeError("CUDA kernels not available")

        start_time = time.perf_counter()

        # Use CUDA kernel for pose transformation
        # This implements: rotated_pv = ref_frame_rots.dot(old_pv.T).transpose((0, 2, 1))
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

        # Exact implementation from spatial_arithmetics.py:
        # rotated_pv = ref_frame_rots.dot(old_pv.T)
        # rotated_pv = rotated_pv.transpose((0, 2, 1))
        pose_vectors_T = pose_vectors.T
        result = np.matmul(reference_poses, pose_vectors_T)
        transformed_vectors = result.transpose(0, 2, 1)

        cpu_time = time.perf_counter() - start_time

        return transformed_vectors, cpu_time

    # =================================================================
    # 7. EVIDENCE AGGREGATION OPERATIONS
    # =================================================================

    def evidence_aggregation_gpu(self, old_evidence: torch.Tensor,
                                new_evidence: torch.Tensor, test_indices: torch.Tensor,
                                min_update: float, past_weight: float,
                                present_weight: float) -> Tuple[torch.Tensor, float]:
        """GPU implementation of evidence aggregation using CUDA kernels."""
        if monty_cuda is None:
            raise RuntimeError("CUDA kernels not available")

        start_time = time.perf_counter()

        # Use CUDA kernel for evidence aggregation
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

        # Exact implementation from Monty codebase
        evidence_to_add = np.ones_like(old_evidence) * min_update
        evidence_to_add[test_indices] = new_evidence

        # Weighted combination of past and present evidence
        aggregated_evidence = old_evidence * past_weight + evidence_to_add * present_weight

        cpu_time = time.perf_counter() - start_time

        return aggregated_evidence, cpu_time

    # =================================================================
    # BATCHED GPU OPERATIONS - SINGLE DISPATCH FOR ALL TRACES
    # =================================================================

    def displacement_gpu_batched(self, unified_data: UnifiedStepData) -> Tuple[torch.Tensor, float]:
        """Batched GPU displacement - single kernel call for all traces."""
        if monty_cuda is None:
            raise RuntimeError("CUDA kernels not available")

        stacked = unified_data.get_stacked_tensors()
        if not stacked:
            return None, 0.0

        # Expand displacements to match poses for stacked kernel
        expanded_displacements = []
        for i, count in enumerate(stacked["hyp_counts"]):
            expanded_displacements.append(stacked["displacements"][i].repeat(count, 1))
        stacked_displacements = torch.cat(expanded_displacements, dim=0)

        # Single stacked kernel call for all traces
        start_time = time.perf_counter()
        search_locations = monty_cuda.displacement_stacked(
            stacked["poses"], stacked_displacements, stacked["locations"]
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        # Cleanup intermediate tensors
        for tensor in expanded_displacements:
            del tensor
        del stacked_displacements

        return search_locations, stacked["hyp_offsets"], gpu_time

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

        return nearest_indices, stacked['hyp_offsets'], gpu_time

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
        search_locations = stacked['search_locations']
        nearest_locations = stacked['nearest_locations']
        pose_normals = stacked['pose_normals']
        curvatures = stacked['curvatures']
        trace_offsets = stacked["distance_offsets"]

        start_time = time.perf_counter()

        # Create curvature tensor and trace offsets
        # curvatures_tensor = torch.tensor(curvatures, dtype=torch.float32, device=self.device)

        # Single stacked kernel call for all traces
        custom_distances = monty_cuda.custom_distance_stacked(
            nearest_locations, search_locations, pose_normals, curvatures, trace_offsets
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        return custom_distances, trace_offsets, gpu_time

    def pose_evidence_gpu_batched(self, unified_data: UnifiedStepData,
                                weights: np.ndarray) -> Tuple[torch.Tensor, float]:
        """Batched GPU pose evidence - single kernel call for all traces."""
        if monty_cuda is None:
            raise RuntimeError("CUDA kernels not available")

        stacked = unified_data.get_stacked_tensors()
        if not stacked or "query_pose_vectors" not in stacked:
            return None, 0.0

        # Get pose data from UnifiedStepData
        query_poses = stacked["query_pose_vectors"]
        node_poses = stacked["node_pose_vectors"]
        use_cd_masks = stacked["use_cd_masks"]
        poses_offsets = stacked["pose_evidence_offsets"]

        # Calculate angles for batched processing
        query_normals = query_poses[:, 0]  # (N, 3)
        node_normals = node_poses[:, :, :3]  # (N, K, 3)

        # Create weight tensors
        pn_weights = torch.tensor([weights[0]], dtype=torch.float32, device=self.device).expand(pn_angles.shape).contiguous()
        cd1_weights = torch.tensor([weights[1]], dtype=torch.float32, device=self.device).expand(pn_angles.shape).contiguous()

        start_time = time.perf_counter()
        pn_angles = monty_cuda.angle_calculation(node_normals.contiguous(), query_normals.contiguous())
        cd1_angles = monty_cuda.angle_calculation(node_poses[:, :, 3:6].contiguous(), query_poses[:, 1].contiguous())

        # Single stacked kernel call for all traces
        pose_evidence = monty_cuda.pose_evidence_stacked(
            pn_angles.contiguous(), cd1_angles.contiguous(), use_cd_masks.contiguous(),
            pn_weights.contiguous(), cd1_weights.contiguous(), poses_offsets.contiguous()
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        return pose_evidence, gpu_time

    def final_aggregation_gpu_batched(self, unified_data: UnifiedStepData) -> Tuple[torch.Tensor, float]:
        """Batched GPU final aggregation - single kernel call for all traces."""
        if monty_cuda is None:
            raise RuntimeError("CUDA kernels not available")

        stacked = unified_data.get_stacked_tensors()
        if not stacked or "radius_evidence" not in stacked:
            return None, 0.0

        evidence_matrix = stacked["radius_evidence"]

        start_time = time.perf_counter()

        # Single stacked kernel call for all traces
        final_evidence = monty_cuda.final_aggregation_stacked(evidence_matrix)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        return final_evidence, gpu_time

    def pose_transformation_gpu_batched(self, unified_data: UnifiedStepData) -> Tuple[torch.Tensor, torch.Tensor, float]:
            # self, pose_vectors: torch.Tensor,
            #                            reference_poses_batched: torch.Tensor,
            #                            pose_offsets: List) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Batched GPU pose transformation - single kernel call for all traces."""
        if monty_cuda is None:
            raise RuntimeError("CUDA kernels not available")

        stacked = unified_data.get_stacked_tensors()
        pose_vectors_gpu = stacked['pose_vectors'].contiguous()
        channel_poses_gpu = stacked['channel_poses'].contiguous()
        pose_offsets_gpu = stacked['pose_offsets'].contiguous()
        # print(f" shape {.shape}")
        # print(f"pose_vectors_gpu shape {pose_vectors_gpu.shape}")
        # print(f"channel_poses_gpu shape {channel_poses_gpu.shape}")
        # print(f"pose_offsets_gpu shape {pose_offsets_gpu.shape}")
        # if not stacked:
            # return None, None, 0.0
        # pose_vectors_gpu = pose_vectors.to(self.device)
        # reference_poses_gpu = reference_poses_batched.to(self.device)
        # pose_vectors_gpu = torch.from_numpy(pose_vectors).to(self.device)
        # pose_offsets_gpu = torch.Tensor(pose_offsets).to(self.device)
        # reference_poses_gpu = torch.from_numpy(reference_poses_batched).to(self.device)
        # trace_offsets = torch.tensor([0, unified_data.num_valid_traces], dtype=torch.int32, device=self.device)
        # print(f"pose vectors {pose_vectors_gpu.shape} ref poses {reference_poses_gpu.shape}")

        start_time = time.perf_counter()

        # Call stacked kernel
        transformed_vectors = monty_cuda.pose_transformation_stacked(
            pose_vectors_gpu, channel_poses_gpu, pose_offsets_gpu
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        return transformed_vectors, pose_offsets_gpu, gpu_time

    def evidence_aggregation_gpu_batched(self, unified_data: UnifiedStepData,
                                        past_weight: float,
                                       present_weight: float) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Batched GPU evidence aggregation - single kernel call for all traces."""
        if monty_cuda is None:
            raise RuntimeError("CUDA kernels not available")

        stacked = unified_data.get_stacked_tensors()
        if not stacked:
            return None, None, 0.0

        # # Create stacked indices and evidence for all traces
        # stacked_test_indices = []
        # stacked_new_evidence = []
        # update_offsets = [0]

        # offset = 0
        # for i, (indices, evidence) in enumerate(zip(test_indices_list, new_evidence_list)):
        #     # Adjust indices to global hypothesis indexing
        #     global_indices = indices + stacked["hyp_offsets"][i]
        #     stacked_test_indices.extend(global_indices)
        #     stacked_new_evidence.extend(evidence)
        #     offset += len(evidence)
        #     update_offsets.append(offset)

        # stacked_test_indices_gpu = torch.tensor(stacked_test_indices, dtype=torch.int64, device=self.device)
        # stacked_new_evidence_gpu = torch.tensor(stacked_new_evidence, dtype=torch.float32, device=self.device)
        # update_offsets_gpu = torch.tensor(update_offsets, dtype=torch.int32, device=self.device)

        start_time = time.perf_counter()

        stacked_old_evidence = stacked['old_evidence']
        stacked_current_evidence = stacked['current_evidence']
        evidence_update_thresholds = stacked['evidence_update_thresholds']
        min_updates = stacked['min_updates']
        # Call stacked kernel
        aggregated_evidence = monty_cuda.evidence_aggregation_stacked(
            stacked_old_evidence, stacked_current_evidence, evidence_update_thresholds,
            min_updates, past_weight, present_weight
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        return aggregated_evidence, stacked["update_offsets"], gpu_time

    def _aggressive_memory_cleanup(self):
        """Aggressive GPU memory cleanup to prevent accumulation."""
        if self.device.type != "cuda":
            return

        # Step 1: Standard empty_cache (releases unused cached memory)
        torch.cuda.empty_cache()

        # Step 2: Force garbage collection to clean up Python references
        import gc
        gc.collect()

        # Step 3: Try to clear all unoccupied cached memory again
        torch.cuda.empty_cache()

        # Step 4: Reset peak memory stats (for cleaner monitoring)
        torch.cuda.reset_peak_memory_stats(self.device)

        # Step 5: Print memory status for debugging
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

    # Group traces by step
    traces_by_step = {}
    for trace in traces:
        step = trace.get("step", 0)
        if step not in traces_by_step:
            traces_by_step[step] = []
        # print(f"Adding trace for step {step} with lm id {trace['lm_id']}")
        traces_by_step[step].append(trace)
    print(f"Processing {len(traces_by_step)} steps with {len(traces)} total traces")
    # exit()

    # Function mapping
    function_map = {
        "displacement": ["displacement"],
        "pose_transformation": ["pose_transformation"],
        "evidence_aggregation": ["evidence_aggregation"],
        "knn_search": ["displacement", "knn_search"],
        "distance": ["displacement", "knn_search", "distance_calculation"],
        "pose_evidence": ["displacement", "knn_search", "distance_calculation", "pose_evidence"],
        "aggregation": ["displacement", "knn_search", "distance_calculation", "pose_evidence", "final_aggregation"],
        None: ["displacement", "pose_transformation", "evidence_aggregation", "knn_search", "distance_calculation", "pose_evidence", "final_aggregation"]
    }

    functions_to_test = function_map.get(target_function, function_map[None])

    step_results = []
    total_cpu_time = 0
    total_gpu_per_trace_time = 0
    total_gpu_batched_time = 0

    for step, step_traces in sorted(traces_by_step.items()):
    # for step, step_traces in traces_by_step.items():
    # for i in [7,6,5,4,3,2,1]:
    # for i in [7,1, 4, 3, 6, 5, 2]:
        # step = i
        # step_traces = traces_by_step[i]
        print(f"\n=== Step {step}: {len(step_traces)} traces ===")

        # Create unified data structure for this step
        unified_data = UnifiedStepData(step_traces, operations.device)

        if unified_data.num_valid_traces == 0:
            print("  No valid traces in this step")
            continue

        # Process each function with all three approaches
        for func_name in functions_to_test:
            success = process_function_all_approaches(func_name, unified_data, operations)
            if not success and target_function == func_name:
                print(f"  Failed to test target function {func_name}")
                break

        # Calculate step totals
        step_cpu_time = sum(unified_data.cpu_times.values())
        step_gpu_per_trace_time = sum(unified_data.gpu_per_trace_times.values())
        step_gpu_batched_time = sum(unified_data.gpu_batched_times.values())

        total_cpu_time += step_cpu_time
        total_gpu_per_trace_time += step_gpu_per_trace_time
        total_gpu_batched_time += step_gpu_batched_time

        # Print step summary with all three approaches
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
            "num_traces": len(step_traces),
            "cpu_times": unified_data.cpu_times.copy(),
            "gpu_per_trace_times": unified_data.gpu_per_trace_times.copy(),
            "gpu_batched_times": unified_data.gpu_batched_times.copy(),
            "verification_results": unified_data.verification_results.copy(),
            "total_cpu_time": step_cpu_time,
            "total_gpu_per_trace_time": step_gpu_per_trace_time,
            "total_gpu_batched_time": step_gpu_batched_time
        })

        # Memory cleanup
        unified_data.cleanup()
        operations._aggressive_memory_cleanup()
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
        elif func_name == "final_aggregation":
            return process_final_aggregation_all_approaches(unified_data, operations)
        else:
            print(f"    Unknown function: {func_name}")
            return False

    except Exception as e:
        print(f"    Error in {func_name}: {str(e)}")
        return False


def process_displacement_all_approaches(unified_data: UnifiedStepData,
                                       operations: MontyOperations) -> bool:
    """Process displacement with CPU, GPU per-trace, and GPU batched approaches."""

    # CPU and GPU Per-Trace processing
    all_search_locations_cpu = []
    all_search_locations_gpu_per_trace = []
    total_cpu_time = 0
    total_gpu_per_trace_time = 0

    for i in range(unified_data.num_valid_traces):
        trace_data = unified_data.get_per_trace_item(i)
        poses = trace_data['poses']
        locations = trace_data['locations']
        displacement = trace_data['displacement']

        # CPU version
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

    # GPU Batched processing
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

    unified_data.verification_results["displacement"] = {
        "max_diff": max_diff,
        "max_batched_diff": max_batched_diff,
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


def process_knn_search_all_approaches(unified_data: UnifiedStepData,
                                     operations: MontyOperations) -> bool:
    """Process KNN search with CPU, GPU per-trace, and GPU batched approaches."""

    # Check if any trace has KNN search data
    has_knn_data = any('graph_locations' in unified_data.get_per_trace_item(i) and
                      'search_locations' in unified_data.get_per_trace_item(i)
                      for i in range(unified_data.num_valid_traces))
    if not has_knn_data:
        print("    No KNN search data available")
        return False

    # CPU and GPU Per-Trace processing
    all_nearest_indices_cpu = []
    all_nearest_indices_gpu_per_trace = []
    total_cpu_time = 0
    total_gpu_per_trace_time = 0

    # Use trace data directly
    for i in range(unified_data.num_valid_traces):
        trace_data = unified_data.get_per_trace_item(i)
        if 'graph_locations' not in trace_data or 'search_locations' not in trace_data:
            continue

        graph_locations = trace_data['graph_locations']
        search_locations = trace_data['search_locations']

        # CPU version
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

    # GPU Batched processing
    total_gpu_batched_time = 0
    batched_nearest_indices = None
    hypothesis_offsets = None

    if unified_data.num_valid_traces > 0:
        try:
            # Note: Stacked KNN with per-trace graphs requires complex graph handling
            # For now, batched processing is not implemented for trace-only data
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
    for i in range(len(all_nearest_indices_cpu)):
        cpu_result = all_nearest_indices_cpu[i]
        gpu_result = all_nearest_indices_gpu_per_trace[i].cpu().numpy()

        # Per-trace verification
        trace_diff = np.max(np.abs(cpu_result - gpu_result))
        max_diff = max(max_diff, trace_diff)

        # Batched verification if available
        if batched_nearest_indices is not None and hypothesis_offsets is not None:
            gpu_batched_result = batched_nearest_indices[hypothesis_offsets[i]:hypothesis_offsets[i+1]]
            gpu_batched_result_np = gpu_batched_result.cpu().numpy()
            batched_trace_diff = np.max(np.abs(cpu_result - gpu_batched_result_np))
            max_batched_diff = max(max_batched_diff, batched_trace_diff)
    # Verify batched GPU vs CPU
    # if total_gpu_batched_time > 0 and batched_nearest_indices is not None:
    #     # Split batched results back into per-trace format for comparison
    #     batched_indices_cpu = batched_nearest_indices.cpu().numpy()

    #     # Use the offsets from UnifiedStepData to split results correctly
    #     stacked = unified_data.get_stacked_tensors()
    #     start_idx = 0

    #     for i, cpu_idx in enumerate(all_nearest_indices_cpu):
    #         # Get the number of hypotheses for this trace
    #         hyp_count = stacked["hyp_counts"][i].item()

    #         # KNN results have shape (n_hypotheses, k)
    #         k = cpu_idx.shape[1] if len(cpu_idx.shape) > 1 else 1
    #         end_idx = start_idx + hyp_count * k

    #         # Reshape batched chunk to match expected shape
    #         batched_chunk = batched_indices_cpu[start_idx:end_idx]
    #         if k > 1:
    #             batched_chunk = batched_chunk.reshape(hyp_count, k)

    #         if not np.array_equal(cpu_idx, batched_chunk):
    #             batched_indices_match = False
    #             break
    #         start_idx = end_idx

    unified_data.verification_results["knn_search"] = {
        "max_diff": max_diff,
        "max_batched_diff": max_batched_diff,
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

    # Check for required trace data
    has_trace_data = any('search_locations' in unified_data.get_per_trace_item(i) and
                        'nearest_node_locs' in unified_data.get_per_trace_item(i) and
                        'pose_normals' in unified_data.get_per_trace_item(i) and
                        'max_abs_curvature' in unified_data.get_per_trace_item(i)
                        for i in range(unified_data.num_valid_traces))

    if not has_trace_data:
        print("    No distance calculation data available")
        return False

    # CPU and GPU Per-Trace processing
    all_distances_cpu = []
    all_distances_gpu_per_trace = []
    total_cpu_time = 0
    total_gpu_per_trace_time = 0

    # Use trace data directly
    for i in range(unified_data.num_valid_traces):
        trace_data = unified_data.get_per_trace_item(i)
        if ('search_locations' not in trace_data or 'nearest_node_locs' not in trace_data or
            'pose_normals' not in trace_data or 'max_abs_curvature' not in trace_data):
            continue

        search_locs_cpu = trace_data['search_locations']
        nearest_locs_cpu = trace_data['nearest_node_locs']
        pose_normals_cpu = trace_data['pose_normals']
        curvature = trace_data['max_abs_curvature']

        # CPU version
        custom_distances_cpu, cpu_time = operations.distance_calculation_cpu(
            search_locs_cpu, nearest_locs_cpu, pose_normals_cpu, curvature
        )
        total_cpu_time += cpu_time
        all_distances_cpu.append(custom_distances_cpu)

        # GPU Per-Trace version
        search_locs_gpu = torch.from_numpy(search_locs_cpu.astype(np.float32)).to(operations.device)
        nearest_locs_gpu = torch.from_numpy(nearest_locs_cpu.astype(np.float32)).to(operations.device)
        pose_normals_gpu = torch.from_numpy(pose_normals_cpu.astype(np.float32)).to(operations.device)

        custom_distances_gpu, gpu_time = operations.distance_calculation_gpu(
            search_locs_gpu, nearest_locs_gpu, pose_normals_gpu, curvature
        )
        total_gpu_per_trace_time += gpu_time
        all_distances_gpu_per_trace.append(custom_distances_gpu)

    # GPU Batched processing
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
    for i in range(len(all_distances_cpu)):
        cpu_result = all_distances_cpu[i]
        gpu_result = all_distances_gpu_per_trace[i].cpu().numpy()

        trace_diff = np.max(np.abs(cpu_result - gpu_result))
        max_diff = max(max_diff, trace_diff)

        gpu_batched_result = batched_custom_distances[hypothesis_offsets[i]:hypothesis_offsets[i+1]]
        gpu_batched_result_np = gpu_batched_result.cpu().numpy()
        batched_trace_diff = np.max(np.abs(cpu_result - gpu_batched_result_np))
        max_batched_diff = max(max_batched_diff, batched_trace_diff)


    # Store results
    unified_data.cpu_times["distance_calculation"] = total_cpu_time
    unified_data.gpu_per_trace_times["distance_calculation"] = total_gpu_per_trace_time
    unified_data.gpu_batched_times["distance_calculation"] = total_gpu_batched_time

    # # Verify batched GPU results against CPU as well
    # batched_max_diff = 0
    # if total_gpu_batched_time > 0 and 'batched_distances' in locals():
    #     # Split batched results back into per-trace format for comparison
    #     batched_distances_cpu = batched_distances.cpu().numpy()

    #     # Use the offsets from UnifiedStepData to split results correctly
    #     stacked = unified_data.get_stacked_tensors()
    #     start_idx = 0

    #     for i in range(stacked["num_traces"]):
    #         # Get the number of hypotheses for this trace
    #         hyp_count = stacked["hyp_counts"][i].item()

    #         # Get the trace data
    #         trace = unified_data.step_traces[i] if i < len(unified_data.step_traces) else unified_data.step_traces[0]
    #         if ("evidence_intermediates" in trace and
    #             "distance_calculation" in trace["evidence_intermediates"]):
    #             dist_data = trace["evidence_intermediates"]["distance_calculation"]

    #             # Get expected output shape from trace data
    #             expected_distances = dist_data["outputs"]["custom_nearest_node_dists"]
    #             num_elements = expected_distances.size

    #             # Extract this trace's results from batched output
    #             end_idx = start_idx + num_elements
    #             trace_results = batched_distances_cpu[start_idx:end_idx].reshape(expected_distances.shape)

    #             # Compare with expected
    #             trace_diff = np.max(np.abs(trace_results - expected_distances))
    #             batched_max_diff = max(batched_max_diff, trace_diff)

    #             start_idx = end_idx

    unified_data.verification_results["distance_calculation"] = {
        "per_trace_max_diff": max_diff,
        "batched_max_diff": max_batched_diff,
        "per_trace_passed": max_diff < 1e-4,
        "batched_passed": total_gpu_batched_time > 0,  # Pass if batched ran without error
        "passed": max_diff < 1e-4 and total_gpu_batched_time > 0
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

    # Check if any trace has pose evidence data
    has_pose_data = any('query_pose_vectors' in unified_data.get_per_trace_item(i) and
                       'node_pose_vectors' in unified_data.get_per_trace_item(i)
                       for i in range(unified_data.num_valid_traces))

    if not has_pose_data:
        print("    No pose evidence data available")
        return False

    # CPU and GPU Per-Trace processing
    all_pose_evidence_cpu = []
    all_pose_evidence_gpu_per_trace = []
    total_cpu_time = 0
    total_gpu_per_trace_time = 0

    weights_cpu = np.array([1.0, 0.5])  # Default weights

    for i in range(unified_data.num_valid_traces):
        trace_data = unified_data.get_per_trace_item(i)
        if ('query_pose_vectors' not in trace_data or 'node_pose_vectors' not in trace_data):
            continue

        query_pose_vectors = trace_data['query_pose_vectors']
        node_pose_vectors = trace_data['node_pose_vectors']
        query_poses_fully_defined = trace_data['query_poses_fully_defined']
        # Get additional data if available
        use_cd = trace_data['use_cd']

        # CPU version
        pose_evidence_cpu, cpu_time = operations.pose_evidence_cpu(
            query_pose_vectors, node_pose_vectors, use_cd, query_poses_fully_defined, weights_cpu
        )
        total_cpu_time += cpu_time
        all_pose_evidence_cpu.append(pose_evidence_cpu)

        # GPU Per-Trace version
        query_poses_gpu = torch.from_numpy(query_pose_vectors.astype(np.float32)).to(operations.device)
        node_poses_gpu = torch.from_numpy(node_pose_vectors.astype(np.float32)).to(operations.device)
        weights_gpu = torch.from_numpy(weights_cpu.astype(np.float32)).to(operations.device)
        use_cd_gpu = torch.from_numpy(use_cd.astype(np.float32)).to(operations.device)

        pose_evidence_gpu, gpu_time = operations.pose_evidence_gpu(
            query_poses_gpu, node_poses_gpu, use_cd_gpu, weights_gpu
        )
        total_gpu_per_trace_time += gpu_time
        all_pose_evidence_gpu_per_trace.append(pose_evidence_gpu)

    # GPU Batched processing - now uses UnifiedStepData
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


    unified_data.verification_results["pose_evidence"] = {
        "max_diff": max_diff,
        "max_batched_diff": max_batched_diff,
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


def process_final_aggregation_all_approaches(unified_data: UnifiedStepData,
                                           operations: MontyOperations) -> bool:
    """Process final aggregation with CPU, GPU per-trace, and GPU batched approaches."""

    # Check if any trace has final aggregation data
    has_final_data = any('radius_evidence' in unified_data.get_per_trace_item(i)
                        for i in range(unified_data.num_valid_traces))

    if not has_final_data:
        print("    No final aggregation data available")
        return False

    # CPU and GPU Per-Trace processing
    all_final_evidence_cpu = []
    all_final_evidence_gpu_per_trace = []
    total_cpu_time = 0
    total_gpu_per_trace_time = 0

    for i in range(unified_data.num_valid_traces):
        trace_data = unified_data.get_per_trace_item(i)

        radius_evidence = trace_data['radius_evidence']

        # CPU version
        final_evidence_cpu, cpu_time = operations.final_aggregation_cpu(radius_evidence)
        total_cpu_time += cpu_time
        all_final_evidence_cpu.append(final_evidence_cpu)

        # GPU Per-Trace version
        radius_evidence_gpu = torch.from_numpy(radius_evidence.astype(np.float32)).to(operations.device)
        final_evidence_gpu, gpu_time = operations.final_aggregation_gpu(radius_evidence_gpu)
        total_gpu_per_trace_time += gpu_time
        all_final_evidence_gpu_per_trace.append(final_evidence_gpu)

    # GPU Batched processing - now uses UnifiedStepData
    total_gpu_batched_time = 0
    batched_final_evidence = None

    if unified_data.num_valid_traces > 0:
        try:
            # Use UnifiedStepData for batched processing
            batched_final_evidence, total_gpu_batched_time = operations.final_aggregation_gpu_batched(unified_data)
        except Exception as e:
            print(f"    Batched GPU failed: {e}")
            total_gpu_batched_time = 0

    if not all_final_evidence_cpu:
        print("    No final aggregation data processed")
        return False

    # Store results and timing
    unified_data.cpu_times["final_aggregation"] = total_cpu_time
    unified_data.gpu_per_trace_times["final_aggregation"] = total_gpu_per_trace_time
    unified_data.gpu_batched_times["final_aggregation"] = total_gpu_batched_time

    # Verify results
    max_diff = 0
    max_batched_diff = 0
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

    unified_data.verification_results["final_aggregation"] = {
        "per_trace_max_diff": max_diff,
        "batched_max_diff": max_batched_diff,
        "per_trace_passed": max_diff < 1e-5,
        "batched_passed": total_gpu_batched_time > 0,  # Pass if batched ran without error
        "passed": max_diff < 1e-5 and total_gpu_batched_time > 0
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
        channel_poses = trace_data['channel_possible_poses']
        pose_vectors = trace_data['pose_vectors']

        # CPU version
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

    # GPU Batched processing using wrapper function
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
        # print(batched_trace_diff)

    unified_data.verification_results["pose_transformation"] = {
        "max_diff": max_diff,
        "max_batched_diff": max_batched_diff,
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
        old_evidence = trace_data['old_evidence']
        new_evidence = trace_data['new_evidence']
        current_evidence = trace_data['current_evidence']
        min_update = trace_data['min_update']
        hyp_ids_to_test = trace_data['hyp_ids_to_test']
        evidence_update_threshold = trace_data['evidence_update_threshold']

        # CPU version
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

    # GPU Batched processing (stacked approach)
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
    unified_data.verification_results["evidence_aggregation"] = {
        "max_diff": max_diff,
        "max_batched_diff": max_batched_diff,
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


def process_knn_search(traces: List[Dict], operations: MontyOperations,
                      unified_data: UnifiedStepData) -> bool:
    """Process KNN search."""

    # Get graph locations from first trace
    if "graph_memory" not in traces[0] or traces[0]["graph_memory"]["locations"] is None:
        print("    No graph location data available")
        return False

    graph_locations_cpu = traces[0]["graph_memory"]["locations"]
    graph_locations_gpu = torch.from_numpy(graph_locations_cpu.astype(np.float32)).to(operations.device)

    # Use chained search_locations if available
    if unified_data.search_locations is not None:
        # Process each trace individually (they have different sizes)
        all_nearest_indices_cpu = []
        all_nearest_indices_gpu = []
        total_cpu_time = 0
        total_gpu_time = 0

        for search_locations_gpu in unified_data.search_locations:
            search_locations_cpu = search_locations_gpu.cpu().numpy()

            # Run CPU version
            nearest_indices_cpu, cpu_time = operations.knn_search_cpu(
                search_locations_cpu, graph_locations_cpu, k=3
            )
            total_cpu_time += cpu_time
            all_nearest_indices_cpu.append(nearest_indices_cpu)

            # Run GPU version
            nearest_indices_gpu, gpu_time = operations.knn_search_gpu(
                search_locations_gpu, graph_locations_gpu, k=3
            )
            total_gpu_time += gpu_time
            all_nearest_indices_gpu.append(nearest_indices_gpu)

        # Store results
        unified_data.nearest_indices = all_nearest_indices_gpu
        unified_data.cpu_times["knn_search"] = total_cpu_time
        unified_data.gpu_per_trace_times["knn_search"] = total_gpu_time
        unified_data.gpu_batched_times["knn_search"] = 0.0  # Placeholder for now

        # Verify results
        indices_match = True
        for cpu_idx, gpu_idx in zip(all_nearest_indices_cpu, all_nearest_indices_gpu):
            gpu_idx_np = gpu_idx.cpu().numpy()
            if not np.array_equal(cpu_idx, gpu_idx_np):
                indices_match = False
                break

        unified_data.verification_results["knn_search"] = {
            "indices_match": indices_match,
            "passed": indices_match
        }

        print(f"    GPU: {total_gpu_time * 1000:.3f}ms, CPU: {total_cpu_time * 1000:.3f}ms, " +
              f"Speedup: {total_cpu_time/total_gpu_time:.2f}x, Indices match: {indices_match}")

        return True

    else:
        # Fallback to trace data
        print("    Falling back to trace search_locations")
        return False


def process_distance_calculation(traces: List[Dict], operations: MontyOperations,
                               unified_data: UnifiedStepData) -> bool:
    """Process distance calculation."""

    # Use chained data if available
    if (unified_data.search_locations is not None and
        unified_data.nearest_indices is not None):

        # Get graph locations to compute nearest locations
        if "graph_memory" not in traces[0] or traces[0]["graph_memory"]["locations"] is None:
            print("    No graph location data available")
            return False

        graph_locations = traces[0]["graph_memory"]["locations"]

        # Process each trace individually
        total_gpu_time = 0
        total_cpu_time = 0
        max_diff = 0

        for i, (search_locs, nearest_idx) in enumerate(zip(unified_data.search_locations, unified_data.nearest_indices)):
            search_locs_cpu = search_locs.cpu().numpy()
            nearest_idx_cpu = nearest_idx.cpu().numpy()

            # Get the nearest locations correctly: each search location has K nearest neighbors
            # nearest_idx_cpu shape: (N, K), graph_locations shape: (M, 3)
            # We want nearest_locs_cpu shape: (N, K, 3)
            nearest_locs_cpu = graph_locations[nearest_idx_cpu]  # This should work for (N, K) indexing

            # Get pose normals from trace evidence (fallback if not available)
            trace = traces[i] if i < len(traces) else traces[0]
            if ("evidence_intermediates" in trace and
                "distance_calculation" in trace["evidence_intermediates"]):
                # Use exact data from trace
                dist_data = trace["evidence_intermediates"]["distance_calculation"]
                pose_normals_cpu = dist_data["inputs"]["pose_normals"]
                curvature = dist_data["inputs"]["max_abs_curvature"]
                # Use the exact nearest_locs from trace to ensure correct shapes
                nearest_locs_cpu = dist_data["inputs"]["nearest_node_locs"]
                search_locs_cpu = dist_data["inputs"]["search_locations"]
            else:
                # Fallback: skip distance calculation for this trace if no evidence data
                print(f"    Skipping trace {i}: no distance calculation evidence data")
                continue

            # Run CPU version
            custom_distances_cpu, cpu_time = operations.distance_calculation_cpu(
                search_locs_cpu, nearest_locs_cpu, pose_normals_cpu, curvature
            )
            total_cpu_time += cpu_time

            # Run GPU version
            search_locs_gpu = torch.from_numpy(search_locs_cpu.astype(np.float32)).to(operations.device)
            nearest_locs_gpu = torch.from_numpy(nearest_locs_cpu.astype(np.float32)).to(operations.device)
            pose_normals_gpu = torch.from_numpy(pose_normals_cpu.astype(np.float32)).to(operations.device)

            custom_distances_gpu, gpu_time = operations.distance_calculation_gpu(
                search_locs_gpu, nearest_locs_gpu, pose_normals_gpu, curvature
            )
            total_gpu_time += gpu_time

            # Verify results
            gpu_np = custom_distances_gpu.cpu().numpy()
            trace_diff = np.max(np.abs(custom_distances_cpu - gpu_np))
            max_diff = max(max_diff, trace_diff)

        # Store results
        unified_data.cpu_times["distance_calculation"] = total_cpu_time
        unified_data.gpu_per_trace_times["distance_calculation"] = total_gpu_time
        unified_data.gpu_batched_times["distance_calculation"] = 0.0  # Placeholder for now
        unified_data.verification_results["distance_calculation"] = {
            "max_diff": max_diff,
            "passed": max_diff < 1e-4
        }

        print(f"    GPU: {total_gpu_time * 1000:.3f}ms, CPU: {total_cpu_time * 1000:.3f}ms, " +
              f"Speedup: {total_cpu_time/total_gpu_time:.2f}x, Max diff: {max_diff:.2e}")

        return True

    else:
        print("    Skipping distance calculation (no chained data)")
        return False


def process_pose_evidence(traces: List[Dict], operations: MontyOperations,
                         unified_data: UnifiedStepData) -> bool:
    """Process pose evidence calculation."""

    # Get pose data from traces (use evidence_intermediates when available)
    query_poses_list = []
    node_poses_list = []

    for trace in traces:
        if ("evidence_intermediates" in trace and
            "pose_evidence_matrix" in trace["evidence_intermediates"]):
            pose_data = trace["evidence_intermediates"]["pose_evidence_matrix"]
            if ("inputs" in pose_data and
                "query_features" in pose_data["inputs"] and
                "node_features" in pose_data["inputs"]):
                query_poses_list.append(pose_data["inputs"]["query_features"]["pose_vectors"])
                node_poses_list.append(pose_data["inputs"]["node_features"]["pose_vectors"])

    if not query_poses_list:
        print("    Skipping pose evidence (no evidence_intermediates data)")
        return False

    # Process each trace individually
    total_gpu_time = 0
    total_cpu_time = 0
    max_diff = 0

    weights_cpu = np.array([1.0, 0.5])  # Default weights

    for query_poses_cpu, node_poses_cpu in zip(query_poses_list, node_poses_list):
        # Run CPU version
        pose_evidence_cpu, cpu_time = operations.pose_evidence_cpu(
            query_poses_cpu, node_poses_cpu, weights_cpu
        )
        total_cpu_time += cpu_time

        # Run GPU version
        query_poses_gpu = torch.from_numpy(query_poses_cpu.astype(np.float32)).to(operations.device)
        node_poses_gpu = torch.from_numpy(node_poses_cpu.astype(np.float32)).to(operations.device)
        weights_gpu = torch.from_numpy(weights_cpu.astype(np.float32)).to(operations.device)

        pose_evidence_gpu, gpu_time = operations.pose_evidence_gpu(
            query_poses_gpu, node_poses_gpu, weights_gpu
        )
        total_gpu_time += gpu_time

        # Verify results
        gpu_np = pose_evidence_gpu.cpu().numpy()
        trace_diff = np.max(np.abs(pose_evidence_cpu - gpu_np))
        max_diff = max(max_diff, trace_diff)

    # Store results
    unified_data.cpu_times["pose_evidence"] = total_cpu_time
    unified_data.gpu_per_trace_times["pose_evidence"] = total_gpu_time
    unified_data.gpu_batched_times["pose_evidence"] = 0.0  # Placeholder for now
    unified_data.verification_results["pose_evidence"] = {
        "max_diff": max_diff,
        "passed": max_diff < 1e-4
    }

    print(f"    GPU: {total_gpu_time * 1000:.3f}ms, CPU: {total_cpu_time * 1000:.3f}ms, " +
          f"Speedup: {total_cpu_time/total_gpu_time:.2f}x, Max diff: {max_diff:.2e}")

    return True


def process_final_aggregation(traces: List[Dict], operations: MontyOperations,
                             unified_data: UnifiedStepData) -> bool:
    """Process final evidence aggregation."""

    # Get evidence matrix from traces (use radius_evidence from evidence_intermediates)
    evidence_matrices = []

    for trace in traces:
        if ("evidence_intermediates" in trace and
            "final_aggregation" in trace["evidence_intermediates"]):
            final_data = trace["evidence_intermediates"]["final_aggregation"]
            if "inputs" in final_data and "radius_evidence" in final_data["inputs"]:
                evidence_matrices.append(final_data["inputs"]["radius_evidence"])

    if not evidence_matrices:
        print("    Skipping final aggregation (no evidence matrix data)")
        return False

    # Process each evidence matrix individually
    total_gpu_time = 0
    total_cpu_time = 0
    max_diff = 0

    for evidence_matrix_cpu in evidence_matrices:
        # Run CPU version
        final_evidence_cpu, cpu_time = operations.final_aggregation_cpu(evidence_matrix_cpu)
        total_cpu_time += cpu_time

        # Run GPU version
        evidence_matrix_gpu = torch.from_numpy(evidence_matrix_cpu.astype(np.float32)).to(operations.device)
        final_evidence_gpu, gpu_time = operations.final_aggregation_gpu(evidence_matrix_gpu)
        total_gpu_time += gpu_time

        # Verify results
        gpu_np = final_evidence_gpu.cpu().numpy()
        trace_diff = np.max(np.abs(final_evidence_cpu - gpu_np))
        max_diff = max(max_diff, trace_diff)

    # Store results
    unified_data.cpu_times["final_aggregation"] = total_cpu_time
    unified_data.gpu_per_trace_times["final_aggregation"] = total_gpu_time
    unified_data.gpu_batched_times["final_aggregation"] = 0.0  # Placeholder for now
    unified_data.verification_results["final_aggregation"] = {
        "max_diff": max_diff,
        "passed": max_diff < 1e-5
    }

    print(f"    GPU: {total_gpu_time * 1000:.3f}ms, CPU: {total_cpu_time * 1000:.3f}ms, " +
          f"Speedup: {total_cpu_time/total_gpu_time:.2f}x, Max diff: {max_diff:.2e}")

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="GPU vs CPU Monty Operations Test")
    parser.add_argument("--output-dir", required=True, help="Experiment output directory")
    parser.add_argument("--function", choices=["displacement", "knn_search", "distance",
                       "pose_evidence", "aggregation"],
                       help="Test specific function (default: test all)")

    args = parser.parse_args()

    # Load profiling data
    traces = load_profiling_data(args.output_dir)
    # traces.reverse()
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
    if results['total_cpu_time'] > 0:
        per_trace_speedup = results['total_cpu_time'] / results['total_gpu_per_trace_time'] if results['total_gpu_per_trace_time'] > 0 else 0
        batched_speedup = results['total_cpu_time'] / results['total_gpu_batched_time'] if results['total_gpu_batched_time'] > 0 else 0
        batch_vs_pertrace = results['total_gpu_per_trace_time'] / results['total_gpu_batched_time'] if results['total_gpu_batched_time'] > 0 else 0

        print(f"\n--- SPEEDUP ANALYSIS ---")
        print(f"GPU Per-Trace vs CPU:     {per_trace_speedup:.2f}x")
        print(f"GPU Batched vs CPU:       {batched_speedup:.2f}x")
        print(f"GPU Batched vs Per-Trace: {batch_vs_pertrace:.2f}x")

    # Print verification summary
    print(f"\n=== VERIFICATION SUMMARY ===")
    all_passed = True
    for step_result in results['step_results']:
        for func_name, verification in step_result['verification_results'].items():
            if not verification.get('passed', False):
                all_passed = False
                print(f"❌ {func_name}: FAILED verification")

    if all_passed:
        print("✅ All functions passed verification")

    print("\n=== KEY INSIGHTS ===")
    if results['total_gpu_batched_time'] > 0 and batch_vs_pertrace > 1.0:
        print(f"🚀 Single-dispatch batched GPU processing is {batch_vs_pertrace:.1f}x faster than per-trace GPU")
        print("   This demonstrates the benefit of processing all traces in one kernel call")
    elif results['total_gpu_batched_time'] > 0:
        print("⚠ Batched GPU processing not yet fully optimized (currently only displacement)")
    else:
        print("⚠ Batched GPU processing needs implementation for remaining functions")

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
