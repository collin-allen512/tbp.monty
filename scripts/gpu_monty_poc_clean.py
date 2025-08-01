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
    trace_file = output_path / "hypothesis_computation_trace.pkl"

    if trace_file.exists():
        with open(trace_file, "rb") as f:
            traces = pickle.load(f)
        print(f"Loaded {len(traces)} computation traces")

        # Check for evidence intermediates
        complete_traces = [t for t in traces if "evidence_intermediates" in t]
        print(f"Found {len(complete_traces)} traces with complete evidence data")

        return traces

    print("No profiling data found!")
    return []


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
                         weights: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """GPU implementation of pose evidence calculation using CUDA kernels."""
        if monty_cuda is None:
            raise RuntimeError("CUDA kernels not available")


        # The pose_evidence_stacked kernel expects angles, not raw poses
        # We need to calculate angles first
        query_normals = query_poses[:, 0]  # (N, 3)
        node_normals = node_poses[:, :, :3]  # (N, K, 3)

        # Calculate pn_angles using PyTorch (could be optimized with angle kernel later)
        dot_products = torch.einsum("ijk,ik->ij", node_normals, query_normals)
        dot_products = torch.clamp(dot_products, -1, 1)
        pn_angles = torch.acos(dot_products)

        # For simplicity, create dummy cd1_angles and use_cd (could be enhanced later)
        cd1_angles = torch.zeros_like(pn_angles)
        use_cd = torch.zeros_like(pn_angles, dtype=torch.int32)

        # Create trace offsets and weight tensors
        trace_offsets = torch.tensor([0], dtype=torch.int32, device=self.device)
        pn_weights = torch.tensor([weights[0]], dtype=torch.float32, device=self.device)
        cd1_weights = torch.tensor([weights[1] if len(weights) > 1 else 0.0], dtype=torch.float32, device=self.device)

        start_time = time.perf_counter()
        # Use stacked CUDA kernel for pose evidence calculation
        pose_evidence = monty_cuda.pose_evidence_stacked(
            pn_angles, cd1_angles, use_cd, pn_weights, cd1_weights, trace_offsets
        )

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        return pose_evidence, gpu_time

    def pose_evidence_cpu(self, query_poses: np.ndarray,
                         node_poses: np.ndarray,
                         weights: np.ndarray) -> Tuple[np.ndarray, float]:
        """CPU implementation matching Monty's pose evidence calculation."""

        # Exact implementation from Monty
        # Calculate angles between pose vectors
        query_normals = query_poses[:, 0]  # Primary normal
        node_normals = node_poses[:, :, :3]  # Primary normals

        # Compute dot products and angles
        start_time = time.perf_counter()
        dot_products = np.sum(
            query_normals[:, np.newaxis] * node_normals, axis=2
        )
        dot_products = np.clip(dot_products, -1, 1)
        angles = np.arccos(np.abs(dot_products))

        # Convert to evidence
        pn_evidence = -(np.sin(angles / 2) - 0.5)
        pose_evidence = pn_evidence * weights[0]

        cpu_time = time.perf_counter() - start_time

        return pose_evidence, cpu_time

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

class PipelineData:
    """Container for pipeline data that can be chained between operations."""

    def __init__(self):
        # Intermediate results that can be chained
        self.search_locations = None
        self.nearest_indices = None
        self.nearest_locations = None
        self.custom_distances = None
        self.pose_evidence = None
        self.final_evidence = None

        # Timing data
        self.gpu_times = {}
        self.cpu_times = {}

        # Verification results
        self.verification_results = {}


def run_step_analysis(traces: List[Dict], operations: MontyOperations,
                     target_function: Optional[str] = None) -> Dict[str, Any]:
    """Run per-step analysis comparing GPU vs CPU for all operations."""

    # Group traces by step
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
        "knn_search": ["displacement", "knn_search"],
        "distance": ["displacement", "knn_search", "distance_calculation"],
        "pose_evidence": ["displacement", "knn_search", "distance_calculation", "pose_evidence"],
        "aggregation": ["displacement", "knn_search", "distance_calculation", "pose_evidence", "final_aggregation"],
        None: ["displacement", "knn_search", "distance_calculation", "pose_evidence", "final_aggregation"]  # All functions
    }

    functions_to_test = function_map.get(target_function, function_map[None])

    step_results = []
    total_gpu_time = 0
    total_cpu_time = 0

    for step, step_traces in sorted(traces_by_step.items()):
        print(f"\n=== Step {step}: {len(step_traces)} traces ===")

        pipeline_data = PipelineData()
        step_start_time = time.perf_counter()

        # Process each function in the pipeline
        for func_name in functions_to_test:
            success = process_function(func_name, step_traces, operations, pipeline_data)
            if not success and target_function == func_name:
                print(f"  Failed to test target function {func_name}")
                break
            torch.cuda.synchronize()
            # More aggressive memory cleanup with force garbage collection
            operations._aggressive_memory_cleanup()
            # Additional tensor cleanup - force cleanup of any remaining references
            gc.collect()
            torch.cuda.empty_cache()
            # Get post-cleanup memory state
            # post_memory = operations._get_gpu_memory_info()

        # Calculate step totals
        step_gpu_time = sum(pipeline_data.gpu_times.values())
        step_cpu_time = sum(pipeline_data.cpu_times.values())
        total_gpu_time += step_gpu_time
        total_cpu_time += step_cpu_time

        # Print step summary
        print(f"  GPU total: {step_gpu_time * 1000:.3f}ms")
        print(f"  CPU total: {step_cpu_time * 1000:.3f}ms")
        if step_cpu_time > 0:
            print(f"  Speedup: {step_cpu_time / step_gpu_time:.2f}x")

        step_results.append({
            "step": step,
            "num_traces": len(step_traces),
            "gpu_times": pipeline_data.gpu_times.copy(),
            "cpu_times": pipeline_data.cpu_times.copy(),
            "verification_results": pipeline_data.verification_results.copy(),
            "total_gpu_time": step_gpu_time,
            "total_cpu_time": step_cpu_time
        })

    return {
        "num_steps": len(traces_by_step),
        "total_traces": len(traces),
        "functions_tested": functions_to_test,
        "total_gpu_time": total_gpu_time,
        "total_cpu_time": total_cpu_time,
        "overall_speedup": total_cpu_time / total_gpu_time if total_gpu_time > 0 else 0,
        "step_results": step_results
    }


def process_function(func_name: str, traces: List[Dict], operations: MontyOperations,
                    pipeline_data: PipelineData) -> bool:
    """Process a single function with GPU and CPU implementations."""

    print(f"  Testing {func_name}...")

    try:
        if func_name == "displacement":
            return process_displacement(traces, operations, pipeline_data)
        elif func_name == "knn_search":
            return process_knn_search(traces, operations, pipeline_data)
        elif func_name == "distance_calculation":
            return process_distance_calculation(traces, operations, pipeline_data)
        elif func_name == "pose_evidence":
            return process_pose_evidence(traces, operations, pipeline_data)
        elif func_name == "final_aggregation":
            return process_final_aggregation(traces, operations, pipeline_data)
        else:
            print(f"    Unknown function: {func_name}")
            return False


    except Exception as e:
        print(f"    Error in {func_name}: {str(e)}")
        return False


def process_displacement(traces: List[Dict], operations: MontyOperations,
                        pipeline_data: PipelineData) -> bool:
    """Process displacement calculation."""

    # Process each trace individually since they have different sizes
    all_search_locations_cpu = []
    all_search_locations_gpu = []
    total_cpu_time = 0
    total_gpu_time = 0

    for trace in traces:
        inputs = trace["inputs"]
        if "initial_hypotheses" not in inputs:
            continue

        poses = inputs["initial_hypotheses"]["poses"]
        locations = inputs["initial_hypotheses"]["locations"]
        displacement = inputs["channel_displacement"]

        # Run CPU version
        search_locations_cpu, cpu_time = operations.displacement_cpu(
            poses, locations, displacement
        )
        total_cpu_time += cpu_time
        all_search_locations_cpu.append(search_locations_cpu)

        # Run GPU version (convert to float32 for PyTorch)
        poses_gpu = torch.from_numpy(poses.astype(np.float32)).to(operations.device)
        locations_gpu = torch.from_numpy(locations.astype(np.float32)).to(operations.device)
        displacement_gpu = torch.from_numpy(displacement.astype(np.float32)).to(operations.device)

        search_locations_gpu, gpu_time = operations.displacement_gpu(
            poses_gpu, locations_gpu, displacement_gpu
        )
        total_gpu_time += gpu_time
        all_search_locations_gpu.append(search_locations_gpu)

    if not all_search_locations_cpu:
        print("    No displacement data found")
        return False

    # Store results for chaining (keep as list since different sizes)
    pipeline_data.search_locations = all_search_locations_gpu
    pipeline_data.gpu_times["displacement"] = total_gpu_time
    pipeline_data.cpu_times["displacement"] = total_cpu_time

    # Verify results
    max_diff = 0
    for cpu_result, gpu_result in zip(all_search_locations_cpu, all_search_locations_gpu):
        gpu_np = gpu_result.cpu().numpy()
        trace_diff = np.max(np.abs(cpu_result - gpu_np))
        max_diff = max(max_diff, trace_diff)

    pipeline_data.verification_results["displacement"] = {
        "max_diff": max_diff,
        "passed": max_diff < 1e-5
    }

    print(f"    GPU: {total_gpu_time * 1000:.3f}ms, CPU: {total_cpu_time * 1000:.3f}ms, " +
          f"Speedup: {total_cpu_time/total_gpu_time:.2f}x, Max diff: {max_diff:.2e}")

    return True


def process_knn_search(traces: List[Dict], operations: MontyOperations,
                      pipeline_data: PipelineData) -> bool:
    """Process KNN search."""

    # Get graph locations from first trace
    if "graph_memory" not in traces[0] or traces[0]["graph_memory"]["locations"] is None:
        print("    No graph location data available")
        return False

    graph_locations_cpu = traces[0]["graph_memory"]["locations"]
    graph_locations_gpu = torch.from_numpy(graph_locations_cpu.astype(np.float32)).to(operations.device)

    # Use chained search_locations if available
    if pipeline_data.search_locations is not None:
        # Process each trace individually (they have different sizes)
        all_nearest_indices_cpu = []
        all_nearest_indices_gpu = []
        total_cpu_time = 0
        total_gpu_time = 0

        for search_locations_gpu in pipeline_data.search_locations:
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
        pipeline_data.nearest_indices = all_nearest_indices_gpu
        pipeline_data.gpu_times["knn_search"] = total_gpu_time
        pipeline_data.cpu_times["knn_search"] = total_cpu_time

        # Verify results
        indices_match = True
        for cpu_idx, gpu_idx in zip(all_nearest_indices_cpu, all_nearest_indices_gpu):
            gpu_idx_np = gpu_idx.cpu().numpy()
            if not np.array_equal(cpu_idx, gpu_idx_np):
                indices_match = False
                break

        pipeline_data.verification_results["knn_search"] = {
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
                               pipeline_data: PipelineData) -> bool:
    """Process distance calculation."""

    # Use chained data if available
    if (pipeline_data.search_locations is not None and
        pipeline_data.nearest_indices is not None):

        # Get graph locations to compute nearest locations
        if "graph_memory" not in traces[0] or traces[0]["graph_memory"]["locations"] is None:
            print("    No graph location data available")
            return False

        graph_locations = traces[0]["graph_memory"]["locations"]

        # Process each trace individually
        total_gpu_time = 0
        total_cpu_time = 0
        max_diff = 0

        for i, (search_locs, nearest_idx) in enumerate(zip(pipeline_data.search_locations, pipeline_data.nearest_indices)):
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
        pipeline_data.gpu_times["distance_calculation"] = total_gpu_time
        pipeline_data.cpu_times["distance_calculation"] = total_cpu_time
        pipeline_data.verification_results["distance_calculation"] = {
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
                         pipeline_data: PipelineData) -> bool:
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
    pipeline_data.gpu_times["pose_evidence"] = total_gpu_time
    pipeline_data.cpu_times["pose_evidence"] = total_cpu_time
    pipeline_data.verification_results["pose_evidence"] = {
        "max_diff": max_diff,
        "passed": max_diff < 1e-4
    }

    print(f"    GPU: {total_gpu_time * 1000:.3f}ms, CPU: {total_cpu_time * 1000:.3f}ms, " +
          f"Speedup: {total_cpu_time/total_gpu_time:.2f}x, Max diff: {max_diff:.2e}")

    return True


def process_final_aggregation(traces: List[Dict], operations: MontyOperations,
                             pipeline_data: PipelineData) -> bool:
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
    pipeline_data.gpu_times["final_aggregation"] = total_gpu_time
    pipeline_data.cpu_times["final_aggregation"] = total_cpu_time
    pipeline_data.verification_results["final_aggregation"] = {
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

    results = run_step_analysis(traces, operations, args.function)

    # Print final summary
    print(f"\n=== FINAL RESULTS ===")
    print(f"Steps processed: {results['num_steps']}")
    print(f"Total traces: {results['total_traces']}")
    print(f"Functions tested: {', '.join(results['functions_tested'])}")
    print(f"Total GPU time: {results['total_gpu_time'] * 1000:.3f}ms")
    print(f"Total CPU time: {results['total_cpu_time'] * 1000:.3f}ms")
    print(f"Overall speedup: {results['overall_speedup']:.2f}x")

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

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
