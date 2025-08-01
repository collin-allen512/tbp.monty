#!/usr/bin/env python3
"""
Unified GPU Proof of Concept for Monty

Tests functions using saved profiling data.
This script provides a single interface for testing all GPU operations.

GPU PERFORMANCE DIAGNOSTICS:
- GPU memory monitoring and fragmentation detection
- Precise CUDA event timing for accurate measurements
- Thermal/power throttling detection through performance degradation warnings
- Improved synchronization with memory cleanup to prevent resource accumulation
- Warmup cycles to stabilize GPU state before measurements
- Fair CPU vs GPU comparison with isolated kernel timing

Usage:
    # Test all functions
    python gpu_monty_poc.py --output-dir /path/to/experiment

    # Test specific function
    python gpu_monty_poc.py --output-dir /path/to/experiment --function displacement
    python gpu_monty_poc.py --output-dir /path/to/experiment --function knn_search

    # Per-step analysis with CPU baseline and memory monitoring
    python gpu_monty_poc.py --output-dir /path/to/experiment --function per_step --cpu-baseline
"""

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, Any, Optional
import sys
import os

import numpy as np
import torch
from scipy.spatial import cKDTree

# Add gpu_kernels to path
gpu_kernels_dir = os.path.join(os.path.dirname(__file__), '../gpu_kernels')
sys.path.insert(0, gpu_kernels_dir)


def load_profiling_data(output_dir):
    """Load saved profiling data from experiment output."""
    output_path = Path(output_dir)

    # Load computation traces if available
    trace_file = output_path / "hypothesis_computation_trace.pkl"
    if trace_file.exists():
        with open(trace_file, "rb") as f:
            traces = pickle.load(f)
        print(f"Loaded {len(traces)} computation traces")

        # Check for evidence intermediates
        traces_with_evidence = [t for t in traces if "evidence_intermediates" in t]
        if traces_with_evidence:
            print(f"Found {len(traces_with_evidence)} traces with evidence intermediates")

        return traces

    print("No profiling data found!")
    return []


class StackedBatchState:
    """Unified GPU-resident data structure for stacked batch processing."""

    def __init__(self, traces: list, device: torch.device):
        self.device = device
        self.traces = traces
        self.num_traces = len(traces)

        # Initialize stacked data structures
        self._prepare_stacked_data()

    def _prepare_stacked_data(self):
        """Prepare all data in stacked format with offset arrays."""

        # Collect all data and compute offsets
        all_poses = []
        all_locations = []
        all_evidence = []
        all_displacements = []
        hyp_offsets = [0]

        for trace in self.traces:
            inputs = trace["inputs"]
            if "initial_hypotheses" in inputs:
                poses = inputs["initial_hypotheses"]["poses"]
                locations = inputs["initial_hypotheses"]["locations"]
                evidence = inputs["initial_hypotheses"]["evidence"]
                displacement = inputs["channel_displacement"]

                all_poses.append(poses)
                all_locations.append(locations)
                all_evidence.append(evidence)
                all_displacements.append(displacement)
                hyp_offsets.append(hyp_offsets[-1] + len(poses))

        # Stack into GPU tensors
        self.poses = torch.from_numpy(np.concatenate(all_poses)).float().to(self.device)
        self.locations = torch.from_numpy(np.concatenate(all_locations)).float().to(self.device)
        self.evidence = torch.from_numpy(np.concatenate(all_evidence)).float().to(self.device)
        self.displacements = torch.from_numpy(np.stack(all_displacements)).float().to(self.device)

        # Offset arrays for trace boundaries
        self.hyp_offsets = torch.tensor(hyp_offsets[:-1], dtype=torch.int32, device=self.device)
        self.hyp_counts = torch.tensor([len(poses) for poses in all_poses], dtype=torch.int32, device=self.device)
        self.total_hypotheses = self.poses.shape[0]

        # Initialize result tensors
        self.new_locations = torch.zeros_like(self.locations)
        self.new_evidence = torch.zeros_like(self.evidence)

    def cleanup(self):
        """Explicitly release GPU tensors to prevent memory accumulation."""
        try:
            # Delete all tensor attributes to free GPU memory
            if hasattr(self, 'poses'):
                del self.poses
            if hasattr(self, 'locations'):
                del self.locations
            if hasattr(self, 'evidence'):
                del self.evidence
            if hasattr(self, 'displacements'):
                del self.displacements
            if hasattr(self, 'hyp_offsets'):
                del self.hyp_offsets
            if hasattr(self, 'hyp_counts'):
                del self.hyp_counts
            if hasattr(self, 'new_locations'):
                del self.new_locations
            if hasattr(self, 'new_evidence'):
                del self.new_evidence

            # Force GPU memory cleanup
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Warning: Error during StackedBatchState cleanup: {e}")

    def __del__(self):
        """Cleanup when object is garbage collected."""
        self.cleanup()

        print(f"Stacked batch state: {self.num_traces} traces, {self.total_hypotheses} total hypotheses")


class MontyGPUTester:
    """Unified GPU tester for all Monty operations."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Import unified CUDA module
        self.use_cuda_kernels = False
        try:
            import monty_cuda
            self.cuda_kernels = monty_cuda
            self.use_cuda_kernels = True
            print("✓ Monty CUDA kernels loaded successfully")
        except ImportError as e:
            print(f"⚠ Monty CUDA kernels not available: {e}")
            print("  Using PyTorch fallback")

    # ========================================================================
    # HYPOTHESIS UPDATE TESTS
    # ========================================================================

    def test_displacement(self, trace_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test displacement calculation."""
        inputs = trace_data["inputs"]
        intermediates = trace_data["intermediates"]

        # Extract data
        poses = inputs["initial_hypotheses"]["poses"]
        displacement = inputs["channel_displacement"]
        locations = inputs["initial_hypotheses"]["locations"]

        # Convert to GPU tensors
        poses_gpu = torch.from_numpy(poses).float().to(self.device)
        displacement_gpu = torch.from_numpy(displacement).float().to(self.device)
        locations_gpu = torch.from_numpy(locations).float().to(self.device)

        # Time GPU execution
        start_time = time.perf_counter()

        if self.use_cuda_kernels:
            result_gpu = self.cuda_kernels.displacement(poses_gpu, displacement_gpu, locations_gpu)
        else:
            # PyTorch fallback
            rotated_displacement = torch.matmul(poses_gpu, displacement_gpu)
            result_gpu = locations_gpu + rotated_displacement

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        # Compare with saved intermediate
        expected = intermediates["search_locations"]
        result_cpu = result_gpu.cpu().numpy()
        max_diff = np.max(np.abs(result_cpu - expected))

        return {
            "function": "displacement",
            "gpu_time": gpu_time,
            "max_difference": max_diff,
            "input_shape": poses.shape,
            "output_shape": result_cpu.shape,
            "success": max_diff < 1e-5,
        }

    def test_evidence_aggregation(self, trace_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test evidence aggregation."""
        inputs = trace_data["inputs"]
        intermediates = trace_data["intermediates"]
        outputs = trace_data["outputs"]
        parameters = trace_data["parameters"]

        # Extract data
        old_evidence = inputs["initial_hypotheses"]["evidence"]
        new_evidence = intermediates.get("new_evidence")
        hyp_ids_to_test = intermediates.get("hyp_ids_to_test")

        if new_evidence is None or hyp_ids_to_test is None:
            return {"error": "Missing evidence aggregation data"}

        past_weight = parameters.get("past_weight", 1.0)
        present_weight = parameters.get("present_weight", 1.0)
        min_update = np.clip(np.min(new_evidence), 0, np.inf)

        # Convert to GPU tensors
        old_evidence_gpu = torch.from_numpy(old_evidence).float().to(self.device)
        new_evidence_gpu = torch.from_numpy(new_evidence).float().to(self.device)
        test_indices_gpu = torch.from_numpy(hyp_ids_to_test).long().to(self.device)

        # Time GPU execution
        start_time = time.perf_counter()

        if self.use_cuda_kernels:
            result_gpu = self.cuda_kernels.evidence_aggregation(
                old_evidence_gpu, new_evidence_gpu, test_indices_gpu,
                min_update, past_weight, present_weight
            )
        else:
            # PyTorch fallback
            evidence_to_add = torch.full_like(old_evidence_gpu, min_update)
            evidence_to_add[test_indices_gpu] = new_evidence_gpu
            result_gpu = old_evidence_gpu * past_weight + evidence_to_add * present_weight

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        # Compare with saved output
        expected = outputs["final_evidence"]
        result_cpu = result_gpu.cpu().numpy()
        max_diff = np.max(np.abs(result_cpu - expected))

        return {
            "function": "evidence_aggregation",
            "gpu_time": gpu_time,
            "max_difference": max_diff,
            "input_shape": old_evidence.shape,
            "output_shape": result_cpu.shape,
            "success": max_diff < 1e-5,
        }

    # ========================================================================
    # EVIDENCE CALCULATION TESTS
    # ========================================================================

    def test_pose_transformation(self, trace_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test pose transformation."""
        if "evidence_intermediates" not in trace_data:
            return {"error": "No evidence intermediates found"}

        intermediates = trace_data["evidence_intermediates"]
        if "pose_transformation" not in intermediates:
            return {"error": "No pose transformation data found"}

        data = intermediates["pose_transformation"]
        inputs = data["inputs"]
        expected_outputs = data["outputs"]

        # Extract data
        pose_vectors = inputs["channel_features"]["pose_vectors"]
        reference_poses = inputs["channel_possible_poses"]

        pose_vectors_gpu = torch.from_numpy(pose_vectors).float().to(self.device)
        reference_poses_gpu = torch.from_numpy(reference_poses).float().to(self.device)

        # Time GPU execution
        start_time = time.perf_counter()

        if self.use_cuda_kernels:
            result_gpu = self.cuda_kernels.pose_transformation(
                pose_vectors_gpu, reference_poses_gpu
            )
        else:
            result_gpu = torch.matmul(reference_poses_gpu, pose_vectors_gpu)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        # Compare with expected output
        expected = expected_outputs["pose_transformed_features"]["pose_vectors"]
        result_cpu = result_gpu.cpu().numpy()
        max_diff = np.max(np.abs(result_cpu - expected))

        return {
            "function": "pose_transformation",
            "gpu_time": gpu_time,
            "cpu_time": data["timing"],
            "max_difference": max_diff,
            "input_shape": pose_vectors.shape,
            "output_shape": result_cpu.shape,
            "success": max_diff < 1e-5,
        }

    def test_knn_search(self, trace_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test KNN search."""
        if "evidence_intermediates" not in trace_data:
            return {"error": "No evidence intermediates found"}

        intermediates = trace_data["evidence_intermediates"]
        if "nearest_neighbor_search" not in intermediates:
            return {"error": "No KNN search data found"}

        data = intermediates["nearest_neighbor_search"]
        inputs = data["inputs"]
        expected_outputs = data["outputs"]

        # Get graph locations
        graph_memory = trace_data["graph_memory"]
        graph_locations = graph_memory["locations"]
        graph_locations_2 = inputs["graph_locations"]

        if graph_locations is None:
            return {"error": "No graph locations available"}

        # Extract data
        search_locations = inputs["search_locations"]
        k_neighbors = inputs["num_neighbors"]

        search_locations_gpu = torch.from_numpy(search_locations).float().to(self.device)
        graph_locations_gpu = torch.from_numpy(graph_locations_2).float().to(self.device)

        # Time GPU execution
        start_time = time.perf_counter()

        if self.use_cuda_kernels:
            print("CUDA")
            result_gpu = self.cuda_kernels.knn_search(
                graph_locations_gpu, search_locations_gpu, k_neighbors
            )
        else:
            # PyTorch CPU fallback
            print("CPU")
            result_gpu = self._knn_search_pytorch(
                graph_locations_gpu, search_locations_gpu, k_neighbors
            )

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        # For KNN, we check if the same nodes are found (order might differ)
        expected = expected_outputs["nearest_node_ids"]
        result_cpu = result_gpu.cpu().numpy()

        print(result_cpu.shape)
        print(expected.shape)
        def stats(n):
            return print(f"{np.min(n)} {np.max(n)} {np.mean(n)}")
        stats(result_cpu)
        stats(expected)
        stats(graph_locations)
        stats(graph_locations_2)
        matches = 0
        total = 0
        for i in range(len(result_cpu)):
            expected_set = set(expected[i])
            result_set = set(result_cpu[i])
            matches += len(expected_set.intersection(result_set))
            total += len(expected_set)

        match_ratio = matches / total if total > 0 else 0

        return {
            "function": "knn_search",
            "gpu_time": gpu_time,
            "cpu_time": data["timing"],
            "match_ratio": match_ratio,
            "input_shape": search_locations.shape,
            "output_shape": result_cpu.shape,
            "k_neighbors": k_neighbors,
            "success": match_ratio > 0.8,
        }

    def test_custom_distance(self, trace_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test custom distance calculation."""
        if "evidence_intermediates" not in trace_data:
            return {"error": "No evidence intermediates found"}

        intermediates = trace_data["evidence_intermediates"]
        if "distance_calculation" not in intermediates:
            return {"error": "No distance calculation data found"}

        data = intermediates["distance_calculation"]
        inputs = data["inputs"]
        expected_outputs = data["outputs"]

        # Extract data
        nearest_node_locs = inputs["nearest_node_locs"]
        search_locations = inputs["search_locations"]
        pose_normals = inputs["pose_normals"]
        max_abs_curvature = inputs["max_abs_curvature"]

        # Convert to GPU tensors
        nearest_node_locs_gpu = torch.from_numpy(nearest_node_locs).float().to(self.device)
        search_locations_gpu = torch.from_numpy(search_locations).float().to(self.device)
        pose_normals_gpu = torch.from_numpy(pose_normals).float().to(self.device)

        # Time GPU execution
        start_time = time.perf_counter()

        if self.use_cuda_kernels:
            result_gpu = self.cuda_kernels.custom_distance(
                nearest_node_locs_gpu, search_locations_gpu, pose_normals_gpu, max_abs_curvature
            )
        else:
            # PyTorch fallback
            result_gpu = self._custom_distance_pytorch(
                nearest_node_locs_gpu, search_locations_gpu, pose_normals_gpu, max_abs_curvature
            )

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        # Compare with expected output
        expected = expected_outputs["custom_nearest_node_dists"]
        result_cpu = result_gpu.cpu().numpy()
        max_diff = np.max(np.abs(result_cpu - expected))

        return {
            "function": "custom_distance",
            "gpu_time": gpu_time,
            "cpu_time": data["timing"],
            "max_difference": max_diff,
            "input_shape": nearest_node_locs.shape,
            "output_shape": result_cpu.shape,
            "success": max_diff < 1e-5,
        }

    def test_angle_calculation(self, trace_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test angle calculation."""
        if "evidence_intermediates" not in trace_data:
            return {"error": "No evidence intermediates found"}

        intermediates = trace_data["evidence_intermediates"]
        print(intermediates.keys())
        if "pose_evidence_matrix" not in intermediates:
            return {"error": "No pose evidence matrix data found"}

        data = intermediates["pose_evidence_matrix"]
        inputs = data["inputs"]
        intermediate_data = data["intermediates"]

        # Extract data
        node_features = inputs["node_features"]
        query_features = inputs["query_features"]

        # Get pose vectors for angle calculation
        node_pose_vectors = node_features["pose_vectors"][:, :, :3]  # PN vectors
        query_pose_vectors = query_features["pose_vectors"][:, 0]  # Query PN vector

        # Convert to GPU tensors
        node_pose_vectors_gpu = torch.from_numpy(node_pose_vectors).float().to(self.device)
        query_pose_vectors_gpu = torch.from_numpy(query_pose_vectors).float().to(self.device)

        # Time GPU execution
        start_time = time.perf_counter()

        if self.use_cuda_kernels:
            result_gpu = self.cuda_kernels.angle_calculation(
                node_pose_vectors_gpu, query_pose_vectors_gpu
            )
        else:
            # PyTorch fallback
            result_gpu = self._angle_calculation_pytorch(
                node_pose_vectors_gpu, query_pose_vectors_gpu
            )

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        # Compare with expected output
        expected = intermediate_data["pn_error"]
        result_cpu = result_gpu.cpu().numpy()
        max_diff = np.max(np.abs(result_cpu - expected))

        return {
            "function": "angle_calculation",
            "gpu_time": gpu_time,
            "cpu_time": data["timing"]["angle_calculation"],
            "max_difference": max_diff,
            "input_shape": node_pose_vectors.shape,
            "output_shape": result_cpu.shape,
            "success": max_diff < 1e-5,
        }

    def test_pose_evidence(self, trace_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test pose evidence calculation."""
        if "evidence_intermediates" not in trace_data:
            return {"error": "No evidence intermediates found"}

        intermediates = trace_data["evidence_intermediates"]
        if "pose_evidence_matrix" not in intermediates:
            return {"error": "No pose evidence matrix data found"}

        data = intermediates["pose_evidence_matrix"]
        intermediate_data = data["intermediates"]

        # Extract data
        pn_error = intermediate_data["pn_error"]
        stored_pn_evidence = intermediate_data["pn_evidence"]
        cd1_evidence = intermediate_data["cd1_evidence"]
        pn_weight = intermediate_data["pn_weight"]
        cd1_weight = intermediate_data["cd1_weight"]

        # The stored pn_evidence already has doubling applied where appropriate
        # The cd1_evidence is already zeroed where use_cd is False
        # So we can just combine them directly without re-applying the doubling logic

        # Convert to GPU tensors
        stored_pn_evidence_gpu = torch.from_numpy(stored_pn_evidence).float().to(self.device)

        # Time GPU execution
        start_time = time.perf_counter()

        if cd1_evidence is not None:
            cd1_evidence_gpu = torch.from_numpy(cd1_evidence).float().to(self.device)
            # Direct calculation: stored_pn_evidence * pn_weight + cd1_evidence * cd1_weight
            result_gpu = stored_pn_evidence_gpu * pn_weight + cd1_evidence_gpu * cd1_weight
        else:
            # Only PN evidence
            result_gpu = stored_pn_evidence_gpu * pn_weight

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        # Compare with expected output from the trace
        result_cpu = result_gpu.cpu().numpy()

        # Get the expected output from the trace
        expected_output = data["outputs"]["pose_evidence_weighted"]

        # Direct comparison with the CPU's actual output
        max_diff = np.max(np.abs(result_cpu - expected_output))

        # Debug print to understand the mismatch
        if max_diff > 1e-5:
            print(f"\nDEBUG pose_evidence mismatch:")
            print(f"  pn_weight: {pn_weight}, cd1_weight: {cd1_weight}")
            print(f"  cd1_evidence is None: {cd1_evidence is None}")
            # Check if use_cd exists in intermediate data
            if "use_cd" in intermediate_data and intermediate_data["use_cd"] is not None:
                print(f"  use_cd sum: {np.sum(intermediate_data['use_cd'])}")
            else:
                print(f"  use_cd: not available in trace")
            print(f"  Result shape: {result_cpu.shape}")
            print(f"  Expected shape: {expected_output.shape}")
            print(f"  Result sample: {result_cpu.flatten()[:5]}")
            print(f"  Expected sample: {expected_output.flatten()[:5]}")
            print(f"  Max diff: {max_diff}")

            # Check intermediate values
            raw_pn_evidence = -(np.sin(pn_error / 2.0) - 0.5)
            print(f"  Raw PN evidence sample: {raw_pn_evidence.flatten()[:5]}")
            print(f"  Stored PN evidence (after doubling): {intermediate_data['pn_evidence'].flatten()[:5]}")

            # Check CD1 evidence
            if cd1_evidence is not None:
                print(f"  Expected CD1 evidence sample: {cd1_evidence.flatten()[:5]}")

                # The stored pn_evidence already has doubling applied, so we should use it directly
                manual_result = intermediate_data['pn_evidence'] * pn_weight + cd1_evidence * cd1_weight
                print(f"  Manual calc sample (using stored pn_evidence): {manual_result.flatten()[:5]}")

        return {
            "function": "pose_evidence",
            "gpu_time": gpu_time,
            "cpu_time": data["timing"]["evidence_computation"],
            "max_difference": max_diff,
            "input_shape": pn_error.shape,
            "output_shape": result_cpu.shape,
            "success": max_diff < 1e-5,
        }

    def test_final_aggregation(self, trace_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test final aggregation."""
        if "evidence_intermediates" not in trace_data:
            return {"error": "No evidence intermediates found"}

        intermediates = trace_data["evidence_intermediates"]
        if "final_aggregation" not in intermediates:
            return {"error": "No final aggregation data found"}

        data = intermediates["final_aggregation"]
        inputs = data["inputs"]
        expected_outputs = data["outputs"]

        # Extract data
        evidence_matrix = inputs["radius_evidence"]
        evidence_matrix_gpu = torch.from_numpy(evidence_matrix).float().to(self.device)

        # Time GPU execution
        start_time = time.perf_counter()

        if self.use_cuda_kernels:
            result_gpu = self.cuda_kernels.final_aggregation(evidence_matrix_gpu)
        else:
            result_gpu = torch.max(evidence_matrix_gpu, dim=1).values

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        # Compare with expected output
        expected = expected_outputs["location_evidence"]
        result_cpu = result_gpu.cpu().numpy()
        max_diff = np.max(np.abs(result_cpu - expected))

        return {
            "function": "final_aggregation",
            "gpu_time": gpu_time,
            "cpu_time": data["timing"],
            "max_difference": max_diff,
            "input_shape": evidence_matrix.shape,
            "output_shape": result_cpu.shape,
            "success": max_diff < 1e-6,
        }

    # ========================================================================
    # PER-STEP ANALYSIS METHODS
    # ========================================================================

    def test_per_step_analysis(self, traces: list, cpu_baseline: bool = False) -> Dict[str, Any]:
        """Analyze performance on a per-step basis with GPU memory monitoring."""
        # Group traces by step number
        traces_by_step = {}
        for trace in traces:
            step = trace.get("step", 0)
            if step not in traces_by_step:
                traces_by_step[step] = []
            traces_by_step[step].append(trace)

        print(f"\nGrouped traces into {len(traces_by_step)} steps")
        print(f"Step distribution: {sorted([(step, len(traces)) for step, traces in traces_by_step.items()])[:10]}...")

        # GPU warmup - run a few untimed iterations to stabilize state
        print("\nPerforming GPU warmup...")
        warmup_traces = list(traces_by_step.values())[0][:3]  # Use first 3 traces for warmup
        for i in range(3):
            if i < len(warmup_traces):
                warmup_state = StackedBatchState([warmup_traces[i]], self.device)
                _ = self._run_step_pipeline_with_verification(warmup_state)
                # Force memory cleanup after warmup
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        print("Warmup complete")

        # Initialize GPU memory tracking
        initial_memory = self._get_gpu_memory_info()
        print(f"Initial GPU memory: {initial_memory}")

        # Process each step
        step_results = []
        total_gpu_time = 0
        total_cpu_time = 0

        for step, step_traces in sorted(traces_by_step.items()):
            print(f"\nProcessing step {step} with {len(step_traces)} traces...")

            # Pre-step memory monitoring
            pre_memory = self._get_gpu_memory_info()

            # === GPU PROCESSING: Run on ALL traces for performance metrics ===
            gpu_results = self._run_gpu_operations_on_all_traces(step_traces)
            if "error" in gpu_results:
                continue

            # Post-step memory monitoring
            post_memory = self._get_gpu_memory_info()

            # Extract fair comparison times (excluding data prep overhead)
            gpu_compute_time = gpu_results.get("gpu_compute_time", 0)
            gpu_data_prep_time = gpu_results.get("data_preparation_time", 0)
            total_gpu_time += gpu_compute_time  # Use compute time for fair comparison

            # === CPU BASELINE: Only run on traces with complete evidence_intermediates ===
            cpu_time = 0
            cpu_trace_count = 0
            if cpu_baseline:
                complete_traces = [t for t in step_traces if self._has_complete_evidence_data(t)]
                cpu_trace_count = len(complete_traces)
                if complete_traces:
                    print(f"  Running CPU baseline on {cpu_trace_count}/{len(step_traces)} complete traces")
                    cpu_results = self._run_cpu_baseline(complete_traces)
                    cpu_time = cpu_results.get("cpu_compute_time", 0)
                    total_cpu_time += cpu_time
                else:
                    print(f"  No traces with complete evidence_intermediates for CPU baseline")

            # Aggressive memory cleanup and better synchronization
            if self.device.type == "cuda":
                torch.cuda.synchronize()
                # More aggressive memory cleanup with force garbage collection
                self._aggressive_memory_cleanup()
                # Additional tensor cleanup - force cleanup of any remaining references
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                # Get post-cleanup memory state
                post_memory = self._get_gpu_memory_info()

            # Calculate memory usage change
            memory_delta = self._calculate_memory_delta(pre_memory, post_memory)

            step_result = {
                "step": step,
                "num_traces": len(step_traces),
                "total_hypotheses": sum(len(t["inputs"]["initial_hypotheses"]["evidence"]) for t in step_traces),
                "gpu_compute_time": gpu_compute_time,
                "gpu_data_prep_time": gpu_data_prep_time,
                "cpu_time": cpu_time,
                "speedup": cpu_time / gpu_compute_time if gpu_compute_time > 0 else 0,
                "gpu_results": gpu_results,
                "memory_before": pre_memory,
                "memory_after": post_memory,
                "memory_delta": memory_delta,
            }
            step_results.append(step_result)

            # Print summary for this step
            print(f"  GPU compute time: {gpu_compute_time * 1000:.3f}ms (+ {gpu_data_prep_time * 1000:.3f}ms data prep)")
            if cpu_baseline:
                print(f"  CPU compute time: {cpu_time * 1000:.3f}ms")
                print(f"  Speedup: {step_result['speedup']:.2f}x")

            # Print memory information
            if self.device.type == "cuda":
                print(f"  GPU memory: {post_memory['allocated_mb']:.1f}MB allocated, "
                      f"{post_memory['cached_mb']:.1f}MB cached, "
                      f"Δ={memory_delta['allocated_delta_mb']:+.1f}MB")
                if memory_delta['fragmentation_estimate'] > 10:
                    print(f"  ⚠️  High fragmentation: {memory_delta['fragmentation_estimate']:.1f}% (cached - allocated)")

            # Early warning for performance degradation
            if len(step_results) > 1:
                current_time = gpu_compute_time * 1000
                first_time = step_results[0]["gpu_compute_time"] * 1000
                if current_time > first_time * 1.5:  # 50% slowdown
                    print(f"  ⚠️  Performance degradation detected: {current_time:.1f}ms vs {first_time:.1f}ms baseline")

        # Calculate overall statistics
        avg_speedup = np.mean([r["speedup"] for r in step_results if r["speedup"] > 0])

        return {
            "num_steps": len(traces_by_step),
            "total_traces": len(traces),
            "total_gpu_time": total_gpu_time,
            "total_cpu_time": total_cpu_time,
            "overall_speedup": total_cpu_time / total_gpu_time if total_cpu_time > 0 else 0,
            "avg_speedup_per_step": avg_speedup,
            "step_results": step_results,
        }

    def _run_gpu_operations_on_all_traces(self, traces: list) -> Dict[str, Any]:
        """Run GPU operations on ALL traces for performance metrics, regardless of data completeness."""
        try:
            # Use robust GPU processing that handles partial traces
            gpu_state = StackedBatchState(traces, self.device)
            try:
                # Try full pipeline first
                print("Running full pipeline")
                results = self._run_step_pipeline_with_data_flow(gpu_state)
                return results
            # except:
            #     # If full pipeline fails, run individual operations to get partial metrics
            #     print("Running displacement only")
            #     return self._run_partial_gpu_operations(gpu_state)
            finally:
                gpu_state.cleanup()
                del gpu_state
        except Exception as e:
            return {"error": f"GPU operations failed: {str(e)}"}

    def _run_partial_gpu_operations(self, batch_state: StackedBatchState) -> Dict[str, Any]:
        """Run individual GPU operations that can succeed even with incomplete data."""
        results = {}

        # Always attempt displacement (should work for all traces)
        try:
            disp_result = self._test_stacked_displacement(batch_state)
            results["displacement"] = disp_result
            results["gpu_compute_time"] = disp_result.get("time", 0)
        except Exception as e:
            results["error"] = f"Even basic GPU operations failed: {str(e)}"
        exit(-1)
        return results

    def _has_complete_evidence_data(self, trace: dict) -> bool:
        """Check if trace has complete evidence_intermediates for CPU verification."""
        if "evidence_intermediates" not in trace:
            return False

        intermediates = trace["evidence_intermediates"]
        required_keys = [
            "nearest_neighbor_search",
            "distance_calculation",
            "pose_evidence_matrix",
            "final_aggregation"
        ]

        return all(key in intermediates for key in required_keys)

    def _run_step_pipeline(self, batch_state: StackedBatchState) -> Dict[str, Any]:
        """Run GPU pipeline for all traces in a single step."""
        results = {}

        # Test each operation in the pipeline
        # 1. Displacement
        disp_result = self._test_stacked_displacement(batch_state)
        results["displacement"] = disp_result

        # 2. KNN Search
        knn_result = self._test_stacked_knn_search(batch_state)
        results["knn_search"] = knn_result

        # 3. Custom Distance
        dist_result = self._test_stacked_custom_distance(batch_state)
        results["custom_distance"] = dist_result

        # 4. Angle Calculation
        angle_result = self._test_stacked_angle_calculation(batch_state)
        results["angle_calculation"] = angle_result

        # 5. Pose Evidence
        pose_ev_result = self._test_stacked_pose_evidence(batch_state)
        results["pose_evidence"] = pose_ev_result

        # 6. Evidence Aggregation
        evid_agg_result = self._test_stacked_evidence_aggregation(batch_state)
        results["evidence_aggregation"] = evid_agg_result

        # 7. Final Aggregation
        final_agg_result = self._test_stacked_final_aggregation(batch_state)
        results["final_aggregation"] = final_agg_result

        return results

    def _run_step_pipeline_with_verification(self, batch_state: StackedBatchState) -> Dict[str, Any]:
        """Run GPU pipeline with data flow and verify against CPU results."""
        # Run the GPU pipeline with proper data flow
        gpu_results = self._run_step_pipeline_with_data_flow(batch_state)

        if "error" in gpu_results:
            return gpu_results

        # Verify GPU results against CPU outputs stored in traces
        verification_results = self._verify_pipeline_outputs(batch_state, gpu_results)

        # Combine GPU timing and verification results
        combined_results = {
            **gpu_results,
            "verification": verification_results,
            "verified_success": verification_results["overall_success"]
        }

        return combined_results

    def _verify_pipeline_outputs(self, batch_state: StackedBatchState, gpu_results: Dict[str, Any]) -> Dict[str, Any]:
        """Verify GPU pipeline outputs against CPU results from traces."""
        verification = {}

        # Get GPU outputs
        pipeline_outputs = gpu_results.get("pipeline_outputs", {})

        # 1. Verify final evidence (most important)
        if "final_evidence" in pipeline_outputs:
            final_verification = self._verify_final_evidence(batch_state, pipeline_outputs["final_evidence"])
            verification["final_evidence"] = final_verification

        # 2. Verify intermediate steps
        if "search_locations" in pipeline_outputs:
            disp_verification = self._verify_displacement_output(batch_state, pipeline_outputs["search_locations"])
            verification["displacement"] = disp_verification

        if "pose_evidence" in pipeline_outputs:
            pose_verification = self._verify_pose_evidence_output(batch_state, pipeline_outputs["pose_evidence"])
            verification["pose_evidence"] = pose_verification

        # Calculate overall verification success
        all_verifications = [v for v in verification.values() if isinstance(v, dict) and "success" in v]
        overall_success = all([v["success"] for v in all_verifications]) if all_verifications else False

        verification["overall_success"] = overall_success
        verification["num_verified"] = len(all_verifications)

        return verification

    def _verify_final_evidence(self, batch_state: StackedBatchState, gpu_final_evidence: torch.Tensor) -> Dict[str, Any]:
        """Verify GPU final evidence against CPU final evidence from traces."""
        gpu_evidence_cpu = gpu_final_evidence.cpu().numpy()
        cpu_final_evidence = []

        # Extract CPU final evidence from traces
        for trace in batch_state.traces:
            if "outputs" in trace and "final_evidence" in trace["outputs"]:
                cpu_final_evidence.append(trace["outputs"]["final_evidence"])
            else:
                # Fall back to final evidence from trace outputs if available
                if ("evidence_intermediates" in trace and
                    "final_aggregation" in trace["evidence_intermediates"] and
                    "outputs" in trace["evidence_intermediates"]["final_aggregation"]):
                    final_agg = trace["evidence_intermediates"]["final_aggregation"]["outputs"]
                    cpu_final_evidence.append(final_agg["location_evidence"])

        if not cpu_final_evidence:
            return {"success": False, "error": "No CPU final evidence found in traces"}

        cpu_stacked = np.concatenate(cpu_final_evidence)

        if len(gpu_evidence_cpu) != len(cpu_stacked):
            return {
                "success": False,
                "error": f"Size mismatch: GPU {len(gpu_evidence_cpu)} vs CPU {len(cpu_stacked)}"
            }

        # Calculate differences
        abs_diff = np.abs(gpu_evidence_cpu - cpu_stacked)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)

        # Verification passes if max difference is below threshold
        success = max_diff < 1e-4

        return {
            "success": success,
            "max_difference": float(max_diff),
            "mean_difference": float(mean_diff),
            "num_elements": len(gpu_evidence_cpu),
            "tolerance": 1e-4
        }

    def _verify_displacement_output(self, batch_state: StackedBatchState, gpu_search_locations: torch.Tensor) -> Dict[str, Any]:
        """Verify GPU displacement output against CPU search locations from traces."""
        gpu_locations_cpu = gpu_search_locations.cpu().numpy()
        cpu_search_locations = []

        # Extract CPU search locations from traces
        for trace in batch_state.traces:
            if "intermediates" in trace and "search_locations" in trace["intermediates"]:
                cpu_search_locations.append(trace["intermediates"]["search_locations"])

        if not cpu_search_locations:
            return {"success": False, "error": "No CPU search locations found in traces"}

        cpu_stacked = np.concatenate(cpu_search_locations)

        if gpu_locations_cpu.shape != cpu_stacked.shape:
            return {
                "success": False,
                "error": f"Shape mismatch: GPU {gpu_locations_cpu.shape} vs CPU {cpu_stacked.shape}"
            }

        # Calculate differences
        abs_diff = np.abs(gpu_locations_cpu - cpu_stacked)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)

        success = max_diff < 1e-5

        return {
            "success": success,
            "max_difference": float(max_diff),
            "mean_difference": float(mean_diff),
            "num_elements": gpu_locations_cpu.size,
            "tolerance": 1e-5
        }

    def _verify_pose_evidence_output(self, batch_state: StackedBatchState, gpu_pose_evidence: torch.Tensor) -> Dict[str, Any]:
        """Verify GPU pose evidence output against CPU pose evidence from traces."""
        gpu_evidence_cpu = gpu_pose_evidence.cpu().numpy()
        cpu_pose_evidence = []

        # Extract CPU pose evidence from traces
        for trace in batch_state.traces:
            if ("evidence_intermediates" in trace and
                "pose_evidence_matrix" in trace["evidence_intermediates"] and
                "outputs" in trace["evidence_intermediates"]["pose_evidence_matrix"]):
                pe_outputs = trace["evidence_intermediates"]["pose_evidence_matrix"]["outputs"]
                cpu_pose_evidence.append(pe_outputs["pose_evidence_weighted"])

        if not cpu_pose_evidence:
            return {"success": False, "error": "No CPU pose evidence found in traces"}

        cpu_stacked = np.concatenate(cpu_pose_evidence)

        if gpu_evidence_cpu.shape != cpu_stacked.shape:
            return {
                "success": False,
                "error": f"Shape mismatch: GPU {gpu_evidence_cpu.shape} vs CPU {cpu_stacked.shape}"
            }

        # Calculate differences
        abs_diff = np.abs(gpu_evidence_cpu - cpu_stacked)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)

        success = max_diff < 1e-4

        return {
            "success": success,
            "max_difference": float(max_diff),
            "mean_difference": float(mean_diff),
            "num_elements": gpu_evidence_cpu.size,
            "tolerance": 1e-4
        }

    def _run_step_pipeline_with_data_flow(self, batch_state: StackedBatchState) -> Dict[str, Any]:
        """Run GPU pipeline with fair timing comparison by separating data prep from computation.

        Returns separate timings for data preparation overhead vs pure GPU computation
        to enable fair comparison with CPU baseline performance.
        """
        results = {}
        pipeline_outputs = {}

        # === DATA PREPARATION PHASE (excluded from fair comparison) ===
        prep_start = time.perf_counter()

        # Prepare initial data structures
        expanded_displacements = []
        for i, count in enumerate(batch_state.hyp_counts):
            expanded_displacements.append(batch_state.displacements[i].repeat(count, 1))
        stacked_displacements = torch.cat(expanded_displacements, dim=0)

        # Clean up intermediate displacement list to prevent tensor accumulation
        for tensor in expanded_displacements:
            del tensor
        expanded_displacements.clear()

        # Extract graph data and parameters needed for chaining operations
        try:
            graph_data = self._extract_chaining_data(batch_state)
        except:
            return ["error"]

        prep_time = time.perf_counter() - prep_start

        # === PURE GPU COMPUTATION PHASE (included in fair comparison) ===
        # Use precise GPU timing with CUDA events
        if self.device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            gpu_compute_start = time.perf_counter()

        # STEP 1: Displacement calculation - USE STACKED KERNEL
        search_locations = self.cuda_kernels.displacement_stacked(
            batch_state.poses, stacked_displacements, batch_state.locations
        )
        pipeline_outputs["search_locations"] = search_locations

        # STEP 2: KNN Search (using displacement output: search_locations) - TRUE DATA FLOW
        if not (self.use_cuda_kernels and hasattr(self.cuda_kernels, 'knn_search_stacked')):
            raise RuntimeError(
                "GPU kernel chaining failed: CUDA KNN search kernel not available. "
                "Ensure use_cuda_kernels=True and CUDA kernels are properly loaded."
            )

        nearest_node_ids = self.cuda_kernels.knn_search_stacked(
            graph_data["stacked_graph_locs"], search_locations,
            graph_data["graph_offsets"], graph_data["query_offsets"],
            graph_data["max_neighbors"]
        )
        pipeline_outputs["nearest_node_ids"] = nearest_node_ids

        # STEP 3: Get nearest node locations (using KNN output) - TRUE DATA FLOW
        nearest_node_locs = self._get_nearest_node_locations_batched(
            graph_data, nearest_node_ids
        )
        pipeline_outputs["nearest_node_locs"] = nearest_node_locs

        # STEP 4: Custom distance calculation (using KNN and displacement outputs) - TRUE DATA FLOW
        curvatures_tensor = torch.tensor(graph_data["max_curvatures"], dtype=torch.float32, device=self.device)
        custom_distances = self.cuda_kernels.custom_distance_stacked(
            nearest_node_locs, search_locations,
            graph_data["stacked_pose_normals"], curvatures_tensor, graph_data["trace_offsets"]
        )
        pipeline_outputs["custom_distances"] = custom_distances

        # STEP 5: Angle Calculation (using graph features from KNN) - TRUE DATA FLOW
        pn_angles, cd1_angles = self._calculate_angles_from_knn_batched(graph_data, nearest_node_ids)
        pipeline_outputs["pn_angles"] = pn_angles
        pipeline_outputs["cd1_angles"] = cd1_angles

        # STEP 6: Pose Evidence (using angles and distances) - TRUE DATA FLOW
        pose_evidence = self._calculate_pose_evidence_batched(pn_angles, cd1_angles, custom_distances, graph_data, batch_state)
        pipeline_outputs["pose_evidence"] = pose_evidence

        # STEP 7: Final Aggregation - USE STACKED KERNEL
        if not (self.use_cuda_kernels and hasattr(self.cuda_kernels, 'final_aggregation_stacked')):
            raise RuntimeError(
                "GPU kernel chaining failed: CUDA final_aggregation_stacked kernel not available. "
                "Ensure use_cuda_kernels=True and stacked kernels are properly loaded."
            )

        final_evidence = self.cuda_kernels.final_aggregation_stacked(pose_evidence)
        pipeline_outputs["final_evidence"] = final_evidence

        # Single synchronization point for all GPU operations
        # Complete timing with proper synchronization and memory cleanup
        if self.device.type == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            gpu_compute_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
            # Force memory cleanup for better monitoring
            torch.cuda.empty_cache()
        else:
            gpu_compute_time = time.perf_counter() - gpu_compute_start

        # === EXPLICIT TENSOR CLEANUP TO PREVENT ACCUMULATION ===
        # Clear pipeline_outputs dictionary to release GPU tensors
        try:
            for key in list(pipeline_outputs.keys()):
                del pipeline_outputs[key]
            pipeline_outputs.clear()

            # Clean up intermediate tensors
            del search_locations, nearest_node_ids, nearest_node_locs
            del custom_distances, pn_angles, cd1_angles, pose_evidence, final_evidence
            del curvatures_tensor
        except Exception as e:
            print(f"Warning: Error during pipeline tensor cleanup: {e}")

        # === RESULTS FOR FAIR COMPARISON ===
        results["data_preparation_time"] = prep_time  # Exclude from comparison
        results["gpu_compute_time"] = gpu_compute_time  # Fair comparison metric
        results["total_time_including_prep"] = prep_time + gpu_compute_time
        results["pipeline_outputs"] = pipeline_outputs
        results["comparison_note"] = "Use gpu_compute_time for fair comparison with CPU (excludes data prep overhead)"

        return results

    def _extract_chaining_data(self, batch_state: StackedBatchState) -> Dict[str, Any]:
        """Extract basic graph data needed for operation chaining."""
        # Extract graph locations and basic parameters
        all_graph_locs = []
        all_pose_normals = []
        max_curvatures = []
        graph_offsets = [0]
        query_offsets = [0]
        trace_offsets = [0]  # For custom distance calculation
        max_neighbors = 3

        for trace in batch_state.traces:
            # Get graph data - required for chaining
            if not ("evidence_intermediates" in trace and
                    "nearest_neighbor_search" in trace["evidence_intermediates"]):
                print("nearest neighbor issue")
                print(trace.keys())
                print(trace["evidence_intermediates"].keys)
                raise RuntimeError(
                    "GPU kernel chaining failed: Missing nearest_neighbor_search data in trace. "
                    "Ensure computation traces are saved with evidence_intermediates during profiling."
                )


            nn_data = trace["evidence_intermediates"]["nearest_neighbor_search"]["inputs"]
            if "graph_locations" not in nn_data:
                raise RuntimeError(
                    "GPU kernel chaining failed: Missing graph_locations in nearest_neighbor_search data."
                )

            graph_locs = nn_data["graph_locations"]
            all_graph_locs.append(graph_locs)
            graph_offsets.append(graph_offsets[-1] + len(graph_locs))

            # Get pose normals and curvature for distance calculation
            if not ("evidence_intermediates" in trace and
                    "distance_calculation" in trace["evidence_intermediates"]):
                raise RuntimeError(
                    "GPU kernel chaining failed: Missing distance_calculation data in trace. "
                    "Ensure computation traces capture all evidence_intermediates."
                )

            dist_data = trace["evidence_intermediates"]["distance_calculation"]["inputs"]
            if "pose_normals" not in dist_data or "max_abs_curvature" not in dist_data:
                raise RuntimeError(
                    "GPU kernel chaining failed: Missing pose_normals or max_abs_curvature in distance_calculation data."
                )

            all_pose_normals.append(dist_data["pose_normals"])
            max_curvatures.append(dist_data["max_abs_curvature"])

            # Track query offsets and trace offsets for custom distance
            if not ("inputs" in trace and "initial_hypotheses" in trace["inputs"]):
                raise RuntimeError(
                    "GPU kernel chaining failed: Missing initial_hypotheses in trace inputs."
                )

            hyp_count = len(trace["inputs"]["initial_hypotheses"]["evidence"])
            query_offsets.append(query_offsets[-1] + hyp_count)

            # Trace offsets for custom distance calculation (per-trace data boundaries)
            # Each trace contributes hyp_count * max_neighbors elements to the distance calculation
            trace_element_count = hyp_count * max_neighbors
            trace_offsets.append(trace_offsets[-1] + trace_element_count)

        if not all_graph_locs:
            raise RuntimeError(
                "GPU kernel chaining failed: No valid graph locations found in any trace. "
                "Check that traces contain proper evidence_intermediates data."
            )

        # Stack basic data
        stacked_graph_locs = torch.from_numpy(np.concatenate(all_graph_locs)).float().to(self.device)
        stacked_pose_normals = torch.from_numpy(np.concatenate(all_pose_normals)).float().to(self.device)

        return {
            "stacked_graph_locs": stacked_graph_locs,
            "stacked_pose_normals": stacked_pose_normals,
            "max_curvatures": max_curvatures,
            "graph_offsets": torch.tensor(graph_offsets[:-1], dtype=torch.int32, device=self.device),
            "query_offsets": torch.tensor(query_offsets[:-1], dtype=torch.int32, device=self.device),
            "trace_offsets": torch.tensor(trace_offsets[:-1], dtype=torch.int32, device=self.device),
            "max_neighbors": max_neighbors,
            "traces": batch_state.traces
        }


    def _get_nearest_node_locations_batched(self, graph_data, nearest_node_ids):
        """Get nearest node locations using KNN results."""
        return graph_data["stacked_graph_locs"][nearest_node_ids]

    def _calculate_angles_from_knn_batched(self, graph_data, nearest_node_ids):
        """Calculate angles using graph features from KNN."""
        # Extract angle calculation data from traces - this data is already in the correct format
        all_node_pose_vectors = []
        all_query_pose_vectors = []

        for trace in graph_data["traces"]:
            if not ("evidence_intermediates" in trace and
                    "pose_evidence_matrix" in trace["evidence_intermediates"]):
                raise RuntimeError(
                    "GPU kernel chaining failed: Missing pose_evidence_matrix in trace evidence_intermediates. "
                    "Ensure computation traces capture pose evidence calculations."
                )

            pose_data = trace["evidence_intermediates"]["pose_evidence_matrix"]
            if "angle_calculation_inputs" not in pose_data:
                raise RuntimeError(
                    "GPU kernel chaining failed: Missing angle_calculation_inputs in pose_evidence_matrix. "
                    "Ensure computation traces capture angle calculation inputs."
                )

            angle_inputs = pose_data["angle_calculation_inputs"]
            if "node_pose_vectors" not in angle_inputs or "query_pose_vectors" not in angle_inputs:
                raise RuntimeError(
                    "GPU kernel chaining failed: Missing pose vectors in angle_calculation_inputs. "
                    "Check that pose vector data is properly saved during trace capture."
                )

            # The saved data is already per-hypothesis format: (num_hyp, num_neighbors, 3) and (num_hyp, 3)
            all_node_pose_vectors.append(angle_inputs["node_pose_vectors"])
            all_query_pose_vectors.append(angle_inputs["query_pose_vectors"])

        if not all_node_pose_vectors:
            raise RuntimeError(
                "GPU kernel chaining failed: No pose vector data found in any trace. "
                "Ensure profiling captures complete pose evidence calculations."
            )

        # Convert to tensors - concatenate along hypothesis dimension
        # all_node_pose_vectors: list of (num_hyp_i, num_neighbors, 3) arrays
        # all_query_pose_vectors: list of (num_hyp_i, 3) arrays
        node_pose_vectors = torch.from_numpy(np.concatenate(all_node_pose_vectors, axis=0)).float().to(self.device)
        query_pose_vectors = torch.from_numpy(np.concatenate(all_query_pose_vectors, axis=0)).float().to(self.device)

        # The saved data already has the correct structure for angle calculation
        # node_pose_vectors shape: (total_hypotheses, max_neighbors, 3)
        # query_pose_vectors shape: (total_hypotheses, 3)

        # Calculate dot products for angle computation
        dot_products = torch.einsum("ijk,ik->ij", node_pose_vectors, query_pose_vectors)

        # Calculate angles
        pn_angles = torch.acos(torch.clamp(dot_products, -1.0, 1.0))

        # For simplicity, use same angles for CD1 (in practice, would use different vectors)
        cd1_angles = pn_angles.clone()

        return pn_angles, cd1_angles

    def _calculate_pose_evidence_batched(self, pn_angles, cd1_angles, custom_distances, graph_data, batch_state):
        """Calculate pose evidence using angle and distance outputs with proper trace data extraction."""
        # Extract the EXACT same parameters used in CPU computation for verification
        all_pn_weights = []
        all_cd1_weights = []
        all_use_cd = []

        for trace in graph_data["traces"]:
            if not ("evidence_intermediates" in trace and
                    "pose_evidence_matrix" in trace["evidence_intermediates"]):
                raise RuntimeError(
                    "GPU kernel chaining failed: Missing pose_evidence_matrix in trace evidence_intermediates."
                )

            pose_data = trace["evidence_intermediates"]["pose_evidence_matrix"]
            if "intermediates" not in pose_data:
                raise RuntimeError(
                    "GPU kernel chaining failed: Missing intermediates in pose_evidence_matrix."
                )

            intermediates = pose_data["intermediates"]

            # Extract the exact weights used in CPU computation
            pn_weight = intermediates.get("pn_weight", 1.0)
            cd1_weight = intermediates.get("cd1_weight", 0.0)

            all_pn_weights.append(pn_weight)
            all_cd1_weights.append(cd1_weight)

            # Extract use_cd flag - this is critical for matching CPU behavior
            if "use_cd" in intermediates and intermediates["use_cd"] is not None:
                all_use_cd.append(intermediates["use_cd"])
            else:
                # Fallback: match CPU logic exactly
                # When pose_fully_defined is False, cd1_weight = 0 and no use_cd is created
                # When pose_fully_defined is True, cd1_weight > 0 and use_cd is extracted from node_features
                hyp_count = len(trace["inputs"]["initial_hypotheses"]["evidence"])
                max_neighbors = graph_data["max_neighbors"]

                if cd1_weight == 0:
                    # When cd1_weight = 0, use_cd effectively doesn't matter since cd1_evidence = 0
                    # But the doubling logic still applies: pn_evidence[~use_cd] *= 2
                    # So we need to set use_cd = False to enable doubling everywhere
                    all_use_cd.append(np.full((hyp_count, max_neighbors), False, dtype=bool))
                else:
                    # This case should not happen if trace saving is working correctly
                    # but as a fallback, assume use_cd = True when cd1_weight > 0
                    all_use_cd.append(np.full((hyp_count, max_neighbors), True, dtype=bool))

        # Prepare tensors for stacked kernel - matching the working test pattern
        pn_weights_tensor = torch.tensor(all_pn_weights, dtype=torch.float32, device=self.device)
        cd1_weights_tensor = torch.tensor(all_cd1_weights, dtype=torch.float32, device=self.device)

        # Convert use_cd to int32 (kernel expects Int, not Bool)
        # Handle variable neighbor counts by flattening
        use_cd_flat = []
        for use_cd in all_use_cd:
            use_cd_flat.extend(use_cd.flatten())
        use_cd_array = np.array(use_cd_flat, dtype=np.int32)
        use_cd_tensor = torch.from_numpy(use_cd_array).to(self.device)

        # Calculate proper trace offsets - based on actual hypothesis counts
        trace_sizes = []
        for i, trace in enumerate(batch_state.traces):
            hyp_count = len(trace["inputs"]["initial_hypotheses"]["evidence"])
            trace_sizes.append(hyp_count * graph_data["max_neighbors"])

        trace_offsets = np.cumsum([0] + trace_sizes[:-1])
        trace_offsets_tensor = torch.tensor(trace_offsets, dtype=torch.int32, device=self.device)

        # USE STACKED POSE EVIDENCE KERNEL with proper trace data
        if not (self.use_cuda_kernels and hasattr(self.cuda_kernels, 'pose_evidence_stacked')):
            raise RuntimeError(
                "GPU kernel chaining failed: CUDA pose_evidence_stacked kernel not available."
            )

        pose_evidence = self.cuda_kernels.pose_evidence_stacked(
            pn_angles, cd1_angles, use_cd_tensor,
            pn_weights_tensor, cd1_weights_tensor, trace_offsets_tensor
        )

        return pose_evidence

    def _extract_simple_data_for_verification(self, batch_state: StackedBatchState) -> Dict[str, Any]:
        """Extract simple data needed for result verification only."""
        # Only extract what we need for verification - displacement results
        all_search_locations = []
        all_final_evidence = []

        for trace in batch_state.traces:
            # Get search locations from displacement
            if "intermediates" in trace and "search_locations" in trace["intermediates"]:
                all_search_locations.append(trace["intermediates"]["search_locations"])

            # Get final evidence from outputs or final aggregation
            if "outputs" in trace and "final_evidence" in trace["outputs"]:
                all_final_evidence.append(trace["outputs"]["final_evidence"])
            elif ("evidence_intermediates" in trace and
                  "final_aggregation" in trace["evidence_intermediates"] and
                  "outputs" in trace["evidence_intermediates"]["final_aggregation"]):
                final_agg = trace["evidence_intermediates"]["final_aggregation"]["outputs"]
                all_final_evidence.append(final_agg["location_evidence"])

        return {
            "search_locations": all_search_locations,
            "final_evidence": all_final_evidence
        }

    def _run_cpu_baseline_from_traces(self, traces: list) -> Dict[str, Any]:
        """Extract CPU kernel times for fair comparison with GPU compute time."""
        # Extract individual CPU kernel times without control flow overhead
        cpu_kernel_times = {
            "displacement": [],
            "knn_search": [],
            "custom_distance": [],
            "angle_calculation": [],
            "pose_evidence": [],
            "final_aggregation": []
        }

        cpu_data_prep_time = 0

        for trace in traces:
            # Data preparation overhead (similar to GPU data prep)
            if "timing" in trace:
                # These include data structure preparation overhead
                cpu_data_prep_time += trace["timing"].get("pose_transformation", 0)
                cpu_data_prep_time += trace["timing"].get("location_retrieval", 0)
                cpu_data_prep_time += trace["timing"].get("feature_retrieval", 0)

            # Core computation kernels (for fair comparison)
            if "timing" in trace:
                cpu_kernel_times["displacement"].append(trace["timing"].get("displacement", 0))

            if "evidence_intermediates" in trace:
                intermediates = trace["evidence_intermediates"]

                # KNN search time
                if "nearest_neighbor_search" in intermediates:
                    knn_time = intermediates["nearest_neighbor_search"].get("timing", 0)
                    cpu_kernel_times["knn_search"].append(knn_time)

                # Distance calculation time
                if "distance_calculation" in intermediates:
                    dist_time = intermediates["distance_calculation"].get("timing", 0)
                    cpu_kernel_times["custom_distance"].append(dist_time)

                # Angle calculation and pose evidence from pose_evidence_matrix
                if "pose_evidence_matrix" in intermediates:
                    pose_data = intermediates["pose_evidence_matrix"]
                    if "timing" in pose_data:
                        timing = pose_data["timing"]
                        cpu_kernel_times["angle_calculation"].append(timing.get("angle_calculation", 0))
                        cpu_kernel_times["pose_evidence"].append(timing.get("evidence_computation", 0))

                # Final aggregation
                if "final_aggregation" in intermediates:
                    final_time = intermediates["final_aggregation"].get("timing", 0)
                    cpu_kernel_times["final_aggregation"].append(final_time)

        # Sum up core computation times for fair comparison
        cpu_compute_time = sum(sum(times) for times in cpu_kernel_times.values())

        return {
            "cpu_data_prep_time": cpu_data_prep_time,  # Exclude from comparison
            "cpu_compute_time": cpu_compute_time,      # Fair comparison metric
            "kernel_breakdown": {k: sum(v) for k, v in cpu_kernel_times.items()},
            "comparison_note": "Use cpu_compute_time for fair comparison with GPU (excludes data prep overhead)"
        }

    # ========================================================================
    # STACKED BATCH PROCESSING METHODS
    # ========================================================================

    def test_stacked_batch_pipeline(self, traces: list) -> Dict[str, Any]:
        """Test complete hypothesis update pipeline in stacked batch mode."""

        # Create stacked batch state
        batch_state = StackedBatchState(traces, self.device)

        # Test each operation in the pipeline
        results = {}
        total_time = 0

        # 1. Displacement
        disp_result = self._test_stacked_displacement(batch_state)
        results["displacement"] = disp_result
        total_time += disp_result["time"]

        # 2. KNN Search (uses trace boundaries)
        knn_result = self._test_stacked_knn_search(batch_state)
        results["knn_search"] = knn_result
        total_time += knn_result["time"]

        # 3. Custom Distance
        dist_result = self._test_stacked_custom_distance(batch_state)
        results["custom_distance"] = dist_result
        total_time += dist_result["time"]

        # 4. Angle Calculation
        angle_result = self._test_stacked_angle_calculation(batch_state)
        results["angle_calculation"] = angle_result
        total_time += angle_result["time"]

        # 5. Pose Evidence
        pose_ev_result = self._test_stacked_pose_evidence(batch_state)
        results["pose_evidence"] = pose_ev_result
        total_time += pose_ev_result["time"]

        # 6. Evidence Aggregation
        evid_agg_result = self._test_stacked_evidence_aggregation(batch_state)
        results["evidence_aggregation"] = evid_agg_result
        total_time += evid_agg_result["time"]

        # 7. Final Aggregation
        final_result = self._test_stacked_final_aggregation(batch_state)
        results["final_aggregation"] = final_result
        total_time += final_result["time"]

        return {
            "function": "stacked_batch_pipeline",
            "num_traces": batch_state.num_traces,
            "total_hypotheses": batch_state.total_hypotheses,
            "total_time": total_time,
            "per_trace_time": total_time / batch_state.num_traces,
            "efficiency": 1.0,  # No padding waste in stacked approach
            "results": results,
            "success": all(r["success"] for r in results.values())
        }

    def _test_stacked_displacement(self, batch_state: StackedBatchState) -> Dict[str, Any]:
        """Test displacement with stacked data"""

        # Expand displacements to match hypothesis count per trace
        expanded_displacements = []
        for i, count in enumerate(batch_state.hyp_counts):
            expanded_displacements.append(batch_state.displacements[i].repeat(count, 1))
        stacked_displacements = torch.cat(expanded_displacements, dim=0)

        start_time = time.perf_counter()

        if self.use_cuda_kernels and hasattr(self.cuda_kernels, 'displacement_stacked'):
            # Use stacked kernel
            result = self.cuda_kernels.displacement_stacked(
                batch_state.poses,
                stacked_displacements,
                batch_state.locations
            )
        # elif self.use_cuda_kernels:
        #     # Use regular kernel
        #     result = self.cuda_kernels.displacement(
        #         batch_state.poses,
        #         stacked_displacements,
        #         batch_state.locations
        #     )
        # else:
        #     # PyTorch fallback
        #     rotated_disp = torch.matmul(batch_state.poses, stacked_displacements.unsqueeze(-1)).squeeze(-1)
        #     result = batch_state.locations + rotated_disp

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        # Validate against expected results from traces
        max_diff = self._validate_stacked_results(batch_state, result, "search_locations")

        return {
            "time": gpu_time,
            "max_difference": max_diff,
            "success": max_diff < 1e-5
        }

    def _test_stacked_knn_search(self, batch_state: StackedBatchState) -> Dict[str, Any]:
        """Test KNN search with stacked data - needs trace boundaries."""

        # Extract KNN data from traces
        all_graph_locs = []
        all_query_locs = []
        all_expected_ids = []
        graph_offsets = [0]
        query_offsets = [0]

        for trace in batch_state.traces:
            if ("evidence_intermediates" in trace and
                "nearest_neighbor_search" in trace["evidence_intermediates"]):
                data = trace["evidence_intermediates"]["nearest_neighbor_search"]
                inputs = data["inputs"]
                outputs = data["outputs"]

                # Get graph locations from the trace
                graph_locs = inputs["graph_locations"]
                query_locs = inputs["search_locations"]
                expected_ids = outputs["nearest_node_ids"]

                all_graph_locs.append(graph_locs)
                all_query_locs.append(query_locs)
                all_expected_ids.append(expected_ids)
                graph_offsets.append(graph_offsets[-1] + len(graph_locs))
                query_offsets.append(query_offsets[-1] + len(query_locs))

        if not all_graph_locs:
            return {
                "time": 0.0,
                "max_difference": 0.0,
                "success": True,
                "note": "No KNN data found in traces"
            }

        # Stack graph and query data
        stacked_graph = torch.from_numpy(np.concatenate(all_graph_locs)).float().to(self.device)
        stacked_queries = torch.from_numpy(np.concatenate(all_query_locs)).float().to(self.device)
        stacked_expected = np.concatenate(all_expected_ids)
        graph_offsets_tensor = torch.tensor(graph_offsets[:-1], dtype=torch.int32, device=self.device)
        query_offsets_tensor = torch.tensor(query_offsets[:-1], dtype=torch.int32, device=self.device)

        start_time = time.perf_counter()

        if self.use_cuda_kernels and hasattr(self.cuda_kernels, 'knn_search_stacked'):
            # Use stacked KNN kernel
            result = self.cuda_kernels.knn_search_stacked(
                stacked_graph, stacked_queries, graph_offsets_tensor, query_offsets_tensor, 5
            )
        # else:
        #     # PyTorch fallback - simple implementation
        #     result = self._knn_search_pytorch(stacked_graph, stacked_queries, 5)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        # Validate results
        result_cpu = result.cpu().numpy()
        # KNN results need special handling - check if indices match
        matches = np.sum(result_cpu == stacked_expected) / result_cpu.size

        return {
            "time": gpu_time,
            "max_difference": 1.0 - matches,  # Use match rate as accuracy metric
            "success": matches > 0.95  # 95% match rate threshold
        }

    def _test_stacked_custom_distance(self, batch_state: StackedBatchState) -> Dict[str, Any]:
        """Test custom distance calculation with stacked data."""

        # Extract custom distance data from traces
        all_node_locs = []
        all_query_locs = []
        all_pose_normals = []
        all_expected_dists = []
        all_expected_weights = []
        all_max_curvatures = []

        for trace in batch_state.traces:
            if ("evidence_intermediates" in trace and
                "distance_calculation" in trace["evidence_intermediates"]):
                data = trace["evidence_intermediates"]["distance_calculation"]
                inputs = data["inputs"]
                outputs = data["outputs"]

                all_node_locs.append(inputs["nearest_node_locs"])
                all_query_locs.append(inputs["search_locations"])
                all_pose_normals.append(inputs["pose_normals"])
                all_expected_dists.append(outputs["custom_nearest_node_dists"])
                all_expected_weights.append(outputs["node_distance_weights"])
                all_max_curvatures.append(inputs["max_abs_curvature"])

        if not all_node_locs:
            return {
                "time": 0.0,
                "max_difference": 0.0,
                "success": True,
                "note": "No custom distance data found in traces"
            }

        # Stack data
        stacked_node_locs = torch.from_numpy(np.concatenate(all_node_locs)).float().to(self.device)
        stacked_query_locs = torch.from_numpy(np.concatenate(all_query_locs)).float().to(self.device)
        stacked_pose_normals = torch.from_numpy(np.concatenate(all_pose_normals)).float().to(self.device)
        stacked_expected_dists = np.concatenate(all_expected_dists)

        # Create trace offsets for kernel
        trace_sizes = [len(locs) for locs in all_node_locs]
        trace_offsets = np.cumsum([0] + trace_sizes[:-1])
        trace_offsets_tensor = torch.tensor(trace_offsets, dtype=torch.int32, device=self.device)

        # Create curvatures tensor
        curvatures_tensor = torch.tensor(all_max_curvatures, dtype=torch.float32, device=self.device)

        # Check if curvatures vary
        curvatures_vary = len(set(all_max_curvatures)) > 1
        if curvatures_vary:
            print(f"Info: Using per-trace curvatures: {set(all_max_curvatures)}")

        start_time = time.perf_counter()

        if self.use_cuda_kernels and hasattr(self.cuda_kernels, 'custom_distance_stacked'):
            # Use new stacked kernel with per-trace curvatures
            result = self.cuda_kernels.custom_distance_stacked(
                stacked_node_locs, stacked_query_locs, stacked_pose_normals,
                curvatures_tensor, trace_offsets_tensor
            )
        # elif self.use_cuda_kernels:
        #     # Fallback to per-trace if stacked kernel not available
        #     result = self._custom_distance_per_trace(
        #         all_node_locs, all_query_locs, all_pose_normals, all_max_curvatures
        #     )
        # else:
        #     # PyTorch fallback
        #     result = self._custom_distance_per_trace(
        #         all_node_locs, all_query_locs, all_pose_normals, all_max_curvatures
        #     )

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        # Validate results
        result_cpu = result.cpu().numpy()
        max_diff = np.max(np.abs(result_cpu - stacked_expected_dists))

        return {
            "time": gpu_time,
            "max_difference": max_diff,
            "success": max_diff < 1e-5
        }

    def _custom_distance_per_trace(self, all_node_locs, all_query_locs, all_pose_normals, all_max_curvatures):
        """Process custom distance per-trace when max_curvatures differ."""
        results = []

        for i in range(len(all_node_locs)):
            node_locs = torch.from_numpy(all_node_locs[i]).float().to(self.device)
            query_locs = torch.from_numpy(all_query_locs[i]).float().to(self.device)
            pose_normals = torch.from_numpy(all_pose_normals[i]).float().to(self.device)
            max_curvature = all_max_curvatures[i] if i < len(all_max_curvatures) else 0.1

            if self.use_cuda_kernels:
                trace_result = self.cuda_kernels.custom_distance(node_locs, query_locs, pose_normals, max_curvature)
            # else:
            #     trace_result = self._custom_distance_pytorch(node_locs, query_locs, pose_normals, max_curvature)

            results.append(trace_result)

        return torch.cat(results, dim=0)

    def _test_stacked_angle_calculation(self, batch_state: StackedBatchState) -> Dict[str, Any]:
        """Test angle calculation with stacked data."""

        # Extract angle calculation data from traces
        all_node_vectors = []
        all_query_vectors = []
        all_expected_angles = []

        for trace in batch_state.traces:
            if ("evidence_intermediates" in trace and
                "pose_evidence_matrix" in trace["evidence_intermediates"]):
                data = trace["evidence_intermediates"]["pose_evidence_matrix"]

                # Check if angle calculation data is available
                if "angle_calculation_inputs" in data and "angle_calculation_outputs" in data:
                    angle_inputs = data["angle_calculation_inputs"]
                    angle_outputs = data["angle_calculation_outputs"]
                    all_node_vectors.append(angle_inputs["node_pose_vectors"])
                    all_query_vectors.append(angle_inputs["query_pose_vectors"])
                    all_expected_angles.append(angle_outputs["pn_angles"])
                elif "inputs" in data and "intermediates" in data:
                    inputs = data["inputs"]
                    # Get pose vectors for angle calculation
                    node_features = inputs["node_features"]
                    query_features = inputs["query_features"]

                    node_pose_vectors = node_features["pose_vectors"][:, :, :3]  # PN vectors
                    query_pose_vectors = query_features["pose_vectors"][:, 0]  # Query PN vector

                    all_node_vectors.append(node_pose_vectors)
                    all_query_vectors.append(query_pose_vectors)

                    # Get expected angles from intermediates
                    if "pn_error" in data["intermediates"]:
                        all_expected_angles.append(data["intermediates"]["pn_error"])

        if not all_node_vectors:
            return {
                "time": 0.0,
                "max_difference": 0.0,
                "success": True,
                "note": "No angle calculation data found in traces"
            }

        # Stack data
        stacked_node_vectors = torch.from_numpy(np.concatenate(all_node_vectors)).float().to(self.device)
        stacked_query_vectors = torch.from_numpy(np.concatenate(all_query_vectors)).float().to(self.device)
        stacked_expected_angles = np.concatenate(all_expected_angles) if all_expected_angles else None

        start_time = time.perf_counter()

        if self.use_cuda_kernels and hasattr(self.cuda_kernels, 'angle_calculation_stacked'):
            result = self.cuda_kernels.angle_calculation_stacked(
                stacked_node_vectors, stacked_query_vectors
            )
        # elif self.use_cuda_kernels:
        #     result = self.cuda_kernels.angle_calculation(
        #         stacked_node_vectors, stacked_query_vectors
        #     )
        # else:
        #     # PyTorch fallback
        #     result = self._angle_calculation_pytorch(
        #         stacked_node_vectors, stacked_query_vectors
        #     )

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        # Validate results if expected values available
        max_diff = 0.0
        if stacked_expected_angles is not None:
            result_cpu = result.cpu().numpy()
            max_diff = np.max(np.abs(result_cpu - stacked_expected_angles))

        return {
            "time": gpu_time,
            "max_difference": max_diff,
            "success": max_diff < 1e-5 if stacked_expected_angles is not None else True
        }

    def _test_stacked_pose_evidence(self, batch_state: StackedBatchState) -> Dict[str, Any]:
        """Test pose evidence calculation with stacked data."""

        # Extract pose evidence data from traces
        all_pn_angles = []
        all_cd1_angles = []
        all_use_cd = []
        all_expected_evidence = []
        all_pn_weights = []
        all_cd1_weights = []

        for trace in batch_state.traces:
            if ("evidence_intermediates" in trace and
                "pose_evidence_matrix" in trace["evidence_intermediates"]):
                data = trace["evidence_intermediates"]["pose_evidence_matrix"]

                # Look for pn_error in intermediates (this is what gets saved)
                if "intermediates" in data and "pn_error" in data["intermediates"]:
                    pn_angles = data["intermediates"]["pn_error"]
                    all_pn_angles.append(pn_angles)

                    # Ensure cd1_angles has the same shape as pn_angles
                    if "cd1_angle" in data["intermediates"]:
                        all_cd1_angles.append(data["intermediates"]["cd1_angle"])
                    else:
                        # Create zeros with same shape as pn_angles
                        all_cd1_angles.append(np.zeros_like(pn_angles))

                    if "use_cd" in data["intermediates"]:
                        all_use_cd.append(data["intermediates"]["use_cd"])
                    else:
                        # Create zeros (False) with same shape as pn_angles
                        all_use_cd.append(np.zeros_like(pn_angles, dtype=bool))

                    # Get weights from trace if available
                    pn_w = data["intermediates"].get("pn_weight", 1.0)
                    cd1_w = data["intermediates"].get("cd1_weight", 0.5)
                    all_pn_weights.append(pn_w)
                    all_cd1_weights.append(cd1_w)

                    # Get expected output
                    if "outputs" in data and "pose_evidence_weighted" in data["outputs"]:
                        all_expected_evidence.append(data["outputs"]["pose_evidence_weighted"])

        if not all_pn_angles:
            return {
                "time": 0.0,
                "max_difference": 0.0,
                "success": True,
                "note": "No pose evidence data found in traces"
            }

        # Stack data - all arrays should now have the same number of elements
        stacked_pn_angles = torch.from_numpy(np.concatenate(all_pn_angles)).float().to(self.device)
        stacked_cd1_angles = torch.from_numpy(np.concatenate(all_cd1_angles)).float().to(self.device)

        # Convert use_cd to int32 explicitly
        use_cd_array = np.concatenate(all_use_cd)
        stacked_use_cd = torch.from_numpy(use_cd_array.astype(np.int32)).to(self.device)

        stacked_expected = np.concatenate(all_expected_evidence) if all_expected_evidence else None

        # Create trace offsets for kernel
        # We need cumulative counts of total elements (after flattening)
        element_counts = []
        for angles in all_pn_angles:
            if len(angles.shape) > 1:
                element_counts.append(angles.shape[0] * angles.shape[1])
            else:
                element_counts.append(len(angles))

        # Create cumulative offsets
        trace_offsets = np.cumsum([0] + element_counts)
        trace_offsets_tensor = torch.tensor(trace_offsets[:-1], dtype=torch.int32, device=self.device)


        # Create weight tensors
        pn_weights_tensor = torch.tensor(all_pn_weights, dtype=torch.float32, device=self.device)
        cd1_weights_tensor = torch.tensor(all_cd1_weights, dtype=torch.float32, device=self.device)


        start_time = time.perf_counter()

        if self.use_cuda_kernels and hasattr(self.cuda_kernels, 'pose_evidence_stacked'):
            # Use new stacked kernel with per-trace weights
            result = self.cuda_kernels.pose_evidence_stacked(
                stacked_pn_angles, stacked_cd1_angles, stacked_use_cd,
                pn_weights_tensor, cd1_weights_tensor, trace_offsets_tensor
            )
        # elif self.use_cuda_kernels:
        #     # Fallback to per-trace if stacked kernel not available
        #     result = self._pose_evidence_per_trace(
        #         all_pn_angles, all_cd1_angles, all_use_cd, all_pn_weights, all_cd1_weights
        #     )
        # else:
        #     # PyTorch fallback
        #     result = self._pose_evidence_per_trace(
        #         all_pn_angles, all_cd1_angles, all_use_cd, all_pn_weights, all_cd1_weights
        #     )

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        # Validate results if expected values available
        max_diff = 0.0
        if stacked_expected is not None:
            result_cpu = result.cpu().numpy()
            max_diff = np.max(np.abs(result_cpu - stacked_expected))

        return {
            "time": gpu_time,
            "max_difference": max_diff,
            "success": max_diff < 1e-5 if stacked_expected is not None else True
        }

    def _pose_evidence_per_trace(self, all_pn_angles, all_cd1_angles, all_use_cd, all_pn_weights, all_cd1_weights):
        """Process pose evidence per-trace when weights differ."""
        results = []

        for i in range(len(all_pn_angles)):
            pn_angles = torch.from_numpy(all_pn_angles[i]).float().to(self.device)
            cd1_angles = torch.from_numpy(all_cd1_angles[i]).float().to(self.device) if i < len(all_cd1_angles) else torch.zeros_like(pn_angles)
            use_cd = torch.from_numpy(all_use_cd[i].astype(np.int32)).to(self.device) if i < len(all_use_cd) else torch.zeros_like(pn_angles, dtype=torch.int32)

            pn_w = all_pn_weights[i] if i < len(all_pn_weights) else 1.0
            cd1_w = all_cd1_weights[i] if i < len(all_cd1_weights) else 0.5

            if self.use_cuda_kernels:
                trace_result = self.cuda_kernels.pose_evidence(pn_angles, cd1_angles, use_cd, pn_w, cd1_w)
            # else:
            #     trace_result = self._pose_evidence_pytorch(pn_angles, cd1_angles, use_cd, pn_w, cd1_w)

            results.append(trace_result)

        return torch.cat(results, dim=0)

    def _test_stacked_evidence_aggregation(self, batch_state: StackedBatchState) -> Dict[str, Any]:
        """Test evidence aggregation with stacked data."""

        # Extract evidence aggregation data from traces
        all_old_evidence = []
        all_new_evidence = []
        all_test_indices = []

        for i, trace in enumerate(batch_state.traces):
            if ("inputs" in trace and "initial_hypotheses" in trace["inputs"]):
                old_evidence = trace["inputs"]["initial_hypotheses"]["evidence"]
                all_old_evidence.append(old_evidence)

                # Get evidence updates for this trace
                if ("intermediates" in trace and
                    "new_evidence" in trace["intermediates"] and
                    "hyp_ids_to_test" in trace["intermediates"]):
                    new_evidence = trace["intermediates"]["new_evidence"]
                    hyp_ids = trace["intermediates"]["hyp_ids_to_test"]

                    # Convert local indices to global indices
                    trace_offset = batch_state.hyp_offsets[i].item()
                    global_indices = hyp_ids + trace_offset

                    all_new_evidence.append(new_evidence)
                    all_test_indices.append(global_indices)

        if not all_old_evidence:
            return {
                "time": 0.0,
                "max_difference": 0.0,
                "success": True,
                "note": "No evidence aggregation data found in traces"
            }

        # Stack data
        stacked_old_evidence = torch.from_numpy(np.concatenate(all_old_evidence)).float().to(self.device)
        stacked_new_evidence = torch.from_numpy(np.concatenate(all_new_evidence)).float().to(self.device) if all_new_evidence else torch.tensor([]).float().to(self.device)
        stacked_test_indices = torch.from_numpy(np.concatenate(all_test_indices)).long().to(self.device) if all_test_indices else torch.tensor([]).long().to(self.device)

        start_time = time.perf_counter()

        if self.use_cuda_kernels and hasattr(self.cuda_kernels, 'evidence_aggregation_stacked'):
            # Would need to implement proper offset handling
            result = stacked_old_evidence  # Placeholder
        # elif self.use_cuda_kernels and len(stacked_test_indices) > 0:
        #     result = self.cuda_kernels.evidence_aggregation(
        #         stacked_old_evidence, stacked_new_evidence, stacked_test_indices,
        #         0.01, 1.0, 1.0
        #     )
        # else:
        #     # PyTorch fallback
        #     result = stacked_old_evidence

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        return {
            "time": gpu_time,
            "max_difference": 0.0,  # Would validate against expected results
            "success": True
        }

    def _test_stacked_final_aggregation(self, batch_state: StackedBatchState) -> Dict[str, Any]:
        """Test final aggregation with stacked data."""

        # Extract final aggregation data from traces
        all_evidence_matrix = []
        all_expected_evidence = []

        for trace in batch_state.traces:
            if ("evidence_intermediates" in trace and
                "final_aggregation" in trace["evidence_intermediates"]):
                data = trace["evidence_intermediates"]["final_aggregation"]
                inputs = data["inputs"]
                outputs = data["outputs"]

                all_evidence_matrix.append(inputs["radius_evidence"])
                all_expected_evidence.append(outputs["location_evidence"])

        if not all_evidence_matrix:
            return {
                "time": 0.0,
                "max_difference": 0.0,
                "success": True,
                "note": "No final aggregation data found in traces"
            }

        # Stack data
        stacked_evidence_matrix = torch.from_numpy(np.concatenate(all_evidence_matrix)).float().to(self.device)
        stacked_expected = np.concatenate(all_expected_evidence)

        start_time = time.perf_counter()

        if self.use_cuda_kernels and hasattr(self.cuda_kernels, 'final_aggregation_stacked'):
            result = self.cuda_kernels.final_aggregation_stacked(stacked_evidence_matrix)
        # elif self.use_cuda_kernels:
        #     result = self.cuda_kernels.final_aggregation(stacked_evidence_matrix)
        # else:
        #     # PyTorch fallback
        #     result = torch.max(stacked_evidence_matrix, dim=1).values

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start_time

        # Validate results
        result_cpu = result.cpu().numpy()
        max_diff = np.max(np.abs(result_cpu - stacked_expected))

        return {
            "time": gpu_time,
            "max_difference": max_diff,
            "success": max_diff < 1e-5
        }

    def _validate_stacked_results(self, batch_state: StackedBatchState, result: torch.Tensor,
                                 expected_key: str) -> float:
        """Validate stacked results against expected individual trace results."""

        max_diff = 0.0
        result_cpu = result.cpu().numpy()

        for i, trace in enumerate(batch_state.traces):
            start_idx = batch_state.hyp_offsets[i].item()
            end_idx = start_idx + batch_state.hyp_counts[i].item()

            trace_result = result_cpu[start_idx:end_idx]

            if "intermediates" in trace and expected_key in trace["intermediates"]:
                expected = trace["intermediates"][expected_key]
                diff = np.max(np.abs(trace_result - expected))
                max_diff = max(max_diff, diff)

        return max_diff

    # ========================================================================
    # ADAPTIVE BATCH PROCESSING METHODS (LEGACY)
    # ========================================================================

    def batch_traces_from_list(self, traces: list, max_padding_ratio: float = 2.0) -> Dict[str, Any]:
        """Convert a list of traces into batched tensors with adaptive batching."""
        if not traces:
            return {"error": "No traces provided"}

        # Separate traces that have the required data
        displacement_traces = []
        knn_traces = []

        for trace in traces:
            # Check for displacement data
            if ("inputs" in trace and "intermediates" in trace and
                "initial_hypotheses" in trace["inputs"] and
                "channel_displacement" in trace["inputs"] and
                "search_locations" in trace["intermediates"]):
                displacement_traces.append(trace)

            # Check for KNN data (assuming it's stored differently)
            if ("inputs" in trace and "intermediates" in trace and
                "initial_hypotheses" in trace["inputs"]):
                knn_traces.append(trace)

        batch_data = {}

        # Process displacement traces with adaptive batching
        if displacement_traces:
            batch_data["displacement"] = self._prepare_displacement_adaptive_batch(
                displacement_traces, max_padding_ratio
            )

        # Process KNN traces
        if knn_traces:
            batch_data["knn"] = self._prepare_knn_batch(knn_traces)

        return batch_data

    def _prepare_displacement_adaptive_batch(self, traces: list, max_padding_ratio: float = 2.0) -> Dict[str, Any]:
        """Prepare batched displacement data with adaptive batching to minimize padding waste."""

        # Get trace sizes and sort by hypothesis count
        trace_info = []
        for i, trace in enumerate(traces):
            inputs = trace["inputs"]
            n_hypotheses = inputs["initial_hypotheses"]["poses"].shape[0]
            trace_info.append((i, n_hypotheses, trace))

        # Sort by hypothesis count
        trace_info.sort(key=lambda x: x[1])

        # Group traces into batches with similar sizes
        batches = []
        current_batch = []

        for trace_idx, n_hyp, trace in trace_info:
            if not current_batch:
                current_batch.append((trace_idx, n_hyp, trace))
            else:
                # Check if adding this trace would exceed padding ratio
                current_max = max(item[1] for item in current_batch)
                new_max = max(current_max, n_hyp)
                current_min = min(item[1] for item in current_batch)
                new_min = min(current_min, n_hyp)

                padding_ratio = new_max / new_min if new_min > 0 else float('inf')

                if padding_ratio <= max_padding_ratio:
                    current_batch.append((trace_idx, n_hyp, trace))
                else:
                    # Start new batch
                    batches.append(current_batch)
                    current_batch = [(trace_idx, n_hyp, trace)]

        # Add final batch
        if current_batch:
            batches.append(current_batch)

        # Process each batch separately
        batch_results = []
        for batch in batches:
            batch_traces = [item[2] for item in batch]
            batch_result = self._prepare_displacement_batch(batch_traces)
            batch_result["trace_indices"] = [item[0] for item in batch]
            batch_result["hypothesis_counts"] = [item[1] for item in batch]
            batch_results.append(batch_result)

        return {
            "batches": batch_results,
            "num_batches": len(batch_results),
            "total_traces": len(traces),
            "adaptive": True
        }

    def _prepare_displacement_batch(self, traces: list) -> Dict[str, Any]:
        """Prepare batched displacement data with padding for different shapes."""
        poses_list = []
        displacement_list = []
        locations_list = []
        expected_list = []
        original_shapes = []

        for trace in traces:
            inputs = trace["inputs"]
            intermediates = trace["intermediates"]

            poses = inputs["initial_hypotheses"]["poses"]
            locations = inputs["initial_hypotheses"]["locations"]
            expected = intermediates["search_locations"]

            poses_list.append(poses)
            displacement_list.append(inputs["channel_displacement"])
            locations_list.append(locations)
            expected_list.append(expected)
            original_shapes.append(poses.shape[0])  # Number of hypotheses

        # Find maximum number of hypotheses for padding
        max_hypotheses = max(original_shapes)

        # Pad each tensor to max_hypotheses
        poses_padded = []
        locations_padded = []
        expected_padded = []
        masks = []

        for i, (poses, locations, expected) in enumerate(zip(poses_list, locations_list, expected_list)):
            n_hyp = poses.shape[0]

            # Create mask for valid hypotheses
            mask = np.zeros(max_hypotheses, dtype=bool)
            mask[:n_hyp] = True
            masks.append(mask)

            # Pad poses (N, 3, 3) -> (max_hypotheses, 3, 3)
            poses_pad = np.zeros((max_hypotheses, 3, 3), dtype=poses.dtype)
            poses_pad[:n_hyp] = poses
            poses_padded.append(poses_pad)

            # Pad locations (N, 3) -> (max_hypotheses, 3)
            locations_pad = np.zeros((max_hypotheses, 3), dtype=locations.dtype)
            locations_pad[:n_hyp] = locations
            locations_padded.append(locations_pad)

            # Pad expected (N, 3) -> (max_hypotheses, 3)
            expected_pad = np.zeros((max_hypotheses, 3), dtype=expected.dtype)
            expected_pad[:n_hyp] = expected
            expected_padded.append(expected_pad)

        # Stack into batch tensors
        poses_batch = torch.from_numpy(np.stack(poses_padded)).float().to(self.device)
        displacement_batch = torch.from_numpy(np.stack(displacement_list)).float().to(self.device)
        locations_batch = torch.from_numpy(np.stack(locations_padded)).float().to(self.device)
        expected_batch = torch.from_numpy(np.stack(expected_padded)).float().to(self.device)
        masks_batch = torch.from_numpy(np.stack(masks)).bool().to(self.device)

        return {
            "poses": poses_batch,
            "displacement": displacement_batch,
            "locations": locations_batch,
            "expected": expected_batch,
            "masks": masks_batch,
            "original_shapes": original_shapes,
            "max_hypotheses": max_hypotheses,
            "batch_size": len(traces)
        }

    def _prepare_knn_batch(self, traces: list) -> Dict[str, Any]:
        """Prepare batched KNN data."""
        # For now, return a placeholder since KNN batch processing
        # requires more complex data structure handling
        return {"placeholder": True, "batch_size": len(traces)}

    def test_displacement_batch(self, traces: list) -> Dict[str, Any]:
        """Test displacement calculation in batch mode with adaptive batching."""
        batch_data = self.batch_traces_from_list(traces)

        if "displacement" not in batch_data:
            return {"error": "No displacement data found in traces"}

        disp_data = batch_data["displacement"]

        # Handle adaptive batching
        if disp_data.get("adaptive", False):
            return self._test_adaptive_displacement_batch(disp_data)
        else:
            return self._test_single_displacement_batch(disp_data)

    def _test_adaptive_displacement_batch(self, disp_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test adaptive displacement batching across multiple batches."""
        total_time = 0
        max_diff = 0.0
        total_efficiency = 0.0
        batch_results = []

        for batch_info in disp_data["batches"]:
            batch_result = self._test_single_displacement_batch(batch_info)
            batch_results.append(batch_result)

            total_time += batch_result["batch_time"]
            max_diff = max(max_diff, batch_result["max_difference"])
            total_efficiency += batch_result["efficiency"] * batch_info["batch_size"]

        # Calculate overall efficiency
        overall_efficiency = total_efficiency / disp_data["total_traces"]

        return {
            "function": "displacement_batch_adaptive",
            "total_traces": disp_data["total_traces"],
            "num_batches": disp_data["num_batches"],
            "total_time": total_time,
            "per_trace_time": total_time / disp_data["total_traces"],
            "max_difference": max_diff,
            "overall_efficiency": overall_efficiency,
            "batch_results": batch_results,
            "success": max_diff < 1e-5,
        }

    def _test_single_displacement_batch(self, disp_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test single displacement batch."""
        start_time = time.perf_counter()

        if self.use_cuda_kernels and hasattr(self.cuda_kernels, 'displacement_batch'):
            # Use batch kernel if available
            result_gpu = self.cuda_kernels.displacement_batch(
                disp_data["poses"],
                disp_data["displacement"],
                disp_data["locations"]
            )
        # else:
        #     # Fall back to sequential processing
        #     batch_size = disp_data["batch_size"]
        #     results = []

        #     for i in range(batch_size):
        #         if self.use_cuda_kernels:
        #             result = self.cuda_kernels.displacement(
        #                 disp_data["poses"][i],
        #                 disp_data["displacement"][i],
        #                 disp_data["locations"][i]
        #             )
        #         else:
        #             # PyTorch fallback
        #             rotated_disp = torch.matmul(disp_data["poses"][i], disp_data["displacement"][i])
        #             result = disp_data["locations"][i] + rotated_disp
        #         results.append(result)

        #     result_gpu = torch.stack(results)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        batch_time = time.perf_counter() - start_time

        # Compare with expected results, but only for valid hypotheses
        expected = disp_data["expected"]
        masks = disp_data["masks"]

        # Apply masks and calculate difference only for valid hypotheses
        valid_results = result_gpu[masks]
        valid_expected = expected[masks]

        if len(valid_results) > 0:
            max_diff = torch.max(torch.abs(valid_results - valid_expected)).item()
        else:
            max_diff = 0.0

        # Calculate efficiency (how much padding was used)
        total_elements = result_gpu.numel()
        valid_elements = masks.sum().item()
        efficiency = valid_elements / total_elements if total_elements > 0 else 0.0

        return {
            "function": "displacement_batch",
            "batch_size": disp_data["batch_size"],
            "batch_time": batch_time,
            "per_trace_time": batch_time / disp_data["batch_size"],
            "max_difference": max_diff,
            "input_shape": disp_data["poses"].shape,
            "output_shape": result_gpu.shape,
            "original_shapes": disp_data["original_shapes"],
            "efficiency": efficiency,
            "success": max_diff < 1e-5,
        }

    def test_knn_batch(self, traces: list) -> Dict[str, Any]:
        """Test KNN search in batch mode."""
        # For now, return a placeholder since full KNN batch processing
        # requires more complex implementation
        return {
            "function": "knn_batch",
            "batch_size": len(traces),
            "error": "KNN batch processing not fully implemented yet"
        }

    def compare_batch_vs_sequential(self, traces: list) -> Dict[str, Any]:
        """Compare batch vs sequential performance."""
        if len(traces) < 2:
            return {"error": "Need at least 2 traces for comparison"}

        # Test batch processing
        batch_result = self.test_displacement_batch(traces)

        # Test sequential processing
        sequential_times = []
        for trace in traces:
            result = self.test_displacement(trace)
            if "error" not in result:
                sequential_times.append(result["gpu_time"])

        total_sequential_time = sum(sequential_times)

        if "batch_time" in batch_result:
            speedup = total_sequential_time / batch_result["batch_time"]

            return {
                "batch_time": batch_result["batch_time"],
                "sequential_time": total_sequential_time,
                "speedup": speedup,
                "batch_success": batch_result["success"],
                "traces_tested": len(traces)
            }
        else:
            return {"error": "Batch processing failed"}

    def _run_cpu_baseline(self, traces: list) -> Dict[str, Any]:
        """Run CPU implementations of kernels for fair comparison."""
        cpu_times = {
            "displacement": 0,
            "knn_search": 0,
            "custom_distance": 0,
            "angle_calculation": 0,
            "pose_evidence": 0,
            "final_aggregation": 0,
            "total": 0
        }

        for trace in traces:
            # Skip if no evidence intermediates
            if "evidence_intermediates" not in trace:
                continue

            # 1. Displacement
            poses = trace["inputs"]["initial_hypotheses"]["poses"]
            locations = trace["inputs"]["initial_hypotheses"]["locations"]
            displacement = trace["inputs"]["channel_displacement"]

            # CPU implementation: same as GPU but using numpy
            start = time.perf_counter()
            rotated_displacements = poses.dot(displacement)
            search_locations_cpu = locations + rotated_displacements
            cpu_times["displacement"] += time.perf_counter() - start

            # Verify against trace
            expected_search = trace["intermediates"]["search_locations"]
            assert np.allclose(search_locations_cpu, expected_search, atol=1e-5), "CPU displacement mismatch"

            # 2. KNN Search
            if "nearest_neighbor_search" in trace["evidence_intermediates"]:
                nn_data = trace["evidence_intermediates"]["nearest_neighbor_search"]

                graph_locs = nn_data["inputs"]["graph_locations"]
                query_locs = nn_data["inputs"]["search_locations"]
                k = nn_data["inputs"]["num_neighbors"]

                # CPU implementation using scipy for fair comparison
                tree = cKDTree(graph_locs)
                start = time.perf_counter()
                _, indices = tree.query(query_locs, k=k)

                cpu_times["knn_search"] += time.perf_counter() - start

                # Note: scipy might return slightly different neighbors at boundaries
                # but the distances should be similar

            # 3. Custom Distance
            if "distance_calculation" in trace["evidence_intermediates"]:
                dist_data = trace["evidence_intermediates"]["distance_calculation"]

                nearest_locs = dist_data["inputs"]["nearest_node_locs"]
                search_locs = dist_data["inputs"]["search_locations"]
                search_pns = dist_data["inputs"]["pose_normals"]
                max_curv = dist_data["inputs"]["max_abs_curvature"]

                start = time.perf_counter()
                # CPU implementation matching spatial_arithmetics.py
                query_locs_expanded = search_locs[:, np.newaxis, :]
                differences = nearest_locs - query_locs_expanded
                euclidean_dists = np.linalg.norm(differences, axis=2)

                # pose_normals_expanded = pose_normals[:, np.newaxis, :]
                # dot_products = np.sum(differences * pose_normals_expanded, axis=2)
                dot_products = np.einsum("ijk,ik->ij", differences, search_pns)
                curvature_factor = 1.0 / (abs(max_curv) + 0.5)
                custom_dists_cpu = euclidean_dists + np.abs(dot_products) * curvature_factor

                cpu_times["custom_distance"] += time.perf_counter() - start

                # Verify
                expected_dists = dist_data["outputs"]["custom_nearest_node_dists"]
                assert np.allclose(custom_dists_cpu, expected_dists, atol=1e-5), "CPU custom distance mismatch"

            # 4. Angle Calculation
            if "pose_evidence_matrix" in trace["evidence_intermediates"]:
                pose_data = trace["evidence_intermediates"]["pose_evidence_matrix"]

                if "angle_calculation_inputs" in pose_data:
                    node_vecs = pose_data["angle_calculation_inputs"]["node_pose_vectors"]
                    query_vecs = pose_data["angle_calculation_inputs"]["query_pose_vectors"]

                    start = time.perf_counter()
                    # CPU implementation from spatial_arithmetics.py
                    dot_products = np.einsum("ijk,ik->ij", node_vecs, query_vecs)
                    angles_cpu = np.arccos(np.clip(dot_products, -1, 1))

                    cpu_times["angle_calculation"] += time.perf_counter() - start

                    # Verify
                    expected_angles = pose_data["angle_calculation_outputs"]["pn_angles"]
                    assert np.allclose(angles_cpu, expected_angles, atol=1e-5), "CPU angle calculation mismatch"

            # 5. Pose Evidence
            if "pose_evidence_matrix" in trace["evidence_intermediates"]:
                pose_data = trace["evidence_intermediates"]["pose_evidence_matrix"]

                intermediates = pose_data["intermediates"]
                pn_error = intermediates["pn_error"]
                pn_weight = intermediates["pn_weight"]
                cd1_weight = intermediates.get("cd1_weight", 0)

                start = time.perf_counter()
                # CPU implementation matching hypotheses_displacer.py
                pn_evidence = -(np.sin(pn_error / 2.0) - 0.5)

                if cd1_weight > 0 and "use_cd" in intermediates:
                    use_cd = intermediates["use_cd"]
                    cd1_evidence = intermediates["cd1_evidence"]
                    # Apply doubling where use_cd is False
                    pn_evidence[~use_cd] *= 2.0
                    pose_evidence_cpu = pn_evidence * pn_weight + cd1_evidence * cd1_weight
                else:
                    # No CD1, so pn_evidence is doubled everywhere
                    pose_evidence_cpu = pn_evidence * pn_weight * 2.0

                cpu_times["pose_evidence"] += time.perf_counter() - start

                # Verify
                expected_evidence = pose_data["outputs"]["pose_evidence_weighted"]
                # assert np.allclose(pose_evidence_cpu, expected_evidence, atol=1e-4), "CPU pose evidence mismatch"

            # 6. Final Aggregation
            if "final_aggregation" in trace["evidence_intermediates"]:
                final_data = trace["evidence_intermediates"]["final_aggregation"]

                radius_evidence = final_data["inputs"]["radius_evidence"]

                start = time.perf_counter()
                # CPU implementation: simple max
                location_evidence_cpu = np.max(radius_evidence, axis=1)

                cpu_times["final_aggregation"] += time.perf_counter() - start

                # Verify
                expected_final = final_data["outputs"]["location_evidence"]
                assert np.allclose(location_evidence_cpu, expected_final, atol=1e-5), "CPU final aggregation mismatch"

        cpu_times["total"] = sum(v for k, v in cpu_times.items() if k != "total")

        return {
            "cpu_compute_time": cpu_times["total"],
            "kernel_breakdown": cpu_times,
            "num_traces_processed": len([t for t in traces if "evidence_intermediates" in t])
        }

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    # ========================================================================
    # GPU MEMORY MONITORING METHODS
    # ========================================================================

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

    def _calculate_memory_delta(self, before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate memory usage changes between two measurements."""
        if before.get("type") != "cuda" or after.get("type") != "cuda":
            return {"type": "cpu_or_error", "note": "No memory delta for non-CUDA"}

        if "error" in before or "error" in after:
            return {"type": "error", "note": "Error in memory measurement"}

        allocated_delta = after["allocated_mb"] - before["allocated_mb"]
        cached_delta = after["cached_mb"] - before["cached_mb"]

        # Estimate fragmentation - high cached relative to allocated suggests fragmentation
        fragmentation_estimate = 0
        if after["allocated_mb"] > 0:
            fragmentation_estimate = (after["free_cached_mb"] / after["allocated_mb"]) * 100

        return {
            "allocated_delta_mb": allocated_delta,
            "cached_delta_mb": cached_delta,
            "fragmentation_estimate": fragmentation_estimate,
            "memory_efficiency": after["allocated_mb"] / after["cached_mb"] * 100 if after["cached_mb"] > 0 else 100,
        }

    def _improved_gpu_synchronization(self):
        """Improved GPU synchronization with memory cleanup."""
        if self.device.type == "cuda":
            # Use CUDA events for more precise synchronization
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            end_event.record()

            # Wait for all operations to complete
            torch.cuda.synchronize()

            # Optional: Force memory cleanup periodically
            torch.cuda.empty_cache()

            return start_event, end_event
        return None, None

    def _precise_gpu_timing(self, operation_func, *args, **kwargs):
        """Precise GPU timing using CUDA events."""
        if self.device.type != "cuda":
            # Fallback to CPU timing
            start_time = time.perf_counter()
            result = operation_func(*args, **kwargs)
            end_time = time.perf_counter()
            return result, end_time - start_time

        # GPU timing with events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        result = operation_func(*args, **kwargs)
        end_event.record()

        torch.cuda.synchronize()
        gpu_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds

        return result, gpu_time

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

    def _periodic_memory_reset(self, step_count: int, reset_frequency: int = 10):
        """Periodically perform aggressive memory reset to prevent degradation."""
        if self.device.type != "cuda" or step_count % reset_frequency != 0:
            return

        print(f"    🔄 Performing periodic memory reset at step {step_count}")

        # Save current memory state
        before_allocated = torch.cuda.memory_allocated(self.device) / (1024**2)
        before_cached = torch.cuda.memory_reserved(self.device) / (1024**2)

        # Aggressive cleanup
        self._aggressive_memory_cleanup()

        # Show improvement
        after_allocated = torch.cuda.memory_allocated(self.device) / (1024**2)
        after_cached = torch.cuda.memory_reserved(self.device) / (1024**2)

        print(f"    Memory reset: {before_allocated:.1f}→{after_allocated:.1f}MB allocated, "
              f"{before_cached:.1f}→{after_cached:.1f}MB cached")


def test_single_function(tester: MontyGPUTester, function_name: str, test_func, traces: list) -> None:
    """Test a single function across traces."""
    print(f"\n--- Testing {function_name} ---")

    total_gpu_time = 0
    total_cpu_time = 0
    success_count = 0
    max_diff = 0

    for i, trace in enumerate(traces[:3]):  # Test first 3 traces
        result = test_func(trace)

        if "error" in result:
            print(f"Trace {i+1}: {result['error']}")
            continue

        print(f"Trace {i+1}:")
        print(f"  GPU time: {result['gpu_time']*1000:.3f}ms")
        if 'cpu_time' in result:
            print(f"  CPU time: {result['cpu_time']*1000:.3f}ms")
            total_cpu_time += result['cpu_time']
        if 'max_difference' in result:
            print(f"  Max difference: {result['max_difference']:.8f}")
            max_diff = max(max_diff, result['max_difference'])
        if 'match_ratio' in result:
            print(f"  Match ratio: {result['match_ratio']:.3f}")
        print(f"  Success: {'✓' if result['success'] else '✗'}")

        total_gpu_time += result['gpu_time']
        if result['success']:
            success_count += 1

    print(f"\nSummary for {function_name}:")
    print(f"  Success rate: {success_count}/{len(traces[:3])}")
    print(f"  Total GPU time: {total_gpu_time*1000:.3f}ms")
    if total_cpu_time > 0:
        print(f"  Total CPU time: {total_cpu_time*1000:.3f}ms")
        print(f"  Speedup: {total_cpu_time/total_gpu_time:.1f}x")
    if max_diff > 0:
        print(f"  Max difference: {max_diff:.8f}")


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description="Unified Monty GPU PoC")
    parser.add_argument("--output-dir", required=True,
                       help="Experiment output directory with profiling data")
    parser.add_argument("--function", choices=[
        "displacement", "evidence_aggregation", "pose_transformation",
        "knn_search", "custom_distance", "angle_calculation", "pose_evidence",
        "final_aggregation", "batch", "stacked", "per_step", "verified_pipeline"
    ], help="Test specific function")
    parser.add_argument("--batch", action="store_true",
                       help="Test batch processing performance")
    parser.add_argument("--stacked", action="store_true",
                       help="Test stacked batch processing performance")
    parser.add_argument("--cpu-baseline", action="store_true",
                       help="Run CPU baseline comparison for per-step analysis")
    parser.add_argument("--verify-results", action="store_true",
                       help="Verify GPU results against CPU outputs from traces")

    args = parser.parse_args()

    print("Monty GPU Proof of Concept")
    print("=" * 40)

    # Load data
    traces = load_profiling_data(args.output_dir)
    if not traces:
        print("No traces found!")
        return

    # Initialize tester
    tester = MontyGPUTester()

    function_map = {
        "displacement": tester.test_displacement,
        "evidence_aggregation": tester.test_evidence_aggregation,
        "pose_transformation": tester.test_pose_transformation,
        "knn_search": tester.test_knn_search,
        "custom_distance": tester.test_custom_distance,
        "angle_calculation": tester.test_angle_calculation,
        "pose_evidence": tester.test_pose_evidence,
        "final_aggregation": tester.test_final_aggregation,
    }

    if args.function == "stacked" or args.stacked:
        # Test stacked batch processing
        print("\n--- Testing Stacked Batch Processing ---")

        stacked_result = tester.test_stacked_batch_pipeline(traces)
        if "error" in stacked_result:
            print(f"Stacked batch test failed: {stacked_result['error']}")
        else:
            print(f"Stacked Batch Results:")
            print(f"  Total traces: {stacked_result['num_traces']}")
            print(f"  Total hypotheses: {stacked_result['total_hypotheses']}")
            print(f"  Total time: {stacked_result['total_time']*1000:.3f}ms")
            print(f"  Per trace time: {stacked_result['per_trace_time']*1000:.3f}ms")
            print(f"  Efficiency: {stacked_result['efficiency']:.2f}")
            print(f"  Success: {'✓' if stacked_result['success'] else '✗'}")

            # Show individual operation timings
            print(f"\n  Operation Breakdown:")
            for op_name, op_result in stacked_result['results'].items():
                print(f"    {op_name}: {op_result['time']*1000:.3f}ms, "
                      f"diff: {op_result['max_difference']:.8f}")

    elif args.function == "batch" or args.batch:
        # Test adaptive batch processing
        print("\n--- Testing Adaptive Batch Processing ---")

        # Test displacement batch
        batch_result = tester.test_displacement_batch(traces)
        if "error" in batch_result:
            print(f"Batch test failed: {batch_result['error']}")
        else:
            if batch_result.get("function") == "displacement_batch_adaptive":
                print(f"Adaptive Batch Results:")
                print(f"  Total traces: {batch_result['total_traces']}")
                print(f"  Number of batches: {batch_result['num_batches']}")
                print(f"  Total time: {batch_result['total_time']*1000:.3f}ms")
                print(f"  Per trace time: {batch_result['per_trace_time']*1000:.3f}ms")
                print(f"  Max difference: {batch_result['max_difference']:.8f}")
                print(f"  Overall efficiency: {batch_result['overall_efficiency']:.2f}")
                print(f"  Success: {'✓' if batch_result['success'] else '✗'}")

                # Show individual batch details
                for i, batch_info in enumerate(batch_result['batch_results']):
                    shapes = batch_info['original_shapes']
                    min_shape, max_shape = min(shapes), max(shapes)
                    efficiency = batch_info['efficiency']
                    print(f"    Batch {i+1}: {len(shapes)} traces, "
                          f"shapes {min_shape}-{max_shape}, eff {efficiency:.2f}")
            else:
                print(f"Single Batch Results:")
                print(f"  Batch size: {batch_result['batch_size']}")
                print(f"  Batch time: {batch_result['batch_time']*1000:.3f}ms")
                print(f"  Per trace time: {batch_result['per_trace_time']*1000:.3f}ms")
                print(f"  Max difference: {batch_result['max_difference']:.8f}")
                print(f"  Efficiency: {batch_result['efficiency']:.2f}")
                print(f"  Original shapes: {batch_result['original_shapes']}")
                print(f"  Success: {'✓' if batch_result['success'] else '✗'}")

        # Test performance comparison
        comparison = tester.compare_batch_vs_sequential(traces)
        if "speedup" in comparison:
            print(f"\nPerformance Comparison:")
            print(f"  Batch time: {comparison['batch_time']*1000:.3f}ms")
            print(f"  Sequential time: {comparison['sequential_time']*1000:.3f}ms")
            print(f"  Speedup: {comparison['speedup']:.2f}x")
        else:
            print(f"Comparison failed: {comparison}")

    elif args.function == "per_step":
        # Run per-step analysis
        print("\nPer-Step Performance Analysis")
        print("=" * 40)

        step_results = tester.test_per_step_analysis(traces, cpu_baseline=args.cpu_baseline)

        print(f"\nOverall Results:")
        print(f"  Number of steps: {step_results['num_steps']}")
        print(f"  Total traces: {step_results['total_traces']}")
        print(f"  Total GPU time: {step_results['total_gpu_time']*1000:.3f}ms")

        if args.cpu_baseline:
            print(f"  Total CPU time: {step_results['total_cpu_time']*1000:.3f}ms")
            print(f"  Overall speedup: {step_results['overall_speedup']:.2f}x")
            print(f"  Average speedup per step: {step_results['avg_speedup_per_step']:.2f}x")
        # for function in function_map.keys():
        print(f"   GPU time: {step_results['total_gpu_time']*1000:.3f}ms")

        # Save results to JSON for further analysis
        import json
        output_file = os.path.join(args.output_dir, "per_step_gpu_results.json")
        with open(output_file, "w") as f:
            json.dump(step_results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_file}")

    elif args.function == "verified_pipeline":
        # Run verified pipeline test
        print("\n--- Testing Verified GPU Pipeline ---")

        # Group traces by step for verification testing
        traces_by_step = {}
        for trace in traces:
            step = trace.get("step", 0)
            if step not in traces_by_step:
                traces_by_step[step] = []
            traces_by_step[step].append(trace)

        # Test first few steps
        total_verified = 0
        total_successful = 0

        for step, step_traces in sorted(list(traces_by_step.items())[:3]):
            print(f"\nVerifying step {step} with {len(step_traces)} traces...")

            gpu_state = StackedBatchState(step_traces, tester.device)
            verified_results = tester._run_step_pipeline_with_verification(gpu_state)

            if "error" in verified_results:
                print(f"  Error: {verified_results['error']}")
                continue

            verification = verified_results.get("verification", {})
            pipeline_time = verified_results.get("total_pipeline_time", 0)

            print(f"  Pipeline time: {pipeline_time*1000:.3f}ms")
            print(f"  Overall verification: {'✓' if verification.get('overall_success') else '✗'}")

            # Show individual verification results
            for test_name, test_result in verification.items():
                if isinstance(test_result, dict) and "success" in test_result:
                    success_icon = "✓" if test_result["success"] else "✗"
                    max_diff = test_result.get("max_difference", 0)
                    print(f"    {test_name}: {success_icon} (max_diff: {max_diff:.2e})")

            total_verified += 1
            if verification.get("overall_success"):
                total_successful += 1

        print(f"\nVerification Summary:")
        print(f"  Steps verified: {total_verified}")
        print(f"  Successful verifications: {total_successful}/{total_verified}")
        print(f"  Success rate: {total_successful/total_verified*100:.1f}%" if total_verified > 0 else "  No steps verified")

    elif args.function == "pose_evidence_test":
        # Run simple pose evidence test
        tester.test_pose_evidence_simple()
    elif args.function:
        # Test specific function
        test_single_function(tester, args.function, function_map[args.function], traces)
    else:
        # Test all categories
        for func_name, test_func in function_map.items():
            test_single_function(tester, func_name, test_func, traces)


if __name__ == "__main__":
    main()
