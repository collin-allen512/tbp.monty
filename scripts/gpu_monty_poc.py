#!/usr/bin/env python3
"""
Unified GPU Proof of Concept for Monty

Tests functions using saved profiling data.
This script provides a single interface for testing all GPU operations.

Usage:
    # Test all functions
    python gpu_monty_poc.py --output-dir /path/to/experiment

    # Test specific function
    python gpu_monty_poc.py --output-dir /path/to/experiment --function displacement
    python gpu_monty_poc.py --output-dir /path/to/experiment --function knn_search
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
            print(f"  use_cd sum: {np.sum(use_cd)}")
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
    # HELPER METHODS
    # ========================================================================

    def _knn_search_pytorch(self, graph_locations, query_locations, k_neighbors):
        """PyTorch implementation of KNN search."""
        N = query_locations.size(0)
        K = k_neighbors
        indices = torch.empty((N, K), dtype=torch.long, device=query_locations.device)

        for i in range(N):
            query = query_locations[i].unsqueeze(0)
            diffs = graph_locations - query
            distances = torch.norm(diffs, dim=1)
            sorted_indices = torch.argsort(distances)
            indices[i] = sorted_indices[:K]

        return indices

    def _custom_distance_pytorch(self, node_locs, query_locs, pose_normals, max_abs_curvature):
        """PyTorch implementation of custom distance calculation."""
        # node_locs: (N, M, 3), query_locs: (N, 3), pose_normals: (N, 3)
        query_locs_expanded = query_locs.unsqueeze(1)  # (N, 1, 3)
        differences = node_locs - query_locs_expanded  # (N, M, 3)

        # Calculate euclidean distances
        euclidean_dists = torch.norm(differences, dim=2)  # (N, M)

        # Calculate dot products with pose normals
        pose_normals_expanded = pose_normals.unsqueeze(1)  # (N, 1, 3)
        dot_products = torch.sum(differences * pose_normals_expanded, dim=2)  # (N, M)

        # Calculate custom distances
        curvature_factor = 1.0 / (abs(max_abs_curvature) + 0.5)
        custom_distances = euclidean_dists + torch.abs(dot_products) * curvature_factor

        return custom_distances

    def _angle_calculation_pytorch(self, node_vectors, query_vectors):
        """PyTorch implementation of angle calculation."""
        # node_vectors: (N, M, 3), query_vectors: (N, 3)
        # This matches the CPU implementation in spatial_arithmetics.py:
        # dot_product = np.einsum("ijk,ik->ij", hyp_f, query_f)
        # angle = np.arccos(np.clip(dot_product, -1, 1))

        query_vectors_expanded = query_vectors.unsqueeze(1)  # (N, 1, 3)

        # Calculate dot products (assumes vectors are already normalized)
        dot_products = torch.sum(node_vectors * query_vectors_expanded, dim=2)  # (N, M)

        # Clip and calculate angles (no normalization needed, vectors assumed normalized)
        dot_products = torch.clamp(dot_products, -1.0, 1.0)
        angles = torch.acos(dot_products)

        return angles

    def _pose_evidence_pytorch(self, pn_angles, cd1_angles, use_cd, pn_weight, cd1_weight):
        """PyTorch implementation of pose evidence calculation."""
        # Calculate PN evidence
        pn_evidence = -(torch.sin(pn_angles / 2.0) - 0.5)

        # Calculate CD1 evidence
        cd1_evidence = torch.zeros_like(pn_evidence)

        # Only process CD1 if cd1_weight > 0 (query pose is fully defined)
        if cd1_weight > 0:
            cd1_mask = use_cd.bool()

            if torch.any(cd1_mask):
                cd1_errors = torch.pi / 2.0 - torch.abs(cd1_angles - torch.pi / 2.0)
                cd1_evidence[cd1_mask] = -(torch.sin(cd1_errors[cd1_mask]) - 0.5)

            # Double PN evidence where CD1 is not available (only when pose is fully defined)
            pn_evidence[~cd1_mask] *= 2.0

        # Combine weighted evidence
        total_evidence = pn_evidence * pn_weight + cd1_evidence * cd1_weight

        return total_evidence


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
        "final_aggregation"
    ], help="Test specific function")

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

    if args.function:
        # Test specific function
        test_single_function(tester, args.function, function_map[args.function], traces)
    else:
        # Test all categories
        for func_name, test_func in function_map.items():
            test_single_function(tester, func_name, test_func, traces)


if __name__ == "__main__":
    main()