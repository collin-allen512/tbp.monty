#!/usr/bin/env python3
"""Unified test script for all Monty CUDA kernels.
Tests both hypothesis update and evidence calculation functions.
"""

import os
import sys
import time

import numpy as np
import torch

# Add gpu_kernels to path
gpu_kernels_dir = os.path.join(os.path.dirname(__file__), "../gpu_kernels")
sys.path.insert(0, gpu_kernels_dir)

def test_monty_cuda_import():
    """Test importing the unified monty_cuda module."""
    print("🔧 Testing monty_cuda import...")

    try:
        import monty_cuda
        print("✅ Successfully imported monty_cuda")
        functions = [f for f in dir(monty_cuda) if not f.startswith("_")]
        print(f"Available functions: {functions}")
        return monty_cuda
    except ImportError as e:
        print(f"❌ Failed to import monty_cuda: {e}")
        print("Run './build_monty_cuda.sh' to build the kernels")
        return None

def test_hypothesis_update_functions(monty_cuda):
    """Test hypothesis update functions (displacement and evidence_aggregation)."""
    print("\n🧪 Testing Hypothesis Update Functions...")

    if not monty_cuda:
        print("❌ Cannot test - module not available")
        return False

    try:
        # Test displacement
        print("  Testing displacement...")
        N = 100
        poses = torch.randn(N, 3, 3).float()
        displacement = torch.randn(3).float()
        locations = torch.randn(N, 3).float()

        if torch.cuda.is_available():
            poses_gpu = poses.cuda()
            displacement_gpu = displacement.cuda()
            locations_gpu = locations.cuda()

            result_gpu = monty_cuda.displacement(poses_gpu, displacement_gpu, locations_gpu)

            # Compare with PyTorch
            expected = locations + torch.matmul(poses, displacement)
            result_cpu = result_gpu.cpu()
            diff = torch.max(torch.abs(result_cpu - expected))

            print(f"    Max difference vs PyTorch: {diff:.8f}")
            displacement_ok = diff < 1e-5
            print(f"    {'✅' if displacement_ok else '❌'} Displacement test")
        else:
            result = monty_cuda.displacement(poses, displacement, locations)
            expected = locations + torch.matmul(poses, displacement)
            diff = torch.max(torch.abs(result - expected))
            displacement_ok = diff < 1e-5
            print(f"    {'✅' if displacement_ok else '❌'} Displacement test (CPU)")

        # Test evidence aggregation
        print("  Testing evidence_aggregation...")
        N = 50
        M = 10
        old_evidence = torch.randn(N).float().abs()
        new_evidence = torch.randn(M).float().abs()
        test_indices = torch.randint(0, N, (M,)).long()

        if torch.cuda.is_available():
            old_evidence_gpu = old_evidence.cuda()
            new_evidence_gpu = new_evidence.cuda()
            test_indices_gpu = test_indices.cuda()

            result_gpu = monty_cuda.evidence_aggregation(
                old_evidence_gpu, new_evidence_gpu, test_indices_gpu, 0.1, 0.8, 0.2
            )

            # Compare with PyTorch
            evidence_to_add = torch.full_like(old_evidence, 0.1)
            evidence_to_add[test_indices] = new_evidence
            expected = old_evidence * 0.8 + evidence_to_add * 0.2

            result_cpu = result_gpu.cpu()
            diff = torch.max(torch.abs(result_cpu - expected))

            print(f"    Max difference vs PyTorch: {diff:.8f}")
            evidence_ok = diff < 1e-5
            print(f"    {'✅' if evidence_ok else '❌'} Evidence aggregation test")
        else:
            result = monty_cuda.evidence_aggregation(
                old_evidence, new_evidence, test_indices, 0.1, 0.8, 0.2
            )
            evidence_to_add = torch.full_like(old_evidence, 0.1)
            evidence_to_add[test_indices] = new_evidence
            expected = old_evidence * 0.8 + evidence_to_add * 0.2
            diff = torch.max(torch.abs(result - expected))
            evidence_ok = diff < 1e-5
            print(f"    {'✅' if evidence_ok else '❌'} Evidence aggregation test (CPU)")

        return displacement_ok and evidence_ok

    except Exception as e:
        print(f"❌ Hypothesis update test failed: {e}")
        return False

def test_evidence_calculation_functions(monty_cuda):
    """Test evidence calculation functions."""
    print("\n🧪 Testing Evidence Calculation Functions...")

    if not monty_cuda:
        print("❌ Cannot test - module not available")
        return False

    results = {}

    try:
        # Test pose transformation
        print("  Testing pose_transformation...")
        N = 30
        pose_vectors = torch.randn(N, 3, 3).float()
        reference_poses = torch.randn(N, 3, 3).float()

        if torch.cuda.is_available():
            pose_vectors_gpu = pose_vectors.cuda()
            reference_poses_gpu = reference_poses.cuda()
            result_gpu = monty_cuda.pose_transformation(pose_vectors_gpu, reference_poses_gpu)
            expected = torch.matmul(reference_poses, pose_vectors)
            diff = torch.max(torch.abs(result_gpu.cpu() - expected))
            print(f"    Max difference: {diff:.8f}")
            results["pose_transformation"] = diff < 1e-5
        else:
            result = monty_cuda.pose_transformation(pose_vectors, reference_poses)
            expected = torch.matmul(reference_poses, pose_vectors)
            diff = torch.max(torch.abs(result - expected))
            results["pose_transformation"] = diff < 1e-5

        print(f"    {'✅' if results['pose_transformation'] else '❌'} Pose transformation")

        # Test KNN search
        print("  Testing knn_search...")
        N_queries = 20
        N_nodes = 100
        K = 5

        graph_locations = torch.randn(N_nodes, 3).float()
        query_locations = torch.randn(N_queries, 3).float()

        if torch.cuda.is_available():
            graph_locations_gpu = graph_locations.cuda()
            query_locations_gpu = query_locations.cuda()
            result_gpu = monty_cuda.knn_search(graph_locations_gpu, query_locations_gpu, K)

            # Basic validation - check shape and range
            knn_ok = (result_gpu.shape == (N_queries, K) and
                     torch.all(result_gpu >= 0) and
                     torch.all(result_gpu < N_nodes))
            print(f"    Shape: {result_gpu.shape}, valid indices: {knn_ok}")
            results["knn_search"] = knn_ok
        else:
            result = monty_cuda.knn_search(graph_locations, query_locations, K)
            knn_ok = (result.shape == (N_queries, K) and
                     torch.all(result >= 0) and
                     torch.all(result < N_nodes))
            results["knn_search"] = knn_ok

        print(f"    {'✅' if results['knn_search'] else '❌'} KNN search")

        # Test radius evidence max
        print("  Testing radius_evidence_max...")
        N = 25
        K = 4
        evidence_matrix = torch.randn(N, K).float()

        if torch.cuda.is_available():
            evidence_matrix_gpu = evidence_matrix.cuda()
            result_gpu = monty_cuda.radius_evidence_max(evidence_matrix_gpu)
            expected = torch.max(evidence_matrix, dim=1).values
            diff = torch.max(torch.abs(result_gpu.cpu() - expected))
            print(f"    Max difference: {diff:.8f}")
            results["radius_evidence_max"] = diff < 1e-6
        else:
            result = monty_cuda.radius_evidence_max(evidence_matrix)
            expected = torch.max(evidence_matrix, dim=1).values
            diff = torch.max(torch.abs(result - expected))
            results["radius_evidence_max"] = diff < 1e-6

        print(f"    {'✅' if results['radius_evidence_max'] else '❌'} Radius evidence max")

        # Test custom distance calculation
        print("  Testing custom_distance...")
        try:
            N, M = 5, 3
            node_locs = torch.randn(N, M, 3).float()
            query_locs = torch.randn(N, 3).float()
            pose_normals = torch.randn(N, 3).float()
            # Normalize pose normals
            pose_normals = pose_normals / torch.norm(pose_normals, dim=1, keepdim=True)
            max_abs_curvature = 0.5

            if torch.cuda.is_available():
                node_locs_gpu = node_locs.cuda()
                query_locs_gpu = query_locs.cuda()
                pose_normals_gpu = pose_normals.cuda()

                result_gpu = monty_cuda.custom_distance(node_locs_gpu, query_locs_gpu, pose_normals_gpu, max_abs_curvature)

                # Test CPU vs GPU consistency
                cpu_result = monty_cuda.custom_distance(node_locs, query_locs, pose_normals, max_abs_curvature)
                diff = torch.max(torch.abs(cpu_result - result_gpu.cpu()))

                print(f"    Result shape: {result_gpu.shape}")
                print(f"    CPU vs GPU max diff: {diff:.6f}")
                results["custom_distance"] = result_gpu.shape == (N, M) and diff < 1e-5
            else:
                result = monty_cuda.custom_distance(node_locs, query_locs, pose_normals, max_abs_curvature)
                results["custom_distance"] = result.shape == (N, M)

            print(f"    {'✅' if results['custom_distance'] else '❌'} Custom distance test")
        except Exception as e:
            print(f"    ❌ Custom distance test failed: {e}")
            results["custom_distance"] = False

        # Test angle calculation
        print("  Testing angle_calculation...")
        try:
            N, M = 5, 3
            node_vectors = torch.randn(N, M, 3).float()
            query_vectors = torch.randn(N, 3).float()
            # Normalize vectors
            node_vectors = node_vectors / torch.norm(node_vectors, dim=2, keepdim=True)
            query_vectors = query_vectors / torch.norm(query_vectors, dim=1, keepdim=True)

            if torch.cuda.is_available():
                node_vectors_gpu = node_vectors.cuda()
                query_vectors_gpu = query_vectors.cuda()

                result_gpu = monty_cuda.angle_calculation(node_vectors_gpu, query_vectors_gpu)

                # Test CPU vs GPU consistency
                cpu_result = monty_cuda.angle_calculation(node_vectors, query_vectors)
                diff = torch.max(torch.abs(cpu_result - result_gpu.cpu()))

                # Verify results are in valid range [0, π]
                in_range = torch.all(result_gpu >= 0) and torch.all(result_gpu <= np.pi)

                print(f"    Result shape: {result_gpu.shape}")
                print(f"    Result range: {result_gpu.min():.3f} to {result_gpu.max():.3f}")
                print(f"    Expected range: 0 to π ({np.pi:.3f})")
                print(f"    CPU vs GPU max diff: {diff:.6f}")
                results["angle_calculation"] = result_gpu.shape == (N, M) and in_range and diff < 1e-5
            else:
                result = monty_cuda.angle_calculation(node_vectors, query_vectors)
                in_range = torch.all(result >= 0) and torch.all(result <= np.pi)
                results["angle_calculation"] = result.shape == (N, M) and in_range

            print(f"    {'✅' if results['angle_calculation'] else '❌'} Angle calculation test")
        except Exception as e:
            print(f"    ❌ Angle calculation test failed: {e}")
            results["angle_calculation"] = False

        # Test pose evidence calculation
        print("  Testing pose_evidence...")
        try:
            N, M = 5, 3
            pn_angles = torch.rand(N, M).float() * np.pi  # Angles between 0 and π
            cd1_angles = torch.rand(N, M).float() * np.pi
            use_cd = torch.randint(0, 2, (N, M), dtype=torch.int32)  # Random boolean as int
            pn_weight = 0.8
            cd1_weight = 0.2

            if torch.cuda.is_available():
                pn_angles_gpu = pn_angles.cuda()
                cd1_angles_gpu = cd1_angles.cuda()
                use_cd_gpu = use_cd.cuda()

                result_gpu = monty_cuda.pose_evidence(pn_angles_gpu, cd1_angles_gpu, use_cd_gpu, pn_weight, cd1_weight)

                # Test CPU vs GPU consistency
                cpu_result = monty_cuda.pose_evidence(pn_angles, cd1_angles, use_cd, pn_weight, cd1_weight)
                diff = torch.max(torch.abs(cpu_result - result_gpu.cpu()))

                print(f"    Result shape: {result_gpu.shape}")
                print(f"    Result range: {result_gpu.min():.3f} to {result_gpu.max():.3f}")
                print(f"    CPU vs GPU max diff: {diff:.6f}")
                results["pose_evidence"] = result_gpu.shape == (N, M) and diff < 1e-5
            else:
                result = monty_cuda.pose_evidence(pn_angles, cd1_angles, use_cd, pn_weight, cd1_weight)
                results["pose_evidence"] = result.shape == (N, M)

            print(f"    {'✅' if results['pose_evidence'] else '❌'} Pose evidence test")
        except Exception as e:
            print(f"    ❌ Pose evidence test failed: {e}")
            results["pose_evidence"] = False

        return all(results.values())

    except Exception as e:
        print(f"❌ Evidence calculation test failed: {e}")
        return False

def benchmark_functions(monty_cuda):
    """Benchmark key functions to show GPU speedup."""
    print("\n⚡ Performance Benchmarks...")

    if not torch.cuda.is_available():
        print("CUDA not available - skipping benchmarks")
        return

    if not monty_cuda:
        print("Module not available - skipping benchmarks")
        return

    try:
        # Benchmark displacement (hypothesis updates)
        print("  Displacement (1000 hypotheses):")
        N = 1000
        poses = torch.randn(N, 3, 3).float()
        displacement = torch.randn(3).float()
        locations = torch.randn(N, 3).float()

        # CPU timing
        start = time.perf_counter()
        for _ in range(10):
            result_cpu = monty_cuda.displacement(poses, displacement, locations)
        cpu_time = (time.perf_counter() - start) / 10

        # GPU timing
        poses_gpu = poses.cuda()
        displacement_gpu = displacement.cuda()
        locations_gpu = locations.cuda()

        # Warmup
        for _ in range(5):
            _ = monty_cuda.displacement(poses_gpu, displacement_gpu, locations_gpu)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(10):
            result_gpu = monty_cuda.displacement(poses_gpu, displacement_gpu, locations_gpu)
        torch.cuda.synchronize()
        gpu_time = (time.perf_counter() - start) / 10

        speedup = cpu_time / gpu_time
        print(f"    CPU: {cpu_time*1000:.3f}ms, GPU: {gpu_time*1000:.3f}ms, Speedup: {speedup:.1f}x")

        # Benchmark KNN search (biggest bottleneck)
        print("  KNN Search (100 queries, 1000 nodes, k=5):")
        N_queries = 100
        N_nodes = 1000
        K = 5

        graph_locations = torch.randn(N_nodes, 3).float()
        query_locations = torch.randn(N_queries, 3).float()

        # CPU timing
        start = time.perf_counter()
        result_cpu = monty_cuda.knn_search(graph_locations, query_locations, K)
        cpu_time = time.perf_counter() - start

        # GPU timing
        graph_locations_gpu = graph_locations.cuda()
        query_locations_gpu = query_locations.cuda()

        # Warmup
        _ = monty_cuda.knn_search(graph_locations_gpu, query_locations_gpu, K)
        torch.cuda.synchronize()

        start = time.perf_counter()
        result_gpu = monty_cuda.knn_search(graph_locations_gpu, query_locations_gpu, K)
        torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start

        speedup = cpu_time / gpu_time
        print(f"    CPU: {cpu_time*1000:.3f}ms, GPU: {gpu_time*1000:.3f}ms, Speedup: {speedup:.1f}x")

    except Exception as e:
        print(f"Benchmark failed: {e}")

def main():
    """Run all tests."""
    print("Monty CUDA Kernel Test Suite")
    print("=" * 50)

    # Test module import
    monty_cuda = test_monty_cuda_import()

    if monty_cuda is None:
        print("\n❌ Cannot test kernels - import failed")
        return False

    # Test hypothesis update functions
    hypothesis_ok = test_hypothesis_update_functions(monty_cuda)

    # Test evidence calculation functions
    evidence_ok = test_evidence_calculation_functions(monty_cuda)

    # Run benchmarks
    benchmark_functions(monty_cuda)

    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"{'✅' if hypothesis_ok else '❌'} Hypothesis Update Functions")
    print(f"{'✅' if evidence_ok else '❌'} Evidence Calculation Functions")

    all_passed = hypothesis_ok and evidence_ok
    print(f"\nOverall: {'✅ All tests passed!' if all_passed else '❌ Some tests failed'}")

    if all_passed:
        print("\n🎉 Monty CUDA kernels are working correctly!")
        print("You can now use them in your experiments.")

    return all_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
