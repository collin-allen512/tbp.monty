# Monty GPU Acceleration Guide

A complete guide to profiling, developing, and testing GPU acceleration for Monty hypothesis updates.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Profiling System](#profiling-system)
3. [GPU Development](#gpu-development)
4. [Testing](#testing)
5. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 🚀 Build and Test GPU Kernels
```bash
# Build all GPU kernels
./build_monty_cuda.sh

# Test basic functionality
python test_monty_cuda.py

# Test with real experiment data
python gpu_monty_poc.py --output-dir /path/to/experiment
```

### 📊 Profile an Experiment
```python
# Add to your experiment config
learning_module_args = dict(
    enable_hypothesis_profiling=True,
    hypotheses_updater_args=dict(
        save_computation_trace=True,  # For GPU verification
    ),
)
```

---

## Profiling System

### Enable Profiling

**For Hypothesis Updates:**
```python
learning_module_args = dict(
    enable_hypothesis_profiling=True,
    hypotheses_updater_args=dict(
        save_computation_trace=True,  # Optional: saves detailed data for GPU testing
    ),
)
```

### Understanding Profiling Output

After running an experiment with profiling enabled, you'll find:

```
experiment_output/
├── hypothesis_profiling_results.json     # Timing breakdown
├── hypothesis_computation_trace.pkl      # Detailed computation data (if enabled)
└── knn_profiling_results.json           # KNN-specific timing
```

### Analyze Results

```python
# Quick analysis
python -c "
import json
with open('experiment_output/hypothesis_profiling_results.json') as f:
    data = json.load(f)

for op, stats in data['displacer'].items():
    if 'total_time' in stats:
        print(f'{op}: {stats[\"total_time\"]:.2f}s')
"
```

---

## GPU Development

### Available GPU Functions

The unified `monty_cuda` module provides:

**Hypothesis Updates:**
- `displacement(poses, displacement, locations)` - Calculate new locations
- `evidence_aggregation(old_evidence, new_evidence, indices, ...)` - Combine evidence

**Evidence Calculations:**
- `pose_transformation(pose_vectors, reference_poses)` - Rotate pose vectors
- `knn_search(graph_locations, query_locations, k)` - Find nearest neighbors
- `custom_distance(...)` - Curvature-modulated distances
- `angle_calculation(...)` - Compute pose angles
- `pose_evidence(...)` - Convert angles to evidence
- `radius_evidence_max(evidence_matrix)` - Max reduction

### Implementation Structure

```
gpu_kernels/
├── monty_cuda_kernels.cu    # All CUDA implementations
├── monty_cpu_only.cpp       # CPU fallbacks
└── setup_all.py             # Build configuration
```

### Adding a New GPU Function

1. **Add CUDA kernel** to `monty_cuda_kernels.cu`:
```cuda
template<typename scalar_t>
__global__ void my_kernel(
    const scalar_t* input,
    scalar_t* output,
    int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Your GPU implementation
    output[idx] = process(input[idx]);
}
```

2. **Add C++ wrapper**:
```cpp
torch::Tensor my_function_cuda(torch::Tensor input) {
    // Setup and launch kernel
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "my_function", ([&] {
        my_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N
        );
    }));
    return output;
}
```

3. **Add CPU fallback** to `monty_cpu_only.cpp`

4. **Add Python binding**:
```cpp
m.def("my_function", &my_function, "My function description");
```

5. **Test**:
```bash
./build_monty_cuda.sh
python test_monty_cuda.py
```

---

## Testing

### Basic Kernel Testing

```bash
# Test all kernels
python test_monty_cuda.py
```

This tests:
- Module import
- Individual function correctness
- GPU vs CPU comparison
- Basic performance benchmarks

### Real Data Testing

```bash
# Test with experiment data
python gpu_monty_poc.py --output-dir /path/to/experiment

# Test specific categories
python gpu_monty_poc.py --output-dir /path/to/experiment --category hypothesis
python gpu_monty_poc.py --output-dir /path/to/experiment --category evidence

# Test specific functions
python gpu_monty_poc.py --output-dir /path/to/experiment --function knn_search
python gpu_monty_poc.py --output-dir /path/to/experiment --function displacement
```
---

## Troubleshooting

### Build Issues

**"nvcc not found"**
```bash
# Install CUDA toolkit or check PATH
export PATH=/usr/local/cuda/bin:$PATH
```

**"PyTorch not found"**
```bash
# Install PyTorch with CUDA support
pip install torch
```

### Import Issues

**"No module named 'monty_cuda'"**
```bash
# Rebuild for your Python version
./fix_cuda_import.sh

# Or manually:
cd gpu_kernels
python setup_all.py build_ext --inplace
```

**"libtorch_cpu.so not found"**
```bash
# Set library path
export LD_LIBRARY_PATH=$(python -c 'import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):$LD_LIBRARY_PATH
```

### Runtime Issues

**"expected scalar type Int but found Long"**
- Ensure indices are `long` tensors: `indices = indices.long()`

**"tensors used as indices must be long"**
- Convert index tensors: `test_indices.to(torch.long)`

**Poor GPU performance**
- Check GPU utilization: `nvidia-smi`
- Ensure tensors are on GPU: `tensor.cuda()`
- Use `torch.cuda.synchronize()` for accurate timing

### Getting Help

1. **Run diagnostics**:
```bash
python debug_cuda_import.py
```

2. **Check profiling results** to ensure you're optimizing the right functions

3. **Verify correctness** before optimizing performance

---

This guide provides everything you need to profile, develop, and test GPU acceleration for Monty. Start with profiling to identify bottlenecks, then use the unified GPU development system to implement and test optimizations.