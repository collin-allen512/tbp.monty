#!/bin/bash

# Unified build script for all Monty CUDA kernels

echo "Building Monty CUDA kernels"
echo ""

# Check if we're in the right directory
if [ ! -f "setup_all.py" ]; then
    echo "Error: Must be run from the directory containing gpu_kernels/"
    exit 1
fi

# Check CUDA installation
echo "🔍 Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    nvcc --version | head -4
else
    echo "⚠️  CUDA compiler not found - building CPU-only version"
fi

# Check PyTorch CUDA support
echo ""
echo "🔍 Checking PyTorch CUDA support..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'Device {i}: {torch.cuda.get_device_name(i)}')
" 2>/dev/null || echo "⚠️  PyTorch not available"

# Clean previous builds
echo ""
echo "🧹 Cleaning previous builds..."
rm -rf build/
rm -rf *.egg-info/
rm -rf *.so
rm -rf __pycache__/

# Build extensions
echo ""
echo "🔨 Building all Monty CUDA extensions..."
python setup_all.py build_ext --inplace

# Check if build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build completed successfully!"

    # List built files
    echo ""
    echo "📦 Built files:"
    ls -la *.so 2>/dev/null || echo "No .so files found"

    # Test the installation
    echo ""
    echo "🧪 Testing installation..."
    python -c "
import sys
sys.path.insert(0, '.')
import monty_cuda
print('✅ Successfully imported monty_cuda')
print('Available functions:')
for func in dir(monty_cuda):
    if not func.startswith('_'):
        print(f'  - {func}')
"

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ All tests passed!"
        echo ""
        echo "🎉 Monty CUDA kernels are ready!"
        echo ""
        echo ""
        echo "You can now use:"
        echo "  python test_monty_cuda.py"
        echo "  python gpu_monty_poc.py --output-dir /path/to/experiment"
    else
        echo ""
        echo "❌ Import test failed"
        echo ""
        echo "🔍 Running diagnostic..."
        cd ..
        python debug_cuda_import.py
        exit 1
    fi
else
    echo "❌ Build failed"
    exit 1
fi

# Navigate back
cd ..

echo ""
echo "🚀 Ready for GPU development!"