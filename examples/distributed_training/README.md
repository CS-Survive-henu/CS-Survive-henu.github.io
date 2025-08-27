# Distributed Training Examples

This directory contains examples and best practices for distributed deep learning training, specifically addressing common DDP (Distributed Data Parallel) issues.

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run single GPU training** (safest option):
   ```bash
   python ddp_training.py --single_gpu --batch_size 32 --epochs 5
   ```

3. **Run multi-GPU training**:
   ```bash
   torchrun --nproc_per_node=2 ddp_training.py --batch_size 16 --epochs 5
   ```

4. **Use launch script for common configurations**:
   ```bash
   ./launch_examples.sh  # Show available options
   ./launch_examples.sh 1  # Run single GPU config
   ```

## 📋 Contents

- **`ddp_training.py`** - Robust multi-GPU DDP training script that fixes common issues:
  - ✅ RuntimeError about unfinished reduction in prior iteration
  - ✅ DataLoader worker crashes 
  - ✅ Parameter indices not receiving gradients for certain ranks
  - ✅ Memory management and cleanup
  - ✅ Comprehensive error handling and logging
  - ✅ Single-GPU fallback mode

- **`requirements.txt`** - Dependencies for the training scripts
- **`DDP_TROUBLESHOOTING.md`** - Comprehensive guide to common DDP issues and solutions
- **`launch_examples.sh`** - Example launch commands for different configurations
- **`test_ddp_training.py`** - Test script to validate functionality

## 🔧 Key Features

### Addresses Specific DDP Issues

1. **"Expected to have finished reduction in the prior iteration"**
   - Proper gradient synchronization
   - Consistent forward passes across ranks
   - Parameter usage tracking

2. **DataLoader worker crashes**
   - Safe multiprocessing configuration
   - Memory management
   - Automatic worker count optimization
   - Fallback to single-threaded loading

3. **Parameter gradient issues**
   - Unused parameter detection
   - Consistent model execution across ranks
   - Proper DDP configuration

### Production-Ready Features

- **Error Recovery**: Continues training on recoverable errors
- **Memory Monitoring**: Tracks GPU and system memory usage
- **Comprehensive Logging**: Detailed logs for debugging
- **Flexible Configuration**: Extensive command-line options
- **Fallback Modes**: Single-GPU and CPU-only training
- **Performance Optimization**: Various optimizations for stability and speed

## 🛠️ Usage Examples

### Basic Multi-GPU Training
```bash
# Train on 2 GPUs
torchrun --nproc_per_node=2 ddp_training.py \
    --batch_size 32 \
    --epochs 10 \
    --learning_rate 1e-4

# Train on 4 GPUs with smaller batch size
torchrun --nproc_per_node=4 ddp_training.py \
    --batch_size 16 \
    --epochs 10
```

### Debug Mode
```bash
# Enable detailed logging and debugging
torchrun --nproc_per_node=2 ddp_training.py \
    --debug \
    --log_interval 10 \
    --batch_size 16
```

### Conservative Mode (For Unstable Environments)
```bash
# Use safer settings to avoid crashes
torchrun --nproc_per_node=2 ddp_training.py \
    --find_unused_parameters \
    --num_workers 0 \
    --batch_size 8 \
    --continue_on_error
```

### CPU-Only Training
```bash
# For testing without GPUs
python ddp_training.py \
    --cpu_only \
    --batch_size 8 \
    --epochs 2
```

## 🧪 Testing

Run the test suite to validate functionality:
```bash
python test_ddp_training.py
```

## 📖 Common Issues and Solutions

See `DDP_TROUBLESHOOTING.md` for detailed solutions to:
- DDP synchronization errors
- DataLoader crashes
- Memory issues
- Performance problems
- Debugging techniques

## 🔍 Environment Variables for Debugging

```bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
```

## 📚 Educational Purpose

This implementation serves as a comprehensive example for students learning about:
- Distributed deep learning
- DDP best practices
- Error handling in ML training
- Production-ready ML code
- Performance optimization techniques

The code is heavily commented and includes extensive error handling to help understand both what can go wrong and how to fix it.