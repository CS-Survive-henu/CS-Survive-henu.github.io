# Robust Distributed Data Parallel (DDP) Training

This directory contains a production-ready solution for multi-GPU training using PyTorch's Distributed Data Parallel (DDP) with comprehensive error handling and debugging capabilities.

## 🚀 Features

### Core Fixes
- ✅ **Unused Parameters**: Handles `RuntimeError: Expected to have finished reduction in the prior iteration` 
- ✅ **DataLoader Crashes**: Robust worker process management with crash protection
- ✅ **Gradient Synchronization**: Proper parameter gradient handling across all ranks
- ✅ **Memory Management**: Advanced memory monitoring and cleanup

### Production-Ready Features
- 🔧 **Multi-GPU & Single-GPU Support**: Automatic fallback modes
- 📊 **Comprehensive Logging**: Rank-aware logging with detailed debugging
- 🛡️ **Error Recovery**: Graceful error handling and recovery mechanisms
- 💾 **Checkpointing**: Automatic model and training state saving
- 🎯 **Memory Optimization**: AMP (Automatic Mixed Precision) support
- 📈 **Monitoring**: Real-time memory and performance tracking

## 📁 Files

- `ddp_training.py` - Main DDP training script with all fixes
- `launch_training.sh` - Convenient launch script for different configurations
- `requirements.txt` - Python dependencies
- `README.md` - This documentation

## 🔧 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Basic Usage

```bash
# Auto-detect GPUs and start training
./launch_training.sh

# Specify GPUs manually
./launch_training.sh --gpus 0,1,2,3

# CPU training (fallback mode)
./launch_training.sh --cpu

# Quick test run
./launch_training.sh --test
```

### 3. Python API Usage

```python
from ddp_training import DDPTrainer

config = {
    'batch_size': 32,
    'learning_rate': 1e-4,
    'num_epochs': 10,
    'use_amp': True,
}

# Single-GPU or CPU
trainer = DDPTrainer(rank=0, world_size=1, config=config)
trainer.train()
```

## 🛠️ Advanced Configuration

### Training Parameters

```python
config = {
    # Model architecture
    'vocab_size': 10000,
    'hidden_size': 512,
    'num_layers': 6,
    'num_heads': 8,
    'dropout': 0.1,
    
    # Training settings
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 0.01,
    'num_epochs': 10,
    'use_amp': True,  # Automatic Mixed Precision
    
    # Data handling
    'dataset_size': 10000,
    'seq_len': 128,
    'num_workers': 4,
    'pin_memory': True,
    
    # Monitoring and logging
    'log_interval': 10,
    'save_interval': 5,
    'memory_check_interval': 50,
    'cleanup_interval': 100,
    'max_errors_per_epoch': 5,
    
    # Resource limits
    'max_memory_gb': 16.0,
    'target_loss': 0.01,
}
```

### Launch Script Options

```bash
./launch_training.sh [OPTIONS]

Options:
  -g, --gpus GPUS         Comma-separated GPU IDs (e.g., 0,1,2,3)
  -b, --batch-size SIZE   Batch size per GPU (default: 32)
  -e, --epochs EPOCHS     Number of training epochs (default: 10)
  -l, --log-dir DIR       Log directory (default: logs)
  -c, --checkpoint-dir DIR Checkpoint directory (default: checkpoints)
  --cpu                   Force CPU training
  --test                  Quick test with small dataset
  --debug                 Enable debug mode with verbose logging
  -h, --help              Show help message
```

## 🏗️ Architecture Overview

### Core Components

1. **DDPLogger**: Rank-aware logging system
   - Separate log files per rank
   - Console output only from rank 0
   - Structured logging with timestamps

2. **MemoryManager**: Advanced memory monitoring
   - CPU and GPU memory tracking
   - Automatic cleanup and garbage collection
   - Memory usage warnings and limits

3. **SafeDataLoader**: Robust data loading
   - Worker crash protection
   - Proper distributed sampling
   - Timeout handling and recovery

4. **DDPTrainer**: Main training orchestrator
   - DDP initialization and cleanup
   - Error handling and recovery
   - Checkpointing and monitoring

### DDP Fixes Implementation

#### 1. Unused Parameters Fix
```python
model = DDP(
    model,
    device_ids=[rank] if torch.cuda.is_available() else None,
    find_unused_parameters=True,  # ✅ Handle unused parameters
    broadcast_buffers=True,       # ✅ Sync buffers across ranks
    bucket_cap_mb=25,            # ✅ Optimize communication
    gradient_as_bucket_view=True  # ✅ Memory optimization
)
```

#### 2. DataLoader Worker Protection
```python
dataloader = DataLoader(
    dataset,
    num_workers=adjusted_workers,    # ✅ Scale workers per world_size
    persistent_workers=True,         # ✅ Reuse worker processes
    timeout=30,                      # ✅ Worker timeout protection
    worker_init_fn=worker_init_fn,   # ✅ Proper random seed setup
)
```

#### 3. Gradient Synchronization
```python
try:
    loss.backward()
    optimizer.step()
except RuntimeError as e:
    if "Expected to have finished reduction" in str(e):
        # ✅ Emergency recovery protocol
        torch.cuda.synchronize()
        optimizer.zero_grad()
        dist.barrier()
        # Continue or re-raise based on severity
```

## 🔍 Debugging and Monitoring

### Log Analysis

Each rank generates its own log file in the logs directory:
```
logs/
├── rank_0.log
├── rank_1.log
├── rank_2.log
└── rank_3.log
```

### Memory Monitoring

The script automatically tracks and logs:
- CPU memory usage (RSS)
- GPU memory allocation per device
- Memory warnings when limits exceeded
- Periodic cleanup operations

### Error Recovery

The trainer implements several recovery mechanisms:
1. **Gradient Recovery**: Clear gradients and synchronize on DDP errors
2. **Memory Recovery**: Aggressive cleanup on memory errors
3. **Worker Recovery**: Restart DataLoader workers on crashes
4. **Process Recovery**: Graceful shutdown on fatal errors

## 🎯 Common Issues and Solutions

### Issue 1: "Expected to have finished reduction"
**Root Cause**: DDP gradient synchronization mismatch between ranks

**Solution**: 
- Set `find_unused_parameters=True` in DDP wrapper
- Ensure consistent model execution across ranks
- Use `drop_last=True` in DataLoader for consistent batch sizes

### Issue 2: DataLoader Worker Crashes
**Root Cause**: Memory issues, signal handling, or resource contention

**Solution**:
- Limit workers per world size: `num_workers = min(workers, cpu_count // world_size)`
- Use `persistent_workers=True` to reuse processes
- Set appropriate timeouts and worker initialization

### Issue 3: Parameter Gradient Sync Issues
**Root Cause**: Some model parameters don't receive gradients in certain ranks

**Solution**:
- Use conditional model branches carefully
- Set `broadcast_buffers=True` for buffer synchronization
- Implement proper error recovery with barrier synchronization

### Issue 4: Memory Leaks in Multi-GPU Training
**Root Cause**: Accumulating gradients, cached tensors, or improper cleanup

**Solution**:
- Regular memory cleanup with `torch.cuda.empty_cache()`
- Gradient accumulation with proper clearing
- Process cleanup on exit with signal handlers

## 🧪 Testing

### Quick Test
```bash
# Run a 2-epoch test with small batch size
./launch_training.sh --test
```

### Debug Mode
```bash
# Enable verbose logging and blocking CUDA calls
./launch_training.sh --debug
```

### Memory Stress Test
```bash
# Test with larger model and monitor memory usage
./launch_training.sh --batch-size 64 --epochs 5
```

## 📈 Performance Tips

1. **Batch Size**: Scale batch size with number of GPUs for optimal throughput
2. **Workers**: Use `num_workers = 4 * num_gpus` as starting point
3. **Memory**: Enable AMP for memory efficiency: `use_amp=True`
4. **Communication**: Tune `bucket_cap_mb` for your model size
5. **Monitoring**: Regular memory cleanup prevents OOM errors

## 🤝 Contributing

This script serves as an educational resource for the CS-Survive-henu community. Contributions are welcome:

1. Report issues with specific error messages and configurations
2. Submit improvements for additional edge cases
3. Add support for more model architectures
4. Enhance monitoring and visualization capabilities

## 📚 References

- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Distributed Training Best Practices](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [Memory Management in PyTorch](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [NCCL Troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html)

---

**Note**: This implementation is designed for educational purposes and production use. Always test thoroughly with your specific model and data before deploying to production environments.