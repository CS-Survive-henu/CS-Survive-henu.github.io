# DDP Training Troubleshooting Guide

This guide addresses common issues encountered during Distributed Data Parallel (DDP) training and provides solutions.

## Common Issues and Solutions

### 1. RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one

**Problem**: This error occurs when gradients from different ranks are not properly synchronized.

**Causes**:
- Inconsistent forward passes across ranks (different code paths)
- Unused parameters in some ranks but not others
- Improper gradient accumulation
- Model not in sync across ranks

**Solutions**:

1. **Ensure consistent forward passes**:
   ```python
   # Bad: Conditional execution that differs across ranks
   if rank == 0:
       output = model(input)
   else:
       output = model(input, extra_param=True)
   
   # Good: Consistent execution across all ranks
   output = model(input)
   ```

2. **Use `find_unused_parameters=True` (temporary fix)**:
   ```python
   model = DDP(model, find_unused_parameters=True)
   ```
   Note: This is slower and should be avoided in production.

3. **Proper gradient accumulation**:
   ```python
   # Ensure accumulation steps are consistent across ranks
   for i, batch in enumerate(dataloader):
       loss = model(batch)
       loss = loss / accumulation_steps
       loss.backward()
       
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

4. **Force synchronization**:
   ```python
   # Add periodic barriers to ensure synchronization
   if step % 100 == 0:
       torch.distributed.barrier()
   ```

### 2. DataLoader worker (pid XXXX) is killed by signal: Aborted

**Problem**: DataLoader workers crash due to memory issues, shared memory problems, or multiprocessing conflicts.

**Causes**:
- Insufficient shared memory (`/dev/shm`)
- Too many workers relative to available memory
- Multiprocessing context issues
- Memory leaks in dataset loading

**Solutions**:

1. **Adjust number of workers**:
   ```python
   # Calculate optimal workers to avoid memory issues
   cpu_count = psutil.cpu_count(logical=False)
   max_workers = min(cpu_count, 8)  # Cap at 8
   num_workers = max_workers // world_size if distributed else max_workers
   ```

2. **Use safer multiprocessing context**:
   ```python
   import multiprocessing as mp
   dataloader = DataLoader(
       dataset,
       num_workers=4,
       multiprocessing_context=mp.get_context('spawn'),  # Safer than 'fork'
       persistent_workers=True,
       timeout=30
   )
   ```

3. **Increase shared memory**:
   ```bash
   # On Linux, increase shared memory size
   sudo mount -o remount,size=8G /dev/shm
   
   # Or use environment variable
   export CUDA_LAUNCH_BLOCKING=1
   ```

4. **Reduce memory usage**:
   ```python
   # Enable memory mapping and reduce prefetch
   dataloader = DataLoader(
       dataset,
       pin_memory=False,  # Disable if memory is limited
       prefetch_factor=1,  # Reduce from default 2
       drop_last=True  # Ensure consistent batch sizes
   )
   ```

5. **Alternative: Use single-threaded loading**:
   ```python
   # Fallback to single-threaded if workers keep crashing
   dataloader = DataLoader(dataset, num_workers=0)
   ```

### 3. Parameter indices which did not receive grad for certain ranks

**Problem**: Some parameters don't receive gradients on certain ranks, causing DDP synchronization issues.

**Causes**:
- Conditional model execution
- Dynamic model structures
- Unused parameters in certain forward passes
- Inconsistent batch processing

**Solutions**:

1. **Track parameter usage**:
   ```python
   class ModelWithTracking(nn.Module):
       def __init__(self):
           super().__init__()
           self._used_parameters = set()
       
       def forward(self, x):
           # Track which parameters are used
           self._used_parameters.add('layer1')
           return self.layer1(x)
       
       def get_unused_parameters(self):
           all_params = set(name for name, _ in self.named_parameters())
           unused = all_params - self._used_parameters
           self._used_parameters.clear()
           return list(unused)
   ```

2. **Ensure all parameters are used**:
   ```python
   # Force all parameters to be used by adding dummy operations
   def forward(self, x):
       output = self.main_forward(x)
       
       # Ensure all parameters receive gradients
       dummy_loss = sum(p.sum() * 0 for p in self.parameters())
       return output + dummy_loss
   ```

3. **Use consistent model execution**:
   ```python
   # Avoid rank-dependent logic
   # Bad:
   if torch.distributed.get_rank() == 0:
       use_extra_layer = True
   
   # Good: Make decisions based on input data, not rank
   use_extra_layer = (batch['sequence_length'] > threshold)
   ```

### 4. CUDA out of memory errors

**Problem**: GPU memory exhaustion during distributed training.

**Solutions**:

1. **Gradient checkpointing**:
   ```python
   from torch.utils.checkpoint import checkpoint
   
   class MemoryEfficientModel(nn.Module):
       def forward(self, x):
           return checkpoint(self.expensive_layer, x)
   ```

2. **Mixed precision training**:
   ```python
   from torch.cuda.amp import autocast, GradScaler
   
   scaler = GradScaler()
   
   for batch in dataloader:
       with autocast():
           loss = model(batch)
       
       scaler.scale(loss).backward()
       scaler.step(optimizer)
       scaler.update()
   ```

3. **Reduce batch size**:
   ```python
   # Automatically reduce batch size on OOM
   try:
       loss = model(batch)
   except RuntimeError as e:
       if "out of memory" in str(e):
           torch.cuda.empty_cache()
           # Reduce batch size and retry
           smaller_batch = {k: v[:len(v)//2] for k, v in batch.items()}
           loss = model(smaller_batch)
   ```

### 5. Deadlocks and hanging processes

**Problem**: Training hangs or deadlocks, especially during initialization or checkpointing.

**Solutions**:

1. **Add timeouts**:
   ```python
   torch.distributed.init_process_group(
       backend='nccl',
       timeout=datetime.timedelta(seconds=300)  # 5 minute timeout
   )
   ```

2. **Synchronize critical sections**:
   ```python
   # Ensure all ranks reach the same point
   torch.distributed.barrier()
   
   if rank == 0:
       # Only rank 0 performs this operation
       save_checkpoint()
   
   torch.distributed.barrier()
   ```

3. **Proper cleanup**:
   ```python
   def cleanup():
       if torch.distributed.is_initialized():
           torch.distributed.destroy_process_group()
       torch.cuda.empty_cache()
   
   try:
       train()
   finally:
       cleanup()
   ```

## Best Practices

### 1. Environment Setup

```bash
# Set environment variables for better stability
export NCCL_DEBUG=INFO
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# For InfiniBand networks
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
```

### 2. Launch Command

```bash
# Use torchrun for better process management
torchrun --standalone --nproc_per_node=4 ddp_training.py

# With explicit configuration
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12355 \
    ddp_training.py
```

### 3. Memory Management

```python
# Clear cache periodically
if step % 100 == 0:
    torch.cuda.empty_cache()

# Monitor memory usage
def log_memory():
    allocated = torch.cuda.memory_allocated() / 1024**2
    cached = torch.cuda.memory_reserved() / 1024**2
    print(f"Memory - Allocated: {allocated:.2f}MB, Cached: {cached:.2f}MB")
```

### 4. Error Recovery

```python
# Implement graceful error handling
def train_with_recovery():
    for epoch in range(num_epochs):
        try:
            train_epoch()
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                reduce_batch_size()
                continue
            elif "reduction" in str(e):
                synchronize_all_ranks()
                continue
            else:
                raise
```

## Debugging Tools

### 1. Environment Variables
```bash
# Enable detailed logging
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO

# For debugging hanging issues
export CUDA_LAUNCH_BLOCKING=1
```

### 2. Monitoring Script
```python
import psutil
import torch

def monitor_resources():
    # GPU memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            cached = torch.cuda.memory_reserved(i) / 1024**2
            print(f"GPU {i}: {allocated:.2f}MB allocated, {cached:.2f}MB cached")
    
    # System memory
    memory = psutil.virtual_memory()
    print(f"System memory: {memory.percent:.1f}% used")
    
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU usage: {cpu_percent:.1f}%")
```

### 3. Gradual Debugging Approach

1. **Start simple**: Test with single GPU first
2. **Add complexity gradually**: Test with 2 GPUs, then more
3. **Enable debug mode**: Use extensive logging
4. **Test with synthetic data**: Eliminate data loading issues
5. **Profile memory usage**: Identify memory bottlenecks

## Performance Optimization

### 1. NCCL Optimization
```bash
# For better performance on specific hardware
export NCCL_TREE_THRESHOLD=0
export NCCL_MIN_NRINGS=4
export NCCL_MAX_NRINGS=16
```

### 2. DataLoader Optimization
```python
# Optimal settings for most cases
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=min(8, cpu_count // world_size),
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)
```

### 3. Model Optimization
```python
# Enable optimizations
model = DDP(
    model,
    broadcast_buffers=True,
    gradient_as_bucket_view=True,  # Memory optimization
    static_graph=True  # If model structure is fixed
)
```

Remember: Start with the most conservative settings and gradually optimize for performance once stability is achieved.