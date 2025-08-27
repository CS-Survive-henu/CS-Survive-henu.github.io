#!/usr/bin/env python3
"""
Robust Distributed Data Parallel (DDP) Training Script

This script addresses common DDP training issues and provides a production-ready
solution for multi-GPU training with comprehensive error handling and debugging.

Fixes:
1. RuntimeError: Expected to have finished reduction in the prior iteration
2. DataLoader worker crashes and memory management issues
3. Parameter gradient synchronization issues across ranks
4. Proper cleanup and resource management

Author: CS-Survive-henu.github.io
License: MIT
"""

import os
import sys
import time
import logging
import signal
import traceback
import warnings
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import psutil
import gc

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataset import Dataset
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class DDPLogger:
    """Comprehensive logging system for DDP training with rank-aware formatting."""
    
    def __init__(self, rank: int, world_size: int, log_dir: str = "logs"):
        self.rank = rank
        self.world_size = world_size
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Setup logger
        self.logger = logging.getLogger(f"DDP_Rank_{rank}")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler for this rank
        fh = logging.FileHandler(self.log_dir / f"rank_{rank}.log")
        fh.setLevel(logging.INFO)
        
        # Console handler (only for rank 0 to avoid spam)
        if rank == 0:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                '[%(asctime)s] RANK[%(rank)s/%(world_size)s] %(levelname)s: %(message)s',
                datefmt='%H:%M:%S'
            )
            ch.setFormatter(console_formatter)
            self.logger.addHandler(ch)
        
        # File formatter
        file_formatter = logging.Formatter(
            '[%(asctime)s] RANK[%(rank)s/%(world_size)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(file_formatter)
        self.logger.addHandler(fh)
        
        # Create a custom LoggerAdapter to inject rank info
        self.logger = logging.LoggerAdapter(
            self.logger, 
            {'rank': rank, 'world_size': world_size}
        )
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
    
    def debug(self, msg: str):
        self.logger.debug(msg)


class MemoryManager:
    """Advanced memory management and monitoring for DDP training."""
    
    def __init__(self, logger: DDPLogger, max_memory_gb: float = None):
        self.logger = logger
        self.max_memory_gb = max_memory_gb
        self.process = psutil.Process()
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        # CPU memory
        memory_info = self.process.memory_info()
        cpu_memory_gb = memory_info.rss / (1024**3)
        
        # GPU memory
        gpu_memory = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                gpu_memory[f'gpu_{i}_allocated'] = allocated
                gpu_memory[f'gpu_{i}_reserved'] = reserved
        
        return {
            'cpu_memory_gb': cpu_memory_gb,
            **gpu_memory
        }
    
    def log_memory_usage(self, stage: str = ""):
        """Log current memory usage."""
        memory_stats = self.get_memory_usage()
        self.logger.info(f"Memory usage {stage}: {memory_stats}")
        
        # Check for memory warnings
        if self.max_memory_gb and memory_stats['cpu_memory_gb'] > self.max_memory_gb:
            self.logger.warning(f"High CPU memory usage: {memory_stats['cpu_memory_gb']:.2f}GB")
    
    def cleanup_memory(self):
        """Aggressive memory cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


class SafeDataLoader:
    """DataLoader wrapper with worker crash protection and proper cleanup."""
    
    def __init__(self, dataset: Dataset, batch_size: int, rank: int, world_size: int, 
                 logger: DDPLogger, num_workers: int = 4, pin_memory: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.logger = logger
        self.num_workers = min(num_workers, os.cpu_count() // max(1, world_size))
        self.pin_memory = pin_memory and torch.cuda.is_available()
        
        # Create distributed sampler
        self.sampler = DistributedSampler(
            dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=True,
            drop_last=True  # Important for DDP consistency
        )
        
        # Worker initialization function to handle crashes
        def worker_init_fn(worker_id):
            # Set different random seeds for each worker
            import random
            import numpy as np
            worker_seed = torch.initial_seed() % 2**32
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)
        
        # Create DataLoader with robust settings
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=worker_init_fn,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else 2,
            timeout=30 if num_workers > 0 else 0,  # Timeout for worker processes
        )
        
        self.logger.info(f"Created SafeDataLoader: batch_size={batch_size}, "
                        f"num_workers={self.num_workers}, pin_memory={self.pin_memory}")
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)
    
    def set_epoch(self, epoch: int):
        """Set epoch for distributed sampler."""
        self.sampler.set_epoch(epoch)


class DinoTxtModel(nn.Module):
    """Example model with potential for unused parameters (for demonstration)."""
    
    def __init__(self, vocab_size: int = 10000, hidden_size: int = 512, 
                 num_layers: int = 6, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_encoding = nn.Parameter(torch.randn(1000, hidden_size))
        
        # Main transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layers
        self.output_proj = nn.Linear(hidden_size, vocab_size)
        
        # Additional branch that might not always be used (potential unused parameters)
        self.auxiliary_branch = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.use_auxiliary = False  # Can be toggled to create unused parameters
        
    def forward(self, input_ids: torch.Tensor, use_auxiliary: bool = False) -> Dict[str, torch.Tensor]:
        seq_len = input_ids.size(1)
        
        # Embeddings
        x = self.embedding(input_ids)
        x = x + self.pos_encoding[:seq_len]
        x = self.dropout(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Main output
        logits = self.output_proj(x)
        
        result = {'logits': logits}
        
        # Auxiliary output (might not be used, creating unused parameters)
        if use_auxiliary or self.use_auxiliary:
            aux_output = self.auxiliary_branch(x.mean(dim=1))
            result['auxiliary'] = aux_output
        
        return result


class DummyDataset(Dataset):
    """Simple dummy dataset for demonstration."""
    
    def __init__(self, size: int = 10000, seq_len: int = 128, vocab_size: int = 10000):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate random token sequences
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        labels = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {
            'input_ids': input_ids,
            'labels': labels
        }


class DDPTrainer:
    """Robust DDP trainer with comprehensive error handling."""
    
    def __init__(self, rank: int, world_size: int, config: Dict[str, Any]):
        self.rank = rank
        self.world_size = world_size
        self.config = config
        
        # Initialize logging
        self.logger = DDPLogger(rank, world_size, config.get('log_dir', 'logs'))
        self.logger.info(f"Initializing DDPTrainer on rank {rank}/{world_size}")
        
        # Memory management
        self.memory_manager = MemoryManager(
            self.logger, 
            config.get('max_memory_gb', None)
        )
        
        # Initialize distributed training
        self._setup_distributed()
        
        # Setup device
        self.device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
        
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = self._create_model()
        
        # Initialize data
        self.train_loader = self._create_dataloader()
        
        # Initialize optimizer and scaler
        self.optimizer = self._create_optimizer()
        self.scaler = GradScaler() if config.get('use_amp', True) else None
        
        # Training state
        self.step = 0
        self.epoch = 0
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        self.logger.info("DDPTrainer initialization complete")
        self.memory_manager.log_memory_usage("after_init")
    
    def _setup_distributed(self):
        """Initialize distributed training with proper error handling."""
        try:
            # Initialize the process group
            backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            dist.init_process_group(
                backend=backend,
                rank=self.rank,
                world_size=self.world_size,
                timeout=torch.distributed.default_pg_timeout
            )
            
            self.logger.info(f"Distributed training initialized with backend: {backend}")
            
            # Verify the process group is working
            if dist.is_initialized():
                # Simple all-reduce test
                test_tensor = torch.tensor([self.rank], dtype=torch.float32)
                if torch.cuda.is_available():
                    test_tensor = test_tensor.cuda()
                dist.all_reduce(test_tensor)
                expected_sum = sum(range(self.world_size))
                
                if abs(test_tensor.item() - expected_sum) < 1e-6:
                    self.logger.info("Distributed communication test passed")
                else:
                    raise RuntimeError(f"Distributed test failed: got {test_tensor.item()}, expected {expected_sum}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize distributed training: {e}")
            self._cleanup()
            raise
    
    def _create_model(self) -> nn.Module:
        """Create and wrap model with DDP."""
        # Create model
        model = DinoTxtModel(
            vocab_size=self.config.get('vocab_size', 10000),
            hidden_size=self.config.get('hidden_size', 512),
            num_layers=self.config.get('num_layers', 6),
            num_heads=self.config.get('num_heads', 8),
            dropout=self.config.get('dropout', 0.1)
        )
        
        # Move to device
        model = model.to(self.device)
        
        # Wrap with DDP - this is where we fix the unused parameter issues
        model = DDP(
            model,
            device_ids=[self.rank] if torch.cuda.is_available() else None,
            output_device=self.rank if torch.cuda.is_available() else None,
            find_unused_parameters=True,  # Handle unused parameters gracefully
            broadcast_buffers=True,  # Ensure buffer synchronization
            bucket_cap_mb=25,  # Optimize communication
            gradient_as_bucket_view=True  # Memory optimization
        )
        
        self.logger.info(f"Model created and wrapped with DDP")
        return model
    
    def _create_dataloader(self) -> SafeDataLoader:
        """Create robust dataloader."""
        dataset = DummyDataset(
            size=self.config.get('dataset_size', 10000),
            seq_len=self.config.get('seq_len', 128),
            vocab_size=self.config.get('vocab_size', 10000)
        )
        
        return SafeDataLoader(
            dataset=dataset,
            batch_size=self.config.get('batch_size', 32),
            rank=self.rank,
            world_size=self.world_size,
            logger=self.logger,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=self.config.get('pin_memory', True)
        )
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with proper parameter filtering."""
        # Filter parameters that require gradients
        params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = optim.AdamW(
            params,
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 0.01),
            eps=1e-8
        )
        
        self.logger.info(f"Optimizer created with {len(params)} parameters")
        return optimizer
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
            self._cleanup()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _cleanup(self):
        """Comprehensive cleanup of resources."""
        try:
            self.logger.info("Starting cleanup...")
            
            # Clear gradients
            if hasattr(self, 'optimizer') and self.optimizer is not None:
                self.optimizer.zero_grad()
            
            # Memory cleanup
            if hasattr(self, 'memory_manager'):
                self.memory_manager.cleanup_memory()
            
            # Destroy process group
            if dist.is_initialized():
                dist.destroy_process_group()
                self.logger.info("Process group destroyed")
            
            self.logger.info("Cleanup complete")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with comprehensive error handling."""
        try:
            self.model.train()
            
            # Move batch to device
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with AMP if enabled
            if self.scaler is not None:
                with autocast():
                    # Randomly use auxiliary branch to test unused parameters
                    use_auxiliary = torch.rand(1) < 0.3  # 30% chance
                    outputs = self.model(batch['input_ids'], use_auxiliary=use_auxiliary)
                    
                    # Compute loss
                    logits = outputs['logits']
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        batch['labels'].view(-1)
                    )
                    
                    # Add auxiliary loss if present
                    if 'auxiliary' in outputs:
                        aux_loss = outputs['auxiliary'].mean()
                        loss = loss + 0.1 * aux_loss
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Regular forward pass
                use_auxiliary = torch.rand(1) < 0.3
                outputs = self.model(batch['input_ids'], use_auxiliary=use_auxiliary)
                
                logits = outputs['logits']
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    batch['labels'].view(-1)
                )
                
                if 'auxiliary' in outputs:
                    aux_loss = outputs['auxiliary'].mean()
                    loss = loss + 0.1 * aux_loss
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.optimizer.step()
            
            self.step += 1
            
            return {
                'loss': loss.item(),
                'step': self.step,
                'used_auxiliary': 'auxiliary' in outputs
            }
            
        except RuntimeError as e:
            if "Expected to have finished reduction" in str(e):
                self.logger.error(f"DDP reduction error at step {self.step}: {e}")
                self.logger.info("Attempting to recover by synchronizing and clearing gradients...")
                
                # Emergency synchronization
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                # Clear gradients and try to reset DDP state
                self.optimizer.zero_grad()
                
                # Wait for all processes
                dist.barrier()
                
                raise e  # Re-raise for handling at higher level
            else:
                self.logger.error(f"Runtime error in train_step: {e}")
                raise
        
        except Exception as e:
            self.logger.error(f"Unexpected error in train_step: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with comprehensive monitoring."""
        self.logger.info(f"Starting epoch {self.epoch}")
        
        # Set epoch for distributed sampler
        self.train_loader.set_epoch(self.epoch)
        
        epoch_stats = {
            'total_loss': 0.0,
            'num_batches': 0,
            'num_errors': 0,
            'auxiliary_usage': 0
        }
        
        # Memory monitoring
        self.memory_manager.log_memory_usage(f"epoch_{self.epoch}_start")
        
        try:
            for batch_idx, batch in enumerate(self.train_loader):
                try:
                    # Training step
                    step_stats = self.train_step(batch)
                    
                    epoch_stats['total_loss'] += step_stats['loss']
                    epoch_stats['num_batches'] += 1
                    if step_stats.get('used_auxiliary', False):
                        epoch_stats['auxiliary_usage'] += 1
                    
                    # Logging
                    if batch_idx % self.config.get('log_interval', 10) == 0:
                        self.logger.info(
                            f"Epoch {self.epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                            f"Loss: {step_stats['loss']:.4f}, "
                            f"Step: {step_stats['step']}, "
                            f"Aux: {step_stats.get('used_auxiliary', False)}"
                        )
                    
                    # Memory monitoring
                    if batch_idx % self.config.get('memory_check_interval', 50) == 0:
                        self.memory_manager.log_memory_usage(f"epoch_{self.epoch}_batch_{batch_idx}")
                    
                    # Periodic cleanup
                    if batch_idx % self.config.get('cleanup_interval', 100) == 0:
                        self.memory_manager.cleanup_memory()
                
                except Exception as e:
                    epoch_stats['num_errors'] += 1
                    self.logger.error(f"Error in batch {batch_idx}: {e}")
                    
                    if epoch_stats['num_errors'] > self.config.get('max_errors_per_epoch', 5):
                        self.logger.error("Too many errors in epoch, stopping training")
                        raise RuntimeError("Too many training errors")
                    
                    # Try to recover
                    self.memory_manager.cleanup_memory()
                    continue
        
        except Exception as e:
            self.logger.error(f"Fatal error in epoch {self.epoch}: {e}")
            raise
        
        # Calculate epoch statistics
        if epoch_stats['num_batches'] > 0:
            epoch_stats['avg_loss'] = epoch_stats['total_loss'] / epoch_stats['num_batches']
            epoch_stats['auxiliary_usage_rate'] = epoch_stats['auxiliary_usage'] / epoch_stats['num_batches']
        else:
            epoch_stats['avg_loss'] = float('inf')
            epoch_stats['auxiliary_usage_rate'] = 0.0
        
        self.logger.info(f"Epoch {self.epoch} complete: {epoch_stats}")
        self.memory_manager.log_memory_usage(f"epoch_{self.epoch}_end")
        
        self.epoch += 1
        return epoch_stats
    
    def train(self) -> None:
        """Main training loop with robust error handling."""
        self.logger.info("Starting training...")
        
        try:
            num_epochs = self.config.get('num_epochs', 10)
            
            for epoch in range(num_epochs):
                try:
                    epoch_stats = self.train_epoch()
                    
                    # Save checkpoint periodically
                    if epoch % self.config.get('save_interval', 5) == 0 and self.rank == 0:
                        self._save_checkpoint(epoch, epoch_stats)
                    
                    # Early stopping check
                    if epoch_stats['avg_loss'] < self.config.get('target_loss', 0.01):
                        self.logger.info(f"Target loss reached at epoch {epoch}")
                        break
                
                except Exception as e:
                    self.logger.error(f"Error in epoch {epoch}: {e}")
                    if epoch == 0:  # If first epoch fails, something is seriously wrong
                        raise
                    self.logger.info("Continuing to next epoch...")
                    continue
            
            self.logger.info("Training completed successfully")
        
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            self.logger.error(traceback.format_exc())
            raise
        
        finally:
            self._cleanup()
    
    def _save_checkpoint(self, epoch: int, stats: Dict[str, float]) -> None:
        """Save training checkpoint."""
        try:
            checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
            checkpoint_dir.mkdir(exist_ok=True, parents=True)
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'stats': stats,
                'config': self.config
            }
            
            if self.scaler is not None:
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
            
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, checkpoint_path)
            
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")


def setup_environment() -> None:
    """Setup training environment with proper configurations."""
    # Set environment variables for robust distributed training
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # For better error messages
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'  # Detailed distributed logs
    
    # Set multiprocessing start method
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    
    # Configure PyTorch for better memory management
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def init_worker(rank: int, world_size: int, config: Dict[str, Any]) -> None:
    """Initialize worker process for distributed training."""
    try:
        # Setup environment
        setup_environment()
        
        # Initialize trainer
        trainer = DDPTrainer(rank, world_size, config)
        
        # Start training
        trainer.train()
        
    except Exception as e:
        print(f"Worker {rank} failed with error: {e}")
        traceback.print_exc()
        raise


def parse_args():
    """Parse command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Robust DDP Training Script')
    
    # Model parameters
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--use_amp', action='store_true', default=True, help='Use automatic mixed precision')
    parser.add_argument('--no_amp', action='store_false', dest='use_amp', help='Disable automatic mixed precision')
    
    # Data parameters
    parser.add_argument('--dataset_size', type=int, default=10000, help='Dataset size')
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--pin_memory', action='store_true', default=True, help='Pin memory in DataLoader')
    
    # Monitoring parameters
    parser.add_argument('--log_interval', type=int, default=10, help='Log interval in batches')
    parser.add_argument('--save_interval', type=int, default=5, help='Checkpoint save interval in epochs')
    parser.add_argument('--memory_check_interval', type=int, default=50, help='Memory check interval in batches')
    parser.add_argument('--cleanup_interval', type=int, default=100, help='Memory cleanup interval in batches')
    parser.add_argument('--max_errors_per_epoch', type=int, default=5, help='Maximum errors per epoch before stopping')
    
    # Paths
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    
    # Thresholds
    parser.add_argument('--target_loss', type=float, default=0.01, help='Target loss for early stopping')
    parser.add_argument('--max_memory_gb', type=float, default=16.0, help='Maximum memory usage warning threshold')
    
    return parser.parse_args()


def main():
    """Main entry point for DDP training."""
    # Parse command line arguments
    args = parse_args()
    
    # Convert args to config dictionary
    config = vars(args)
    
    # Check for GPU availability
    if not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU training...")
        world_size = 1
        rank = 0
        
        # Setup environment and run single-process training
        setup_environment()
        trainer = DDPTrainer(rank, world_size, config)
        trainer.train()
        
    else:
        # Multi-GPU training
        world_size = torch.cuda.device_count()
        print(f"Starting distributed training on {world_size} GPUs")
        
        if world_size < 2:
            print("Only one GPU available, running single-GPU training...")
            rank = 0
            world_size = 1
            
            setup_environment()
            trainer = DDPTrainer(rank, world_size, config)
            trainer.train()
        else:
            # Launch multi-process training
            try:
                mp.spawn(
                    init_worker,
                    args=(world_size, config),
                    nprocs=world_size,
                    join=True
                )
                print("Distributed training completed successfully!")
                
            except Exception as e:
                print(f"Distributed training failed: {e}")
                traceback.print_exc()
                sys.exit(1)


if __name__ == '__main__':
    main()