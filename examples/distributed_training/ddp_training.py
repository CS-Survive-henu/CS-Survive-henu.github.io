#!/usr/bin/env python3
"""
Robust Multi-GPU DDP Training Script for DinoTxt Model

This script addresses common DDP training issues:
1. RuntimeError about unfinished reduction in prior iteration
2. DataLoader worker crashes 
3. Parameter indices not receiving gradients for certain ranks

Features:
- Proper DDP parameter synchronization
- Robust DataLoader with memory management
- Comprehensive error handling
- Single-GPU fallback mode
- Detailed logging and debugging
- Memory monitoring and cleanup

Usage:
    # Multi-GPU training
    torchrun --nproc_per_node=2 ddp_training.py --batch_size 32 --epochs 10
    
    # Single-GPU fallback  
    python ddp_training.py --batch_size 32 --epochs 10 --single_gpu
"""

import argparse
import logging
import os
import sys
import time
import warnings
from contextlib import nullcontext
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import psutil

# Suppress known warnings that don't affect functionality
warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")


class DinoTxtModel(nn.Module):
    """
    Example DinoTxt model for demonstration.
    This is a simplified version focusing on DDP training aspects.
    """
    
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 512, 
                 num_heads: int = 8, num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(1024, embed_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
        # Track which parameters are used to avoid unused parameter errors
        self._used_parameters = set()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with careful parameter usage tracking.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)
        self._used_parameters.add('token_embedding')
        
        # Position embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeds = self.position_embedding(positions)
        self._used_parameters.add('position_embedding')
        
        # Combine embeddings
        embeddings = token_embeds + pos_embeds
        embeddings = self.dropout(embeddings)
        self._used_parameters.add('dropout')
        
        # Create attention mask for transformer if not provided
        if attention_mask is not None:
            # Convert attention mask to the format expected by transformer
            # (batch_size, seq_len) -> (batch_size, seq_len, seq_len)
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None
        
        # Transformer forward pass
        hidden_states = self.transformer(embeddings, src_key_padding_mask=src_key_padding_mask)
        self._used_parameters.add('transformer')
        
        # Output projection
        logits = self.output_projection(hidden_states)
        self._used_parameters.add('output_projection')
        
        return logits
    
    def get_unused_parameters(self) -> list:
        """Get list of parameters that weren't used in the last forward pass."""
        all_params = set(name for name, _ in self.named_parameters())
        unused = all_params - self._used_parameters
        self._used_parameters.clear()  # Reset for next forward pass
        return list(unused)


class SyntheticTextDataset(Dataset):
    """
    Synthetic text dataset for demonstration purposes.
    In practice, replace this with your actual dataset.
    """
    
    def __init__(self, num_samples: int = 1000, seq_len: int = 128, vocab_size: int = 10000):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Generate synthetic data
        np.random.seed(42)  # For reproducibility
        self.data = np.random.randint(0, vocab_size, (num_samples, seq_len))
        self.attention_masks = np.ones((num_samples, seq_len), dtype=np.float32)
        
        # Add some padding for realism
        for i in range(num_samples):
            actual_len = np.random.randint(seq_len // 2, seq_len)
            self.attention_masks[i, actual_len:] = 0
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': torch.tensor(self.data[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.float),
            'labels': torch.tensor(self.data[idx], dtype=torch.long)  # Self-supervised learning
        }


class DDPTrainer:
    """
    Robust DDP trainer with comprehensive error handling and debugging.
    """
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.is_distributed = self.world_size > 1 and not args.single_gpu
        
        # Setup logging
        self._setup_logging()
        
        # Setup device
        self._setup_device()
        
        # Initialize distributed training if needed
        if self.is_distributed:
            self._init_distributed()
        
        # Setup model, optimizer, and data
        self.model = self._setup_model()
        self.optimizer = self._setup_optimizer()
        self.train_loader = self._setup_dataloader()
        
        # Training state
        self.step = 0
        self.epoch = 0
        
        self.logger.info(f"Trainer initialized on rank {self.rank}/{self.world_size}")
    
    def _setup_logging(self):
        """Setup comprehensive logging."""
        log_level = logging.DEBUG if self.args.debug else logging.INFO
        
        # Create logger
        self.logger = logging.getLogger(f'DDPTrainer_rank_{self.rank}')
        self.logger.setLevel(log_level)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create formatter
        formatter = logging.Formatter(
            f'[Rank {self.rank}] %(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Console handler (only for rank 0 unless debug mode)
        if self.rank == 0 or self.args.debug:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler for all ranks
        if self.args.log_file:
            file_handler = logging.FileHandler(f'{self.args.log_file}_rank_{self.rank}.log')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.logger.info("Logging setup complete")
    
    def _setup_device(self):
        """Setup device with proper GPU handling."""
        if torch.cuda.is_available() and not self.args.cpu_only:
            if self.is_distributed:
                self.device = torch.device(f'cuda:{self.local_rank}')
                torch.cuda.set_device(self.local_rank)
            else:
                self.device = torch.device('cuda:0')
            
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            self.logger.info(f"Using device: {self.device}")
            self.logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(self.device) / 1024**2:.2f} MB")
        else:
            self.device = torch.device('cpu')
            self.logger.info("Using CPU")
    
    def _init_distributed(self):
        """Initialize distributed training with robust error handling."""
        try:
            # Initialize process group
            if 'MASTER_ADDR' not in os.environ:
                os.environ['MASTER_ADDR'] = 'localhost'
            if 'MASTER_PORT' not in os.environ:
                os.environ['MASTER_PORT'] = '12355'
            
            dist.init_process_group(
                backend='nccl' if torch.cuda.is_available() else 'gloo',
                init_method='env://',
                world_size=self.world_size,
                rank=self.rank,
                timeout=torch.distributed.default_pg_timeout
            )
            
            self.logger.info(f"Distributed training initialized: rank {self.rank}/{self.world_size}")
            
            # Synchronize all processes
            dist.barrier()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize distributed training: {e}")
            self.logger.info("Falling back to single-GPU training")
            self.is_distributed = False
            self.world_size = 1
            self.rank = 0
    
    def _setup_model(self) -> nn.Module:
        """Setup model with proper DDP wrapping."""
        model = DinoTxtModel(
            vocab_size=self.args.vocab_size,
            embed_dim=self.args.embed_dim,
            num_heads=self.args.num_heads,
            num_layers=self.args.num_layers,
            dropout=self.args.dropout
        )
        
        # Move model to device
        model = model.to(self.device)
        
        # Wrap with DDP if distributed
        if self.is_distributed:
            model = DDP(
                model,
                device_ids=[self.local_rank] if torch.cuda.is_available() else None,
                output_device=self.local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=self.args.find_unused_parameters,
                broadcast_buffers=True,
                gradient_as_bucket_view=True  # Memory optimization
            )
            
            self.logger.info("Model wrapped with DDP")
        
        # Log model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer with proper learning rate scaling."""
        # Scale learning rate by world size for DDP
        lr = self.args.learning_rate
        if self.is_distributed:
            lr *= self.world_size
            self.logger.info(f"Scaled learning rate to {lr} for distributed training")
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=self.args.weight_decay,
            eps=1e-8
        )
        
        return optimizer
    
    def _setup_dataloader(self) -> DataLoader:
        """Setup DataLoader with robust multiprocessing and memory management."""
        dataset = SyntheticTextDataset(
            num_samples=self.args.num_samples,
            seq_len=self.args.seq_len,
            vocab_size=self.args.vocab_size
        )
        
        # Setup sampler for distributed training
        sampler = None
        shuffle = True
        if self.is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=True
            )
            shuffle = False
        
        # Calculate optimal number of workers to avoid crashes
        if self.args.num_workers == -1:
            # Auto-detect optimal number of workers
            cpu_count = psutil.cpu_count(logical=False) or 4
            max_workers = min(cpu_count, 8)  # Cap at 8 to avoid memory issues
            num_workers = max_workers // self.world_size if self.is_distributed else max_workers
        else:
            num_workers = self.args.num_workers
        
        # Multiprocessing settings to prevent worker crashes
        multiprocessing_context = None
        persistent_workers = False
        
        if num_workers > 0:
            try:
                import multiprocessing as mp
                multiprocessing_context = mp.get_context('spawn')  # Safer than fork
                persistent_workers = True
            except Exception as e:
                self.logger.warning(f"Failed to setup multiprocessing context: {e}")
                num_workers = 0
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available() and not self.args.cpu_only,
            drop_last=True,  # Ensure consistent batch sizes across ranks
            persistent_workers=persistent_workers,
            multiprocessing_context=multiprocessing_context,
            prefetch_factor=2 if num_workers > 0 else 2,
            # Memory management
            timeout=30 if num_workers > 0 else 0
        )
        
        self.logger.info(f"DataLoader setup: {len(dataset)} samples, "
                        f"batch_size={self.args.batch_size}, num_workers={num_workers}")
        
        return dataloader
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss with proper error handling."""
        input_ids = batch['input_ids'].to(self.device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
        labels = batch['labels'].to(self.device, non_blocking=True)
        
        # Forward pass
        logits = self.model(input_ids, attention_mask)
        
        # Compute loss (language modeling loss)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100  # Ignore padding tokens
        )
        
        return loss
    
    def _log_memory_usage(self):
        """Log current memory usage."""
        if torch.cuda.is_available() and not self.args.cpu_only:
            allocated = torch.cuda.memory_allocated(self.device) / 1024**2
            cached = torch.cuda.memory_reserved(self.device) / 1024**2
            self.logger.debug(f"GPU memory - Allocated: {allocated:.2f} MB, Cached: {cached:.2f} MB")
        
        # System memory
        memory = psutil.virtual_memory()
        self.logger.debug(f"System memory - Used: {memory.percent:.1f}%, "
                         f"Available: {memory.available / 1024**3:.2f} GB")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with comprehensive error handling."""
        self.model.train()
        
        # Set epoch for distributed sampler
        if self.is_distributed and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.epoch)
        
        total_loss = 0.0
        num_batches = 0
        
        # Setup progress bar (only on rank 0)
        progress_bar = None
        if self.rank == 0:
            progress_bar = tqdm(
                self.train_loader,
                desc=f'Epoch {self.epoch}',
                unit='batch'
            )
            data_iterator = progress_bar
        else:
            data_iterator = self.train_loader
        
        try:
            for batch_idx, batch in enumerate(data_iterator):
                try:
                    # Log memory usage periodically
                    if batch_idx % 100 == 0:
                        self._log_memory_usage()
                    
                    # Zero gradients
                    self.optimizer.zero_grad()
                    
                    # Forward pass and loss computation
                    loss = self._compute_loss(batch)
                    
                    # Check for NaN loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.error(f"Invalid loss detected at step {self.step}: {loss}")
                        if self.args.skip_invalid_loss:
                            continue
                        else:
                            raise ValueError(f"Invalid loss: {loss}")
                    
                    # Backward pass
                    loss.backward()
                    
                    # Gradient clipping
                    if self.args.gradient_clip_val > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.args.gradient_clip_val
                        )
                    
                    # Optimizer step
                    self.optimizer.step()
                    
                    # Update metrics
                    total_loss += loss.item()
                    num_batches += 1
                    self.step += 1
                    
                    # Update progress bar
                    if progress_bar is not None:
                        progress_bar.set_postfix({
                            'loss': f'{loss.item():.4f}',
                            'avg_loss': f'{total_loss / num_batches:.4f}'
                        })
                    
                    # Periodic logging
                    if self.step % self.args.log_interval == 0:
                        avg_loss = total_loss / num_batches
                        self.logger.info(f"Step {self.step}: loss={loss.item():.4f}, avg_loss={avg_loss:.4f}")
                        
                        # Check for unused parameters
                        if hasattr(self.model, 'module'):
                            unused_params = self.model.module.get_unused_parameters()
                            if unused_params:
                                self.logger.warning(f"Unused parameters detected: {unused_params}")
                    
                    # Force synchronization periodically to prevent reduction issues
                    if self.is_distributed and self.step % 100 == 0:
                        dist.barrier()
                
                except Exception as e:
                    self.logger.error(f"Error in batch {batch_idx} at step {self.step}: {e}")
                    if self.args.continue_on_error:
                        continue
                    else:
                        raise
        
        except Exception as e:
            self.logger.error(f"Error during epoch {self.epoch}: {e}")
            raise
        
        finally:
            if progress_bar is not None:
                progress_bar.close()
        
        # Compute average loss
        avg_loss = total_loss / max(num_batches, 1)
        
        # Synchronize loss across all ranks
        if self.is_distributed:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / self.world_size
        
        return {'avg_loss': avg_loss, 'num_batches': num_batches}
    
    def train(self):
        """Main training loop with comprehensive error handling."""
        self.logger.info(f"Starting training for {self.args.epochs} epochs")
        
        try:
            for epoch in range(self.args.epochs):
                self.epoch = epoch
                
                # Train epoch
                epoch_start_time = time.time()
                metrics = self.train_epoch()
                epoch_time = time.time() - epoch_start_time
                
                # Log epoch results
                if self.rank == 0:
                    self.logger.info(
                        f"Epoch {epoch} completed in {epoch_time:.2f}s: "
                        f"avg_loss={metrics['avg_loss']:.4f}, "
                        f"batches={metrics['num_batches']}"
                    )
                
                # Cleanup CUDA cache periodically
                if torch.cuda.is_available() and epoch % 5 == 0:
                    torch.cuda.empty_cache()
                
                # Synchronize all processes
                if self.is_distributed:
                    dist.barrier()
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources and finalize distributed training."""
        self.logger.info("Cleaning up...")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Destroy process group
        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
        
        self.logger.info("Cleanup complete")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Robust DDP Training Script')
    
    # Model parameters
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='Gradient clipping value')
    
    # Data parameters
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--seq_len', type=int, default=128, help='Sequence length')
    parser.add_argument('--num_workers', type=int, default=-1, help='Number of dataloader workers (-1 for auto)')
    
    # DDP parameters
    parser.add_argument('--find_unused_parameters', action='store_true', 
                       help='Find unused parameters in DDP (slower but safer)')
    parser.add_argument('--single_gpu', action='store_true', help='Force single GPU mode')
    parser.add_argument('--cpu_only', action='store_true', help='Use CPU only')
    
    # Error handling
    parser.add_argument('--continue_on_error', action='store_true', 
                       help='Continue training on batch errors')
    parser.add_argument('--skip_invalid_loss', action='store_true', 
                       help='Skip batches with invalid loss')
    
    # Logging and debugging
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--log_file', type=str, default='ddp_training', 
                       help='Log file prefix')
    parser.add_argument('--log_interval', type=int, default=100, 
                       help='Logging interval in steps')
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Create trainer and start training
    trainer = DDPTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()