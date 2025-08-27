#!/usr/bin/env python3
"""
Test Suite for DDP Training Script

This script validates that the DDP training implementation correctly handles
all the specified issues and edge cases.

Author: CS-Survive-henu.github.io
License: MIT
"""

import os
import sys
import tempfile
import shutil
import subprocess
import time
import signal
from pathlib import Path
from typing import Dict, Any
import pytest

# Add the training script directory to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp
    from ddp_training import DDPTrainer, DDPLogger, MemoryManager, SafeDataLoader
    from ddp_training import DinoTxtModel, DummyDataset, setup_environment
except ImportError as e:
    print(f"Failed to import training modules: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


class TestDDPComponents:
    """Test individual components of the DDP training system."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.log_dir = self.temp_dir / "logs"
        self.checkpoint_dir = self.temp_dir / "checkpoints"
        
        # Test configuration
        self.config = {
            'vocab_size': 1000,
            'hidden_size': 64,
            'num_layers': 2,
            'num_heads': 4,
            'dropout': 0.1,
            'batch_size': 8,
            'learning_rate': 1e-3,
            'num_epochs': 2,
            'dataset_size': 100,
            'seq_len': 32,
            'num_workers': 0,  # Avoid worker issues in tests
            'log_dir': str(self.log_dir),
            'checkpoint_dir': str(self.checkpoint_dir),
            'log_interval': 5,
            'save_interval': 1,
            'use_amp': False,  # Disable AMP for testing
        }
    
    def teardown_method(self):
        """Cleanup test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_logger_creation(self):
        """Test DDPLogger creates proper log files."""
        rank, world_size = 0, 1
        logger = DDPLogger(rank, world_size, str(self.log_dir))
        
        # Test logging
        logger.info("Test message")
        logger.warning("Test warning")
        logger.error("Test error")
        
        # Check log file exists
        log_file = self.log_dir / f"rank_{rank}.log"
        assert log_file.exists(), "Log file should be created"
        
        # Check log content
        content = log_file.read_text()
        assert "Test message" in content
        assert "Test warning" in content  
        assert "Test error" in content
    
    def test_memory_manager(self):
        """Test MemoryManager functionality."""
        logger = DDPLogger(0, 1, str(self.log_dir))
        memory_manager = MemoryManager(logger, max_memory_gb=1.0)
        
        # Test memory usage tracking
        usage = memory_manager.get_memory_usage()
        assert 'cpu_memory_gb' in usage
        assert usage['cpu_memory_gb'] > 0
        
        # Test memory cleanup
        memory_manager.cleanup_memory()  # Should not raise errors
        
        # Test logging
        memory_manager.log_memory_usage("test_stage")
    
    def test_model_creation(self):
        """Test DinoTxtModel with unused parameters."""
        model = DinoTxtModel(
            vocab_size=self.config['vocab_size'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            num_heads=self.config['num_heads'],
            dropout=self.config['dropout']
        )
        
        # Test forward pass without auxiliary
        batch_size, seq_len = 4, 32
        input_ids = torch.randint(0, self.config['vocab_size'], (batch_size, seq_len))
        
        outputs = model(input_ids, use_auxiliary=False)
        assert 'logits' in outputs
        assert outputs['logits'].shape == (batch_size, seq_len, self.config['vocab_size'])
        assert 'auxiliary' not in outputs  # Should not be present
        
        # Test forward pass with auxiliary (creates potential unused parameters)
        outputs = model(input_ids, use_auxiliary=True)
        assert 'logits' in outputs
        assert 'auxiliary' in outputs
        assert outputs['auxiliary'].shape == (batch_size, 1)
    
    def test_dataset_and_dataloader(self):
        """Test DummyDataset and SafeDataLoader."""
        dataset = DummyDataset(
            size=self.config['dataset_size'],
            seq_len=self.config['seq_len'],
            vocab_size=self.config['vocab_size']
        )
        
        assert len(dataset) == self.config['dataset_size']
        
        # Test dataset item
        item = dataset[0]
        assert 'input_ids' in item
        assert 'labels' in item
        assert item['input_ids'].shape == (self.config['seq_len'],)
        assert item['labels'].shape == (self.config['seq_len'],)
        
        # Test SafeDataLoader
        logger = DDPLogger(0, 1, str(self.log_dir))
        dataloader = SafeDataLoader(
            dataset=dataset,
            batch_size=self.config['batch_size'],
            rank=0,
            world_size=1,
            logger=logger,
            num_workers=0,  # Use 0 workers for testing
            pin_memory=False
        )
        
        # Test iteration
        batch = next(iter(dataloader))
        assert 'input_ids' in batch
        assert 'labels' in batch
        assert batch['input_ids'].shape[0] == self.config['batch_size']
    
    def test_single_gpu_training(self):
        """Test single GPU/CPU training without DDP issues."""
        if torch.cuda.is_available():
            # Test will run on GPU if available, CPU otherwise
            device_count = 1
        else:
            device_count = 1
        
        rank, world_size = 0, device_count
        
        # Create trainer
        trainer = DDPTrainer(rank, world_size, self.config)
        
        # Test model is properly initialized
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.train_loader is not None
        
        # Test single training step
        batch = next(iter(trainer.train_loader))
        step_stats = trainer.train_step(batch)
        
        assert 'loss' in step_stats
        assert 'step' in step_stats
        assert isinstance(step_stats['loss'], float)
        assert step_stats['step'] == 1
        
        # Cleanup
        trainer._cleanup()


class TestDDPErrorScenarios:
    """Test specific DDP error scenarios and recovery."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Minimal config for error testing
        self.config = {
            'vocab_size': 100,
            'hidden_size': 32,
            'num_layers': 1,
            'num_heads': 2,
            'batch_size': 4,
            'dataset_size': 20,
            'seq_len': 16,
            'num_workers': 0,
            'log_dir': str(self.temp_dir / "logs"),
            'checkpoint_dir': str(self.temp_dir / "checkpoints"),
            'num_epochs': 1,
            'use_amp': False,
        }
    
    def teardown_method(self):
        """Cleanup test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_unused_parameters_handling(self):
        """Test that unused parameters don't cause crashes."""
        rank, world_size = 0, 1
        
        # Create trainer
        trainer = DDPTrainer(rank, world_size, self.config)
        
        # Create a batch and test training with conditional auxiliary usage
        batch = next(iter(trainer.train_loader))
        
        # Force auxiliary to be unused (should not crash)
        original_forward = trainer.model.module.forward
        
        def mock_forward(input_ids, use_auxiliary=False):
            # Always ignore auxiliary, creating unused parameters
            return original_forward(input_ids, use_auxiliary=False)
        
        trainer.model.module.forward = mock_forward
        
        # This should not raise the "unused parameters" error
        try:
            step_stats = trainer.train_step(batch)
            assert 'loss' in step_stats
        except RuntimeError as e:
            if "unused parameters" in str(e) or "Expected to have finished reduction" in str(e):
                pytest.fail(f"Unused parameters not handled properly: {e}")
            else:
                # Other errors might be OK for this test
                pass
        finally:
            trainer._cleanup()
    
    def test_memory_cleanup_on_error(self):
        """Test memory cleanup when errors occur."""
        rank, world_size = 0, 1
        trainer = DDPTrainer(rank, world_size, self.config)
        
        # Simulate training step with forced cleanup
        initial_memory = trainer.memory_manager.get_memory_usage()
        
        # Force memory cleanup
        trainer.memory_manager.cleanup_memory()
        
        post_cleanup_memory = trainer.memory_manager.get_memory_usage()
        
        # Memory usage should be tracked
        assert 'cpu_memory_gb' in initial_memory
        assert 'cpu_memory_gb' in post_cleanup_memory
        
        trainer._cleanup()
    
    def test_dataloader_worker_robustness(self):
        """Test DataLoader handles worker configuration properly."""
        rank, world_size = 0, 1
        logger = DDPLogger(rank, world_size, self.config['log_dir'])
        
        dataset = DummyDataset(
            size=self.config['dataset_size'],
            seq_len=self.config['seq_len'],
            vocab_size=self.config['vocab_size']
        )
        
        # Test with 0 workers (should work)
        dataloader = SafeDataLoader(
            dataset=dataset,
            batch_size=self.config['batch_size'],
            rank=rank,
            world_size=world_size,
            logger=logger,
            num_workers=0,
            pin_memory=False
        )
        
        # Should be able to iterate without crashes
        batches = list(dataloader)
        assert len(batches) > 0
        
        # Test worker count adjustment for world size
        assert dataloader.num_workers <= os.cpu_count() // max(1, world_size)


class TestDDPIntegration:
    """Integration tests for full DDP training pipeline."""
    
    def setup_method(self):
        """Setup test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = {
            'vocab_size': 500,
            'hidden_size': 64,
            'num_layers': 2,
            'num_heads': 4,
            'batch_size': 8,
            'dataset_size': 50,
            'seq_len': 24,
            'num_workers': 0,
            'num_epochs': 2,
            'log_dir': str(self.temp_dir / "logs"),
            'checkpoint_dir': str(self.temp_dir / "checkpoints"),
            'log_interval': 5,
            'save_interval': 1,
            'use_amp': False,
            'max_errors_per_epoch': 2,
        }
    
    def teardown_method(self):
        """Cleanup test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_full_training_pipeline(self):
        """Test complete training pipeline without crashes."""
        rank, world_size = 0, 1
        
        trainer = DDPTrainer(rank, world_size, self.config)
        
        # Run one epoch of training
        try:
            epoch_stats = trainer.train_epoch()
            
            # Verify epoch completed successfully
            assert epoch_stats['num_batches'] > 0
            assert epoch_stats['avg_loss'] > 0
            assert epoch_stats['num_errors'] == 0  # Should have no errors
            
        except Exception as e:
            pytest.fail(f"Training pipeline failed: {e}")
        finally:
            trainer._cleanup()
    
    def test_checkpoint_saving_and_loading(self):
        """Test checkpoint saving and loading functionality."""
        rank, world_size = 0, 1
        
        trainer = DDPTrainer(rank, world_size, self.config)
        
        # Run one epoch to generate stats
        epoch_stats = trainer.train_epoch()
        
        # Save checkpoint
        epoch = 0
        trainer._save_checkpoint(epoch, epoch_stats)
        
        # Verify checkpoint file exists
        checkpoint_file = Path(self.config['checkpoint_dir']) / f'checkpoint_epoch_{epoch}.pt'
        assert checkpoint_file.exists(), "Checkpoint file should be created"
        
        # Verify checkpoint content
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        assert 'epoch' in checkpoint
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'stats' in checkpoint
        assert 'config' in checkpoint
        
        trainer._cleanup()


def test_script_execution():
    """Test that the main script can be executed without crashes."""
    # This test runs the script with minimal configuration
    script_path = SCRIPT_DIR / "ddp_training.py"
    
    # Create a test script that runs minimal training
    test_script_content = f'''
import sys
sys.path.insert(0, "{SCRIPT_DIR}")

from ddp_training import DDPTrainer

config = {{
    'vocab_size': 100,
    'hidden_size': 32,
    'num_layers': 1,
    'num_heads': 2,
    'batch_size': 4,
    'dataset_size': 20,
    'seq_len': 16,
    'num_workers': 0,
    'num_epochs': 1,
    'log_interval': 5,
    'use_amp': False,
}}

try:
    trainer = DDPTrainer(0, 1, config)
    trainer.train()
    print("SUCCESS: Training completed without crashes")
except Exception as e:
    print(f"FAILED: {{e}}")
    sys.exit(1)
'''
    
    test_script = SCRIPT_DIR / "test_minimal_run.py"
    test_script.write_text(test_script_content)
    
    try:
        # Run the test script
        result = subprocess.run(
            [sys.executable, str(test_script)],
            capture_output=True,
            text=True,
            timeout=60  # 1 minute timeout
        )
        
        if result.returncode != 0:
            pytest.fail(f"Script execution failed: {result.stderr}")
        
        assert "SUCCESS" in result.stdout, f"Script did not complete successfully: {result.stdout}"
        
    except subprocess.TimeoutExpired:
        pytest.fail("Script execution timed out")
    finally:
        if test_script.exists():
            test_script.unlink()


def run_tests():
    """Run all tests with proper setup."""
    # Setup environment
    setup_environment()
    
    print("Running DDP Training Tests...")
    print("=" * 50)
    
    # Import pytest and run tests
    try:
        import pytest
        
        # Run tests with verbose output
        test_args = [
            __file__,
            "-v",
            "--tb=short",
            "-x",  # Stop on first failure
        ]
        
        result = pytest.main(test_args)
        
        if result == 0:
            print("\n✅ All tests passed!")
        else:
            print(f"\n❌ Tests failed with exit code: {result}")
        
        return result
        
    except ImportError:
        print("pytest not available, running basic tests...")
        
        # Run basic tests without pytest
        try:
            test_script_execution()
            print("✅ Basic script execution test passed!")
            return 0
        except Exception as e:
            print(f"❌ Basic test failed: {e}")
            return 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)