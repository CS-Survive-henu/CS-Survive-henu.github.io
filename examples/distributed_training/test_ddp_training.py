#!/usr/bin/env python3
"""
Test script to validate DDP training functionality.
Tests basic functionality without requiring GPUs.
"""

import sys
import os
import tempfile
import subprocess
import torch

def test_basic_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    try:
        import torch
        import torch.nn as nn
        import torch.distributed as dist
        from torch.nn.parallel import DistributedDataParallel as DDP
        import numpy as np
        import tqdm
        import psutil
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_model_creation():
    """Test model creation and basic forward pass."""
    print("Testing model creation...")
    try:
        # Import the model from our script
        sys.path.append(os.path.dirname(__file__))
        from ddp_training import DinoTxtModel
        
        model = DinoTxtModel(vocab_size=1000, embed_dim=128, num_heads=4, num_layers=2)
        
        # Test forward pass
        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        with torch.no_grad():
            output = model(input_ids, attention_mask)
        
        expected_shape = (batch_size, seq_len, 1000)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        
        print("✓ Model creation and forward pass successful")
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def test_dataset():
    """Test dataset creation and data loading."""
    print("Testing dataset...")
    try:
        from ddp_training import SyntheticTextDataset
        from torch.utils.data import DataLoader
        
        dataset = SyntheticTextDataset(num_samples=100, seq_len=32, vocab_size=1000)
        dataloader = DataLoader(dataset, batch_size=4, num_workers=0)
        
        # Test batch loading
        for i, batch in enumerate(dataloader):
            assert 'input_ids' in batch
            assert 'attention_mask' in batch
            assert 'labels' in batch
            
            # Check shapes
            batch_size = batch['input_ids'].shape[0]
            assert batch['attention_mask'].shape == (batch_size, 32)
            assert batch['labels'].shape == (batch_size, 32)
            
            if i >= 2:  # Test just a few batches
                break
        
        print("✓ Dataset and DataLoader test successful")
        return True
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        return False

def test_cpu_training():
    """Test basic training loop on CPU."""
    print("Testing CPU training...")
    try:
        # Create a temporary script to run training
        script_content = '''
import sys
import os
sys.path.append("{}")

from ddp_training import main
import argparse

# Mock command line arguments
sys.argv = [
    "test_script", 
    "--cpu_only", 
    "--epochs", "1",
    "--batch_size", "2",
    "--num_samples", "20",
    "--seq_len", "16",
    "--vocab_size", "100",
    "--embed_dim", "64",
    "--num_heads", "2",
    "--num_layers", "1",
    "--log_interval", "5",
    "--single_gpu"
]

if __name__ == "__main__":
    main()
'''.format(os.path.dirname(__file__))
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            temp_script = f.name
        
        try:
            # Run the training script
            result = subprocess.run(
                [sys.executable, temp_script], 
                capture_output=True, 
                text=True, 
                timeout=120  # 2 minute timeout
            )
            
            if result.returncode == 0:
                print("✓ CPU training test successful")
                if "Training interrupted" not in result.stdout:
                    print("  Training completed without interruption")
                return True
            else:
                print(f"✗ CPU training failed with return code {result.returncode}")
                print(f"  stdout: {result.stdout}")
                print(f"  stderr: {result.stderr}")
                return False
        finally:
            # Clean up temporary file
            os.unlink(temp_script)
            
    except Exception as e:
        print(f"✗ CPU training test failed: {e}")
        return False

def test_help_and_args():
    """Test that help works and arguments are parsed correctly."""
    print("Testing argument parsing...")
    try:
        script_path = os.path.join(os.path.dirname(__file__), 'ddp_training.py')
        
        # Test help
        result = subprocess.run(
            [sys.executable, script_path, '--help'], 
            capture_output=True, 
            text=True,
            timeout=30
        )
        
        if result.returncode == 0 and "usage:" in result.stdout:
            print("✓ Help text and argument parsing successful")
            return True
        else:
            print(f"✗ Help test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Argument parsing test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running DDP Training Script Tests")
    print("=" * 40)
    
    tests = [
        test_basic_imports,
        test_model_creation,
        test_dataset,
        test_help_and_args,
        test_cpu_training,  # This one takes longer
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print()
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print()
    print("=" * 40)
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("✓ All tests passed!")
        return True
    else:
        print("✗ Some tests failed")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)