#!/bin/bash

# Example launch scripts for different configurations

echo "DDP Training Launch Scripts"
echo "=========================="

# Function to check if CUDA is available
check_cuda() {
    python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA devices:', torch.cuda.device_count())" 2>/dev/null
}

echo "Checking CUDA availability..."
check_cuda

echo ""
echo "Available launch configurations:"
echo ""

echo "1. Single GPU (fallback mode):"
echo "   python ddp_training.py --single_gpu --batch_size 32 --epochs 5"
echo ""

echo "2. Multi-GPU on single node (2 GPUs):"
echo "   torchrun --nproc_per_node=2 ddp_training.py --batch_size 16 --epochs 5"
echo ""

echo "3. Multi-GPU on single node (4 GPUs):"
echo "   torchrun --nproc_per_node=4 ddp_training.py --batch_size 8 --epochs 5"
echo ""

echo "4. Debug mode with detailed logging:"
echo "   torchrun --nproc_per_node=2 ddp_training.py --debug --log_interval 10 --batch_size 16"
echo ""

echo "5. Conservative mode (for unstable environments):"
echo "   torchrun --nproc_per_node=2 ddp_training.py --find_unused_parameters --num_workers 0 --batch_size 8"
echo ""

echo "6. CPU-only training:"
echo "   python ddp_training.py --cpu_only --batch_size 8 --epochs 2"
echo ""

echo "Environment variables for debugging:"
echo "   export TORCH_DISTRIBUTED_DEBUG=DETAIL"
echo "   export NCCL_DEBUG=INFO" 
echo "   export CUDA_LAUNCH_BLOCKING=1"
echo ""

# Function to run a configuration
run_config() {
    case $1 in
        1)
            echo "Running single GPU configuration..."
            python ddp_training.py --single_gpu --batch_size 32 --epochs 2 --debug
            ;;
        2)
            echo "Running 2 GPU configuration..."
            torchrun --nproc_per_node=2 ddp_training.py --batch_size 16 --epochs 2 --debug
            ;;
        3)
            echo "Running 4 GPU configuration..."
            torchrun --nproc_per_node=4 ddp_training.py --batch_size 8 --epochs 2 --debug
            ;;
        4)
            echo "Running debug mode..."
            torchrun --nproc_per_node=2 ddp_training.py --debug --log_interval 10 --batch_size 16 --epochs 1
            ;;
        5)
            echo "Running conservative mode..."
            torchrun --nproc_per_node=2 ddp_training.py --find_unused_parameters --num_workers 0 --batch_size 8 --epochs 1
            ;;
        6)
            echo "Running CPU-only training..."
            python ddp_training.py --cpu_only --batch_size 8 --epochs 2 --debug
            ;;
        *)
            echo "Invalid configuration number"
            ;;
    esac
}

# If an argument is provided, run that configuration
if [ $# -eq 1 ]; then
    run_config $1
else
    echo "To run a specific configuration, use: $0 <config_number>"
    echo "Example: $0 1  (runs single GPU configuration)"
fi