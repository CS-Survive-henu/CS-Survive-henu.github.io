#!/bin/bash
# DDP Training Launch Script
# This script provides convenient ways to launch DDP training with different configurations

set -e

# Default values
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/ddp_training.py"
GPUS=""
BATCH_SIZE=32
EPOCHS=10
LOG_DIR="logs"
CHECKPOINT_DIR="checkpoints"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -g, --gpus GPUS         Comma-separated list of GPU IDs (e.g., 0,1,2,3)"
    echo "  -b, --batch-size SIZE   Batch size per GPU (default: 32)"
    echo "  -e, --epochs EPOCHS     Number of training epochs (default: 10)"
    echo "  -l, --log-dir DIR       Log directory (default: logs)"
    echo "  -c, --checkpoint-dir DIR Checkpoint directory (default: checkpoints)"
    echo "  --cpu                   Force CPU training (ignore GPUs)"
    echo "  --test                  Run a quick test with small dataset"
    echo "  --debug                 Enable debug mode with verbose logging"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                      # Auto-detect GPUs and run training"
    echo "  $0 --gpus 0,1           # Run on GPU 0 and 1"
    echo "  $0 --cpu               # Force CPU training"
    echo "  $0 --test              # Quick test run"
    echo "  $0 --debug             # Debug mode"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--gpus)
            GPUS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -e|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -l|--log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        -c|--checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --cpu)
            FORCE_CPU=1
            shift
            ;;
        --test)
            TEST_MODE=1
            shift
            ;;
        --debug)
            DEBUG_MODE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Check if Python script exists
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Setup environment
echo "Setting up training environment..."

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$CHECKPOINT_DIR"

# Check for required packages
echo "Checking Python dependencies..."
if ! python -c "import torch, psutil" 2>/dev/null; then
    echo "Installing required packages..."
    pip install -r "$SCRIPT_DIR/requirements.txt"
fi

# Configure environment variables
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

if [[ -n "$DEBUG_MODE" ]]; then
    export CUDA_LAUNCH_BLOCKING=1
    export TORCH_DISTRIBUTED_DEBUG=INFO
    echo "Debug mode enabled"
fi

# Handle test mode
if [[ -n "$TEST_MODE" ]]; then
    echo "Running in test mode..."
    EPOCHS=2
    BATCH_SIZE=16
    export TEST_MODE=1
fi

# GPU detection and setup
if [[ -n "$FORCE_CPU" ]]; then
    echo "Forcing CPU training..."
    export CUDA_VISIBLE_DEVICES=""
    WORLD_SIZE=1
elif [[ -n "$GPUS" ]]; then
    echo "Using specified GPUs: $GPUS"
    export CUDA_VISIBLE_DEVICES="$GPUS"
    WORLD_SIZE=$(echo "$GPUS" | tr ',' '\n' | wc -l)
else
    # Auto-detect GPUs
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | wc -l)
        if [[ $GPU_COUNT -gt 0 ]]; then
            echo "Auto-detected $GPU_COUNT GPUs"
            WORLD_SIZE=$GPU_COUNT
            GPUS=$(seq -s, 0 $((GPU_COUNT-1)))
            export CUDA_VISIBLE_DEVICES="$GPUS"
        else
            echo "No GPUs detected, falling back to CPU"
            WORLD_SIZE=1
            export CUDA_VISIBLE_DEVICES=""
        fi
    else
        echo "nvidia-smi not found, falling back to CPU"
        WORLD_SIZE=1
        export CUDA_VISIBLE_DEVICES=""
    fi
fi

echo "Configuration:"
echo "  GPUs: ${GPUS:-None (CPU)}"
echo "  World Size: $WORLD_SIZE"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Log Directory: $LOG_DIR"
echo "  Checkpoint Directory: $CHECKPOINT_DIR"

# Launch training
echo ""
echo "Starting DDP training..."
echo "=========================="

if [[ $WORLD_SIZE -gt 1 ]]; then
    echo "Launching multi-GPU training with $WORLD_SIZE processes..."
    # For multi-GPU, the script handles multiprocessing internally
    python "$PYTHON_SCRIPT" \
        --batch_size "$BATCH_SIZE" \
        --num_epochs "$EPOCHS" \
        --log_dir "$LOG_DIR" \
        --checkpoint_dir "$CHECKPOINT_DIR"
else
    echo "Launching single-process training..."
    python "$PYTHON_SCRIPT" \
        --batch_size "$BATCH_SIZE" \
        --num_epochs "$EPOCHS" \
        --log_dir "$LOG_DIR" \
        --checkpoint_dir "$CHECKPOINT_DIR"
fi

echo ""
echo "Training completed!"
echo "Logs saved to: $LOG_DIR"
echo "Checkpoints saved to: $CHECKPOINT_DIR"