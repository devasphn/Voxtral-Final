#!/bin/bash
# Test script to verify TCP server can start independently

set -e

echo "=== Testing TCP Server Startup ==="
echo ""

# Set environment
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export PYTHONPATH="$(pwd):$PYTHONPATH"
export PYTHONUNBUFFERED=1

echo "📁 Working directory: $(pwd)"
echo "📁 Python path: $PYTHONPATH"
echo ""

# Clean up first
echo "🧹 Cleaning up any existing processes..."
pkill -f "python.*tcp_server" 2>/dev/null || true
fuser -k 8766/tcp 2>/dev/null || true
sleep 2

# Create logs directory
mkdir -p logs

# Start TCP server directly (not in background to see output)
echo "🚀 Starting TCP server directly..."
echo "📝 This will show all output. Press Ctrl+C to stop."
echo ""

python3 -m src.streaming.tcp_server
