#!/bin/bash
# Enhanced process cleanup script for Voxtral Real-time Streaming

echo "=== Cleaning up existing Voxtral processes ==="

# Function to kill process by port
kill_by_port() {
    local port=$1
    local service_name=$2
    
    echo "Checking port $port for $service_name..."
    
    # Get PIDs using the port
    PIDS=$(lsof -ti:$port 2>/dev/null)
    
    if [ ! -z "$PIDS" ]; then
        echo "Found processes using port $port: $PIDS"
        for PID in $PIDS; do
            echo "Killing process $PID using port $port"
            kill -9 $PID 2>/dev/null || true
        done
        
        # Wait and check again
        sleep 1
        REMAINING=$(lsof -ti:$port 2>/dev/null)
        if [ ! -z "$REMAINING" ]; then
            echo "Force killing remaining processes on port $port: $REMAINING"
            kill -9 $REMAINING 2>/dev/null || true
        fi
    else
        echo "No processes found using port $port"
    fi
}

# Kill all Python processes related to Voxtral streaming
echo "Killing existing processes by name pattern..."
pkill -f "python.*health_check" || true
pkill -f "python.*ui_server" || true  
pkill -f "python.*tcp_server" || true
pkill -f "python.*websocket_server" || true
pkill -f "uvicorn.*ui_server" || true
pkill -f "uvicorn.*health_check" || true
pkill -f "nohup python" || true

# Also kill any orphaned Python processes on our ports
for port in 8000 8005 8766 8765; do
    fuser -k ${port}/tcp 2>/dev/null || true
done

# Wait for processes to terminate
sleep 2

# Kill by port (more aggressive)
echo "Killing processes using specific ports..."
kill_by_port 8000 "UI Server"
kill_by_port 8005 "Health Check Server"
kill_by_port 8766 "TCP Server"
kill_by_port 8765 "Old WebSocket Server"

# Additional cleanup for any remaining uvicorn processes
echo "Cleaning up any remaining uvicorn processes..."
pkill -f uvicorn || true

# Wait for cleanup to complete
sleep 3

# Final verification
echo "Final port availability check..."
for port in 8000 8005 8766; do
    if lsof -i:$port >/dev/null 2>&1; then
        echo "⚠️  Port $port is still in use after cleanup"
        # Try one more time with more force
        STUBBORN_PID=$(lsof -ti:$port 2>/dev/null)
        if [ ! -z "$STUBBORN_PID" ]; then
            echo "Force killing stubborn process $STUBBORN_PID on port $port"
            kill -9 $STUBBORN_PID 2>/dev/null || true
            sleep 1
        fi
        
        # Check again
        if lsof -i:$port >/dev/null 2>&1; then
            echo "❌ Port $port could not be freed"
        else
            echo "✅ Port $port is now available"
        fi
    else
        echo "✅ Port $port is available"
    fi
done

echo ""
echo "✅ Cleanup completed!"
echo "Available ports:"
echo "  - 8000: UI Server + WebSocket"
echo "  - 8005: Health Check"
echo "  - 8766: TCP Server"
echo ""
echo "You can now run ./run.sh to start the servers"
