#!/bin/bash

# Exit on any error
set -e

# Build the project
echo "Building..."
make

# Kill any existing test_server processes
echo "Cleaning up any existing processes..."
pkill test_server 2>/dev/null || true

# Start the server
echo "Starting server..."
# ./test_server &
./Arc_Cuda &
SERVER_PID=$!

# Wait a moment for the server to initialize
sleep 1

# Run the Python client
echo "Running client..."
python3 ../test/test_client.py

# Clean up the server
echo "Cleaning up..."
kill $SERVER_PID 2>/dev/null || true

killall Arc_Cuda

echo "Test complete!"
