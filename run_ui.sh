#!/bin/bash

echo "🚀 Starting Kiosk Agent AG-UI Dashboard..."

# Function to kill background processes on exit
cleanup() {
    echo "Stopping servers..."
    kill $(jobs -p)
    exit
}
trap cleanup EXIT

# 1. Start the Python Backend
echo "Starting Backend (FastAPI)..."
/opt/anaconda3/envs/gui/bin/python src/server.py &
BACKEND_PID=$!

# 2. Start the Next.js Frontend
echo "Starting Frontend (Next.js)..."
cd frontend
npm run dev &
FRONTEND_PID=$!

echo "------------------------------------------------"
echo "✅ Dashboard is running!"
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "------------------------------------------------"

# Keep the script running
wait
