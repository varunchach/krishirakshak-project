#!/bin/bash
# Run KrishiRakshak locally for development
set -euo pipefail

echo "=== KrishiRakshak Local Dev ==="

# Check Python
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null || {
    echo "ERROR: PyTorch not installed. Run: pip install -r requirements.txt"
    exit 1
}

# Start API server
echo "Starting FastAPI server on http://localhost:8000"
echo "Docs at http://localhost:8000/docs"
echo ""
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

sleep 2

# Start Streamlit
echo "Starting Streamlit on http://localhost:8501"
streamlit run frontend/streamlit_app.py --server.port 8501 &
UI_PID=$!

echo ""
echo "Running! API=$API_PID, UI=$UI_PID"
echo "Press Ctrl+C to stop both."

trap "kill $API_PID $UI_PID 2>/dev/null" EXIT
wait
