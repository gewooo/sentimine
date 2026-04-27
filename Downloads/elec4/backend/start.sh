#!/bin/bash
echo "============================================"
echo "  SentimentIQ — Backend API Server"
echo "============================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.8+"
    exit 1
fi

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install fastapi uvicorn scikit-learn pandas numpy python-multipart --quiet

echo ""
echo "🚀 Starting API server on http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""

cd "$(dirname "$0")"
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
