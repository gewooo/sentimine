#!/bin/bash
echo "============================================"
echo "  SentimentIQ — Frontend Dev Server"
echo "============================================"
echo ""

if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 18+"
    exit 1
fi

cd "$(dirname "$0")"

echo "📦 Installing dependencies..."
npm install --silent

echo ""
echo "🚀 Starting frontend on http://localhost:5173"
echo ""

npm run dev
