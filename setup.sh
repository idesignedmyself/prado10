#!/bin/bash
# PRADO9_EVO Setup Script
# Automatically sets up the virtual environment and installs all dependencies

set -e  # Exit on error

echo "🚀 PRADO9_EVO Setup Script"
echo "=========================="
echo ""

# Check if .env exists
if [ ! -d ".env" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .env
else
    echo "✓ Virtual environment already exists"
fi

echo ""
echo "🔧 Activating virtual environment..."
source .env/bin/activate

echo ""
echo "⬆️  Upgrading pip, setuptools, and wheel..."
python -m pip install --upgrade pip setuptools wheel --quiet

echo ""
echo "📥 Installing PRADO9_EVO and dependencies..."
python -m pip install -e . --quiet

echo ""
echo "✅ Setup complete!"
echo ""
echo "=========================="
echo "Available commands:"
echo "  prado help              - Show all commands"
echo "  prado train QQQ start 01 01 2020 end 12 31 2023"
echo "  prado backtest QQQ --standard"
echo "  prado backtest QQQ --walk-forward"
echo "  prado backtest QQQ --crisis"
echo "  prado backtest QQQ --monte-carlo 10000"
echo "  prado predict QQQ"
echo "  prado live QQQ --mode simulate"
echo ""
echo "🎯 To use PRADO commands, run:"
echo "   source .env/bin/activate"
echo ""
