#!/bin/bash

echo "🎯 Starting PromptTune Application..."
echo "====================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Launch the application
echo "🚀 Launching PromptTune..."
echo "The application will be available at: http://localhost:7860"
echo "Press Ctrl+C to stop the application"
echo ""

python app.py
