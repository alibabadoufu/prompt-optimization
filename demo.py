#!/usr/bin/env python3
"""
PromptTune Demo Script
=====================

This script demonstrates how to use PromptTune programmatically to optimize prompts.
"""

import os
import sys
from app import PromptTuneApp

def main():
    print("🎯 PromptTune Demo")
    print("=================\n")
    
    # Initialize the app
    app = PromptTuneApp()
    print("✅ PromptTune app initialized")
    
    # Load sample context
    context_file = "sample_data/sample_context.txt"
    if os.path.exists(context_file):
        with open(context_file, 'r') as f:
            context_content = f.read()
        
        success, message = app.process_context_document(context_file)
        if success:
            print(f"✅ Context document loaded: {len(context_content)} characters")
        else:
            print(f"❌ Failed to load context: {message}")
            return
    
    # Load sample ground truth data
    ground_truth_file = "sample_data/sample_ground_truth.csv"
    if os.path.exists(ground_truth_file):
        success, message, _ = app.process_ground_truth_data(ground_truth_file)
        if success:
            print(f"✅ Ground truth data loaded: {len(app.ground_truth_data)} examples")
        else:
            print(f"❌ Failed to load ground truth: {message}")
            return
    
    print("\n🚀 To run the full interactive application:")
    print("   ./launch.sh")
    print("   or")  
    print("   python app.py")
    print("\nThe web interface will be available at: http://localhost:7860")

if __name__ == "__main__":
    main()
