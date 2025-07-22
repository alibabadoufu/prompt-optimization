#!/usr/bin/env python3
"""
Test script for PromptTune application
"""

import sys
import os
import pandas as pd
from unittest.mock import Mock, patch

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app import PromptTuneApp
    print("✅ Successfully imported PromptTuneApp")
except ImportError as e:
    print(f"❌ Failed to import PromptTuneApp: {e}")
    sys.exit(1)

def test_app_initialization():
    """Test that the app initializes correctly."""
    print("\n🧪 Testing app initialization...")
    app = PromptTuneApp()
    assert app.lm is None
    assert app.context_content == ""
    assert app.ground_truth_data is None
    assert app.optimization_results == []
    assert app.best_prompt == ""
    assert app.compiled_program is None
    print("✅ App initialization test passed")

def test_file_processing():
    """Test file processing functionality."""
    print("\n🧪 Testing file processing...")
    app = PromptTuneApp()
    
    # Test context document processing
    print("  Testing context document processing...")
    
    # Create a mock file object
    class MockFile:
        def __init__(self, name, content):
            self.name = name
            self.content = content
    
    # Test with sample context file
    with open('sample_data/sample_context.txt', 'w') as f:
        f.write("Test context content")
    
    mock_file = MockFile('sample_data/sample_context.txt', '')
    success, message = app.process_context_document(mock_file)
    assert success == True
    assert len(app.context_content) > 0
    print("    ✅ Context document processing passed")
    
    print("✅ File processing tests passed")

def test_metrics():
    """Test evaluation metrics."""
    print("\n🧪 Testing evaluation metrics...")
    app = PromptTuneApp()
    
    # Test exact match
    score = app.exact_match_metric("hello world", "hello world")
    assert score == 1.0
    
    score = app.exact_match_metric("hello world", "goodbye world")
    assert score == 0.0
    print("  ✅ Exact match metric test passed")
    
    # Test F1 score
    score = app.f1_score_metric("hello world", "hello world")
    assert score == 1.0
    
    score = app.f1_score_metric("hello", "hello world")
    assert score > 0.0 and score < 1.0
    print("  ✅ F1 score metric test passed")
    
    print("✅ Metrics tests passed")

def test_ground_truth_processing():
    """Test ground truth data processing."""
    print("\n🧪 Testing ground truth data processing...")
    app = PromptTuneApp()
    
    # Create test CSV
    test_data = {
        'question': ['What is 2+2?', 'What is the capital of France?'],
        'ground_truth_answer': ['4', 'Paris']
    }
    df = pd.DataFrame(test_data)
    df.to_csv('test_ground_truth.csv', index=False)
    
    class MockFile:
        def __init__(self, name):
            self.name = name
    
    mock_file = MockFile('test_ground_truth.csv')
    success, message, preview = app.process_ground_truth_data(mock_file)
    
    assert success == True
    assert app.ground_truth_data is not None
    assert len(app.ground_truth_data) == 2
    print("✅ Ground truth processing test passed")
    
    # Clean up
    os.remove('test_ground_truth.csv')

def main():
    """Run all tests."""
    print("🎯 PromptTune Application Tests")
    print("==============================")
    
    try:
        test_app_initialization()
        test_file_processing()
        test_metrics()
        test_ground_truth_processing()
        
        print("\n🎉 All tests passed! PromptTune is ready to use.")
        print("\nTo run the application:")
        print("  ./launch.sh")
        print("  or")
        print("  python app.py")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
