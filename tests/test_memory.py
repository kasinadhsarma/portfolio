"""Memory usage and functionality test for VisionAI components."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.portfolio import Portfolio
import psutil
import numpy as np
import jax.numpy as jnp
from data.processor import DataProcessor

def test_model_memory():
    """Test memory usage during model operations."""
    # Record initial memory
    initial_memory = get_memory_usage()

    # Initialize model
    portfolio = Portfolio()
    after_init_memory = get_memory_usage()

    # Verify memory usage after initialization is reasonable
    memory_increase = after_init_memory - initial_memory
    assert memory_increase < 1024, f"Model initialization used too much memory: {memory_increase}MB"

    # Test processing
    message = "What can you tell me about machine learning?"
    result = portfolio.process_message(message)

    # Verify result structure
    assert isinstance(result, dict), "Result should be a dictionary"
    assert 'text' in result, "Result should contain 'text' key"
    assert 'emotion' in result, "Result should contain 'emotion' key"
    assert isinstance(result['text'], str), "Response text should be a string"
    assert isinstance(result['emotion'], str), "Emotion should be a string"

    # Verify memory usage after processing
    final_memory = get_memory_usage()
    processing_memory = final_memory - after_init_memory
    assert processing_memory < 512, f"Message processing used too much memory: {processing_memory}MB"

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024

if __name__ == "__main__":
    test_model_memory()
