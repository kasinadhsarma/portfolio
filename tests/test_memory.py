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
    print("\nInitial Memory Usage:")
    print_memory_usage()

    portfolio = Portfolio()
    print("\nAfter Model Initialization:")
    print_memory_usage()

    # Test processing
    message = "What can you tell me about machine learning?"
    result = portfolio.process_message(message)
    print("\nAfter Processing Message:")
    print_memory_usage()

    print("\nResponse:", result['text'])
    print("Emotion:", result['emotion'])

    return result

def print_memory_usage():
    """Print current memory usage."""
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"RSS: {memory_info.rss / 1024 / 1024:.2f} MB")
    print(f"VMS: {memory_info.vms / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    test_model_memory()
