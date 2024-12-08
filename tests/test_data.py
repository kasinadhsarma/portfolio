"""
Tests for data processing functionality
"""
import pytest
import jax.numpy as jnp
from data.data import DataProcessor
from data.utils import tokenize_text, create_vocabulary, text_to_tensor

def test_tokenize_text():
    text = "Hello World!"
    tokens = tokenize_text(text)
    assert tokens == ["hello", "world!"]

def test_create_vocabulary():
    texts = ["hello world", "world hello"]
    vocab = create_vocabulary(texts)
    assert len(vocab) == 2
    assert "hello" in vocab
    assert "world" in vocab

def test_text_to_tensor():
    vocab = {"hello": 0, "world": 1}
    text = "hello world"
    tensor = text_to_tensor(text, vocab)
    assert jnp.array_equal(tensor, jnp.array([0, 1]))

def test_data_processor_initialization():
    processor = DataProcessor()
    assert hasattr(processor, 'training_data')
    assert hasattr(processor, 'vocab')

def test_data_processor_process_data():
    processor = DataProcessor()
    test_data = [
        {"input": "hello", "output": "hi there"},
        {"input": "how are you", "output": "good"}
    ]
    result = processor.process_data(test_data)
    assert "inputs" in result
    assert "outputs" in result
    assert "vocab_size" in result
    assert isinstance(result["inputs"], jnp.ndarray)
    assert isinstance(result["outputs"], jnp.ndarray)
    assert result["inputs"].shape[0] == 2  # batch size
