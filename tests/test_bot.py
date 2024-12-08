import sys
import os
import pytest
import jax
import jax.numpy as jnp

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.bot import Portfolio
from data.processor import DataProcessor
from models.portfolio import Transformer

@pytest.fixture
def portfolio():
    return Portfolio()

def test_portfolio_initialization(portfolio):
    """Test portfolio initialization and configuration"""
    assert isinstance(portfolio.data_processor, DataProcessor)
    assert isinstance(portfolio.transformer, Transformer)
    # Variables are stored as FrozenDict in Flax
    assert isinstance(portfolio.variables, (dict, jax.tree_util.PyTreeDef))
    assert isinstance(portfolio.key, jnp.ndarray)

def test_portfolio_transformer_attributes(portfolio):
    """Test transformer configuration"""
    assert portfolio.transformer.num_layers == 6
    assert portfolio.transformer.d_model == 512
    assert portfolio.transformer.num_heads == 8
    assert portfolio.transformer.dff == 2048

def test_portfolio_message_processing(portfolio):
    """Test message processing pipeline"""
    test_message = "Hello, how are you?"
    response = portfolio.process_message(test_message)

    assert isinstance(response, dict)
    assert 'text' in response
    assert 'emotion' in response
    assert isinstance(response['text'], str)
    assert isinstance(response['emotion'], str)
    assert response['emotion'] in ['happy', 'sad', 'neutral']

def test_error_handling(portfolio):
    """Test error handling in message processing"""
    # Test invalid input
    response = portfolio.process_message("")
    assert response['emotion'] == 'sad'
    assert 'apologize' in response['text'].lower()

    # Test None input
    response = portfolio.process_message(None)
    assert response['emotion'] == 'sad'
    assert 'apologize' in response['text'].lower()

def test_emotion_detection(portfolio):
    """Test emotion detection logic"""
    # Test happy emotion
    response = portfolio.process_message("Hello!")
    assert response['emotion'] == 'happy'

    # Test sad emotion
    response = portfolio.process_message("I'm sorry about that")
    assert response['emotion'] == 'sad'

    # Test excited emotion
    response = portfolio.process_message("That's fascinating!")
    assert response['emotion'] == 'excited'

    # Test neutral emotion
    response = portfolio.process_message("What is AI?")
    assert response['emotion'] == 'neutral'
