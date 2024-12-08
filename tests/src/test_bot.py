from src.bot import Portfolio
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from bot import Portfolio
from data import data, utils
import jax as np

from models.portfolio import (
    Transformer,
    CrossModalAttention,
    AdaptiveFeedForward,
    MetaLearningEmbedding
)

def test_portfolio_initialization():
    portfolio = Portfolio()
    assert portfolio.sonnet is Sonnet
    assert isinstance(portfolio.transformer, Transformer)
    assert isinstance(portfolio.attention, CrossModalAttention)
    assert isinstance(portfolio.feed_forward, AdaptiveFeedForward)
    assert isinstance(portfolio.embeddings, MetaLearningEmbedding)
    assert portfolio.data == data
    assert portfolio.utils == utils
    assert portfolio.np == np

def test_portfolio_transformer_attributes():
    portfolio = Portfolio()
    # Example attribute checks, adjust based on actual Transformer implementation
    assert hasattr(portfolio.transformer, 'num_layers')
    assert hasattr(portfolio.transformer, 'hidden_size')

def test_portfolio_attention_mechanism():
    portfolio = Portfolio()
    # Example method check, adjust based on actual CrossModalAttention implementation
    assert callable(getattr(portfolio.attention, 'forward', None))

def test_portfolio_feed_forward_configuration():
    portfolio = Portfolio()
    # Example configuration check, adjust based on actual AdaptiveFeedForward implementation
    assert hasattr(portfolio.feed_forward, 'dropout_rate')
    assert portfolio.feed_forward.dropout_rate >= 0

def test_portfolio_embeddings_integrity():
    portfolio = Portfolio()
    # Example integrity check, adjust based on actual MetaLearningEmbedding implementation
    assert hasattr(portfolio.embeddings, 'embedding_size')
    assert portfolio.embeddings.embedding_size > 0

def test_portfolio_data_utils():
    portfolio = Portfolio()
    # Example data utilities check
    assert callable(getattr(portfolio.utils, 'process_data', None))
    assert portfolio.data is not None