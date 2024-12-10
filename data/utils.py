"""
Utility functions for data processing
"""
from typing import List, Dict
import jax.numpy as jnp

def tokenize_text(text: str) -> List[str]:
    """Convert text to list of tokens"""
    return text.lower().split()

def create_vocabulary(texts: List[str]) -> Dict[str, int]:
    """Create vocabulary from list of texts"""
    vocab = set()
    for text in texts:
        vocab.update(tokenize_text(text))
    return {word: idx for idx, word in enumerate(sorted(vocab))}

def text_to_tensor(text: str, vocab: Dict[str, int]) -> jnp.ndarray:
    """Convert text to tensor using vocabulary"""
    tokens = tokenize_text(text)
    return jnp.array([vocab.get(token, 0) for token in tokens])
