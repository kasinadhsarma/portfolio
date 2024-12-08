"""
Data processing module for Portfolio chatbot
"""
from typing import List, Dict, Any
import jax.numpy as jnp
from .utils import tokenize_text, create_vocabulary, text_to_tensor

class DataProcessor:
    """Handles data processing for the Portfolio chatbot"""

    def __init__(self):
        self.training_data = [
            {"input": "Hello", "output": "Hi there!"},
            {"input": "How are you?", "output": "I'm doing well, thank you."},
        ]
        self.vocab = None
        self._initialize_vocabulary()

    def _initialize_vocabulary(self):
        """Initialize vocabulary from training data"""
        all_texts = []
        for item in self.training_data:
            all_texts.extend([item["input"], item["output"]])
        self.vocab = create_vocabulary(all_texts)

    def process_data(self, data: List[Dict[str, str]]) -> Dict[str, jnp.ndarray]:
        """Process input data into tensor format"""
        if not self.vocab:
            self._initialize_vocabulary()

        inputs = [text_to_tensor(item["input"], self.vocab) for item in data]
        outputs = [text_to_tensor(item["output"], self.vocab) for item in data]

        # Pad sequences to same length
        max_input_len = max(len(x) for x in inputs)
        max_output_len = max(len(x) for x in outputs)

        padded_inputs = [
            jnp.pad(x, (0, max_input_len - len(x)), constant_values=0)
            for x in inputs
        ]
        padded_outputs = [
            jnp.pad(x, (0, max_output_len - len(x)), constant_values=0)
            for x in outputs
        ]

        return {
            "inputs": jnp.array(padded_inputs),
            "outputs": jnp.array(padded_outputs),
            "vocab_size": len(self.vocab)
        }

    def process_input(self, text):
        """Process input text into tensor format"""
        if not self.vocab:
            self._initialize_vocabulary()
        return text_to_tensor(text, self.vocab)

    def process_output(self, tensor):
        """Process output tensor into text format"""
        return self.decode_tokens(tensor)

    def decode_tokens(self, tokens: jnp.ndarray) -> str:
        """
        Convert token indices back to text

        Args:
            tokens: Array of token indices

        Returns:
            Decoded text string
        """
        # Create reverse vocabulary mapping
        id_to_token = {idx: token for token, idx in self.vocab.items()}
        # Convert tokens to text
        return " ".join(id_to_token.get(int(idx), "") for idx in tokens if idx > 0)
