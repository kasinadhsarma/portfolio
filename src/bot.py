from typing import Dict, Any
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from models.portfolio import (
    Transformer,
    AdaptiveMultiModalEncoderLayer,
    MetaLearningEmbedding,
    CrossModalAttention,
    AdaptiveFeedForward
)
from data import utils
from data.processor import DataProcessor
from .exceptions import ModelInferenceError, InputValidationError, TokenizationError, DataProcessingError
from .logger import logger

class Portfolio:
    def __init__(self):
        self.data = utils
        self.np = jnp
        self.data_processor = DataProcessor()

        self.config = {
            'num_layers': 6,
            'd_model': 512,
            'num_heads': 8,
            'dff': 2048,
            'dropout_rate': 0.1,
            'input_modalities': ['text'],
            'output_modalities': ['text'],
            'vocab_sizes': {'text': 10000}
        }

        # Initialize model components with proper PRNG keys
        self.transformer = Transformer(**self.config)
        self.key = jax.random.PRNGKey(0)

        # Initialize with proper input shapes and PRNG keys
        dummy_input = jnp.ones((1, 32), dtype=jnp.int32)  # Batch size 1, sequence length 32
        self.variables = self.transformer.init(
            {'params': self.key, 'dropout': jax.random.PRNGKey(1)},
            {'text': dummy_input},
            training=False
        )

    def process_message(self, message: str) -> Dict[str, Any]:
        try:
            # Input validation
            if not message or not isinstance(message, str):
                raise InputValidationError("Invalid input message")

            # Determine emotion from raw input message
            emotion = self._determine_emotion(message)

            # Process input
            try:
                processed_input = self.data_processor.process_data([
                    {"input": message, "output": ""}
                ])
            except Exception as e:
                raise DataProcessingError(f"Data processing error: {str(e)}")

            # Model inference with proper PRNG handling
            try:
                dropout_key = jax.random.PRNGKey(2)
                outputs = self.transformer.apply(
                    self.variables,
                    processed_input["inputs"],
                    training=False,
                    rngs={'dropout': dropout_key}
                )

                # Generate response using transformer output
                response_text = self._decode_response(outputs)

                return {
                    "text": response_text,
                    "emotion": emotion
                }
            except Exception as e:
                raise ModelInferenceError(f"Model inference error: {str(e)}")

        except (InputValidationError, DataProcessingError, ModelInferenceError, TokenizationError) as e:
            logger.error(str(e))
            return {
                "text": f"I apologize, but {str(e)}. Please try again.",
                "emotion": "sad"
            }
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {
                "text": "I apologize, but something went wrong. Please try again later.",
                "emotion": "sad"
            }

    def _decode_response(self, output: Any) -> str:
        try:
            # Handle dictionary output from transformer
            if isinstance(output, dict):
                output = output.get('text', output)

            # Convert to numpy array if needed
            if isinstance(output, (jnp.ndarray, np.ndarray)):
                # Get predicted token indices
                predictions = jnp.argmax(output, axis=-1)

                # Create a data processor instance for decoding
                data_processor = DataProcessor()

                # Convert predictions to flat list if needed
                if isinstance(predictions, jnp.ndarray):
                    predictions = predictions.flatten()

                # Convert to text using data processor
                decoded = data_processor.decode_tokens(predictions)
                return decoded if decoded else "I'm here to help with questions about AI and development."

            return "I'm here to help with questions about AI and development."

        except Exception as e:
            logger.error(f"Response decoding error: {str(e)}")
            return "I'm here to help with questions about AI and development."

    def _determine_emotion(self, text: str) -> str:
        try:
            if not isinstance(text, str):
                return "neutral"

            # Basic text normalization
            text = text.lower().strip()

            # Check for excited phrases first (with exclamation)
            excited_phrases = ["fascinating!", "amazing!", "wow!", "incredible!"]
            if any(phrase in text for phrase in excited_phrases):
                return "excited"

            # Check for greetings
            greetings = ["hello!", "hi!", "hey!", "hello", "hi", "hey", "welcome", "greetings"]
            if any(text.startswith(greeting) for greeting in greetings):
                return "happy"

            # Check for complete sad phrases
            sad_phrases = ["i'm sorry", "im sorry", "i am sorry", "sorry about"]
            if any(phrase in text for phrase in sad_phrases):
                return "sad"

            # Split into words for other emotion checks
            words = text.split()

            # Check individual words for emotions
            sad_words = ["sorry", "apologize", "error", "trouble", "unfortunately"]
            if any(word in words for word in sad_words):
                return "sad"

            excited_words = ["interesting", "fascinating", "amazing", "wow", "incredible"]
            if any(word in words for word in excited_words):
                return "excited"

            helpful_words = ["help", "assist", "support"]
            if any(word in words for word in helpful_words):
                return "helpful"

            return "neutral"

        except Exception as e:
            logger.error(f"Emotion detection error: {str(e)}")
            return "neutral"
