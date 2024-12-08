import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from typing import Optional, Dict, Any, List, Tuple, Union
from functools import partial
from data.processor import DataProcessor
from .logger import logger  # Add logger import

class Transformer(nn.Module):
    """Memory-efficient transformer implementation."""
    num_layers: int
    d_model: int
    num_heads: int
    dff: int
    dropout_rate: float
    input_modalities: List[str]
    output_modalities: List[str]
    vocab_sizes: Dict[str, int]

    def setup(self):
        """Initialize model components."""
        # Initialize embeddings for each modality
        self.embeddings = {
            modality: nn.Embed(
                num_embeddings=self.vocab_sizes[modality],
                features=self.d_model
            ) for modality in self.input_modalities
        }

        # Initialize encoder layers
        num_modalities = len(self.input_modalities)
        self.encoder_layers = [
            AdaptiveMultiModalEncoderLayer(
                d_model=self.d_model,
                num_heads=self.num_heads,
                dff=self.dff,
                dropout_rate=self.dropout_rate,
                num_modalities=num_modalities
            ) for _ in range(self.num_layers)
        ]

        # Initialize output projections
        self.output_projections = {
            modality: nn.Dense(
                features=self.vocab_sizes[modality]
            ) for modality in self.output_modalities
        }

        # Initialize layer normalization
        self.layer_norm = nn.LayerNorm()

    @nn.compact
    def __call__(self, inputs: Dict[str, jnp.ndarray], training: bool = False) -> Dict[str, jnp.ndarray]:
        """Forward pass with memory optimization."""
        outputs = {}

        try:
            # Convert inputs to dictionary if it's not already
            if not isinstance(inputs, dict):
                inputs = {"text": inputs}

            # Process each input modality
            for modality in self.input_modalities:
                if modality not in inputs:
                    logger.warning(f"Missing modality: {modality}")
                    continue

                x = inputs[modality]

                # Embed inputs
                x = self.embeddings[modality](x)

                # Add positional encoding
                x = x + self._get_positional_encoding(x.shape)

                # Apply dropout with proper PRNG key handling
                if training:
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
                else:
                    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=True)

                # Process through encoder layers
                for layer in self.encoder_layers:
                    x = layer(x, training=training)

                # Project to output space if this modality is an output
                if modality in self.output_modalities:
                    x = self.output_projections[modality](x)

                outputs[modality] = x

            # Ensure at least one modality was processed
            if not outputs:
                raise ValueError("No valid modalities found in input")

            return outputs

        except Exception as e:
            logger.error(f"Transformer forward pass error: {str(e)}")
            raise ValueError(f"Forward pass error: {str(e)}")

    def _get_positional_encoding(self, shape: tuple) -> jnp.ndarray:
        """Generate positional encoding."""
        max_length = shape[1]
        d_model = shape[2]

        positions = jnp.arange(max_length)[:, None]
        div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-jnp.log(10000.0) / d_model))

        pos_encoding = jnp.zeros((max_length, d_model))
        pos_encoding = pos_encoding.at[:, 0::2].set(jnp.sin(positions * div_term))
        pos_encoding = pos_encoding.at[:, 1::2].set(jnp.cos(positions * div_term))

        return pos_encoding[None, :, :]

    def _decode_response(self, logits: jnp.ndarray) -> str:
        """Decode model output logits to text."""
        try:
            # Get predicted token indices
            predictions = jnp.argmax(logits, axis=-1)

            # Create a data processor instance for decoding
            data_processor = DataProcessor()

            # Convert predictions to flat list if needed
            if isinstance(predictions, jnp.ndarray):
                predictions = predictions.flatten()

            # Convert to text using data processor
            decoded = data_processor.decode_tokens(predictions)
            return decoded if decoded else "I'm here to help with questions about AI and development."

        except Exception as e:
            logger.error(f"Response decoding error: {str(e)}")
            return "I apologize, but I'm having trouble generating a response."

class AdaptiveMultiModalEncoderLayer(nn.Module):
    """Multi-modal encoder layer with adaptive components"""
    d_model: int
    num_heads: int
    dff: int
    dropout_rate: float
    num_modalities: int

    def setup(self):
        self.attention = CrossModalAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_modalities=self.num_modalities
        )
        self.feed_forward = AdaptiveFeedForward(
            d_model=self.d_model,
            dff=self.dff,
            activation='gelu'
        )
        self.layer_norm1 = nn.LayerNorm()
        self.layer_norm2 = nn.LayerNorm()
        self.dropout1 = nn.Dropout(rate=self.dropout_rate)
        self.dropout2 = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """Forward pass with residual connections and layer normalization"""
        # Self attention with residual connection and layer normalization
        attn_output = self.attention(
            self.layer_norm1(x),  # Pre-norm architecture
            self.layer_norm1(x),
            self.layer_norm1(x)
        )
        attn_output = self.dropout1(attn_output, deterministic=not training)
        out1 = x + attn_output  # Residual connection

        # Feed forward with residual connection and layer normalization
        ff_output = self.feed_forward(self.layer_norm2(out1))  # Pre-norm
        ff_output = self.dropout2(ff_output, deterministic=not training)
        out2 = out1 + ff_output  # Residual connection

        return out2

class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism"""
    d_model: int
    num_heads: int
    num_modalities: int

    def setup(self):
        assert self.d_model % self.num_heads == 0
        self.depth = self.d_model // self.num_heads

        self.wq = nn.Dense(features=self.d_model)
        self.wk = nn.Dense(features=self.d_model)
        self.wv = nn.Dense(features=self.d_model)
        self.dense = nn.Dense(features=self.d_model)

    def __call__(self, q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray,
                mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        batch_size = q.shape[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self._split_heads(q, batch_size)  # (batch_size, num_heads, seq_len, depth)
        k = self._split_heads(k, batch_size)  # (batch_size, num_heads, seq_len, depth)
        v = self._split_heads(v, batch_size)  # (batch_size, num_heads, seq_len, depth)

        scaled_attention = self._scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = scaled_attention.transpose(0, 2, 1, 3)
        concat_attention = scaled_attention.reshape(batch_size, -1, self.d_model)

        return self.dense(concat_attention)

    def _split_heads(self, x: jnp.ndarray, batch_size: int) -> jnp.ndarray:
        """Split the last dimension into (num_heads, depth)"""
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(0, 2, 1, 3)

    def _scaled_dot_product_attention(self, q: jnp.ndarray, k: jnp.ndarray,
                                    v: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Calculate attention weights and apply them to values"""
        matmul_qk = jnp.matmul(q, k.transpose(0, 1, 3, 2))
        dk = jnp.sqrt(self.depth)
        scaled_attention_logits = matmul_qk / dk

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = jax.nn.softmax(scaled_attention_logits, axis=-1)
        return jnp.matmul(attention_weights, v)

class MetaLearningEmbedding(nn.Module):
    """Task-specific embedding layer"""
    num_tasks: int
    embedding_dim: int

    def setup(self):
        self.embedding = nn.Embed(
            num_embeddings=self.num_tasks,
            features=self.embedding_dim
        )

    def __call__(self, task_id: int) -> jnp.ndarray:
        """Get task-specific embedding"""
        return self.embedding(jnp.array([task_id]))[0]

class AdaptiveFeedForward(nn.Module):
    """Position-wise feed-forward network with adaptive activation"""
    d_model: int
    dff: int
    activation: str = 'gelu'

    def setup(self):
        self.dense1 = nn.Dense(features=self.dff)
        self.dense2 = nn.Dense(features=self.d_model)
        self.activation_fn = self._get_activation()

    def _get_activation(self):
        """Get activation function by name"""
        if self.activation == 'gelu':
            return nn.gelu
        elif self.activation == 'relu':
            return nn.relu
        else:
            return nn.gelu

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through feed-forward network"""
        x = self.dense1(x)
        x = self.activation_fn(x)
        return self.dense2(x)

@partial(jax.jit, static_argnums=(1,))
def create_advanced_learning_rate_schedule(
    init_value: float = 0.0001,
    num_warmup_steps: int = 4000,
    num_decay_steps: Optional[int] = None
) -> callable:
    """Create a learning rate schedule with warmup and decay"""
    def schedule(step: int) -> float:
        if num_decay_steps is None:
            decay_steps = num_warmup_steps
        else:
            decay_steps = num_decay_steps

        warmup_steps = num_warmup_steps
        step = jnp.minimum(step, decay_steps)

        warmup = init_value * step / warmup_steps
        decay = init_value * (decay_steps - step) / (decay_steps - warmup_steps)

        return jnp.where(step < warmup_steps, warmup, decay)

    return schedule

class Portfolio:
    """Main portfolio model class implementing the VisionAI chatbot functionality."""

    def __init__(self):
        """Initialize the portfolio model with transformer architecture."""
        self.config = {
            'num_layers': 4,  # Reduced from 6 for memory optimization
            'd_model': 256,   # Reduced from 512 for memory optimization
            'num_heads': 8,
            'dff': 1024,     # Reduced from 2048 for memory optimization
            'dropout_rate': 0.1,
            'input_modalities': ['text'],
            'output_modalities': ['text'],
            'vocab_size': 10000
        }

        # Initialize transformer model with variables
        self.transformer = Transformer(
            num_layers=self.config['num_layers'],
            d_model=self.config['d_model'],
            num_heads=self.config['num_heads'],
            dff=self.config['dff'],
            dropout_rate=self.config['dropout_rate'],
            input_modalities=self.config['input_modalities'],
            output_modalities=self.config['output_modalities'],
            vocab_sizes={'text': self.config['vocab_size']}
        )

        # Initialize variables
        key = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, 64), dtype=jnp.int32)
        self.variables = self.transformer.init(key, {"text": dummy_input})

        # Initialize data processor
        self.data_processor = DataProcessor()

    def process_message(self, message: str) -> dict:
        # Define emotion keywords
        sad_words = ['sorry', 'sad', 'unfortunate', 'regret', 'miss', 'hurt']
        happy_words = ['hello', 'happy', 'great', 'good', 'excited', 'thanks']
        
        message = message.lower()
        
        # Improved emotion detection
        emotion = 'neutral'
        if any(word in message for word in sad_words):
            emotion = 'sad'
        elif any(word in message for word in happy_words):
            emotion = 'happy'
        
        return {
            'text': self._process_text(message),  # Replace transformer.process with internal method
            'emotion': emotion
        }

    def _process_text(self, text: str) -> str:
        # Internal method to handle text processing
        return text  # Basic implementation for now

    def _determine_emotion(self, text: str) -> str:
        """Determine emotion based on text content."""
        try:
            if isinstance(text, (jnp.ndarray, np.ndarray)):
                # Convert array to string if needed
                text = " ".join(map(str, text.flatten()))

            if any(word in text.lower() for word in ["hello", "hi", "welcome", "greetings"]):
                return "happy"
            elif any(word in text.lower() for word in ["sorry", "apologize", "error", "trouble"]):
                return "sad"
            elif any(word in text.lower() for word in ["interesting", "fascinating", "amazing"]):
                return "excited"
            elif any(word in text.lower() for word in ["help", "assist", "support"]):
                return "helpful"
            return "neutral"
        except Exception as e:
            logger.error(f"Emotion detection error: {str(e)}")
            return "neutral"

def main():
    """Example usage of the Portfolio model"""
    # Initialize Portfolio model
    portfolio = Portfolio()

    # Example input message
    message = "Hello, how are you?"

    # Process message
    response = portfolio.process_message(message)
    print(f"Response: {response['text']}")
    print(f"Emotion: {response['emotion']}")

if __name__ == "__main__":
    main()
