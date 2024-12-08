import jax
import jax.numpy as jnp
import sonnet as snt
import haiku as hk
import optax
from typing import Optional, Dict, Any, List, Tuple, Union

class Transformer(snt.Module):
    def __init__(self, 
                 config: Dict[str, Any]):
        """
        Highly configurable transformer with advanced features
        
        Args:
            config: Comprehensive configuration dictionary
        """
        super().__init__(name=config.get('name', 'AdvancedTransformer'))
        
        # Advanced configuration parsing
        self.num_layers = config.get('num_layers', 6)
        self.d_model = config.get('d_model', 512)
        self.num_heads = config.get('num_heads', 8)
        self.dff = config.get('dff', 2048)
        self.dropout_rate = config.get('dropout_rate', 0.1)
        
        # Multi-modal support
        self.input_modalities = config.get('input_modalities', ['text'])
        self.output_modalities = config.get('output_modalities', ['text'])
        
        # Modality-specific embeddings
        self.modality_embeddings = {
            modality: snt.Embed(
                config.get(f'{modality}_vocab_size', 10000), 
                self.d_model
            ) for modality in self.input_modalities
        }
        
        # Advanced encoder with multi-modal support
        self.encoder_layers = [
            AdaptiveMultiModalEncoderLayer(
                d_model=self.d_model, 
                num_heads=self.num_heads, 
                dff=self.dff, 
                dropout_rate=self.dropout_rate * (1 + 0.1 * i),
                num_modalities=len(self.input_modalities)
            ) for i in range(self.num_layers)
        ]
        
        # Adaptive output projection with multi-modal support
        self.output_projections = {
            modality: snt.Linear(
                config.get(f'{modality}_vocab_size', 10000)
            ) for modality in self.output_modalities
        }
        
        # Meta-learning components
        self.meta_embedding = MetaLearningEmbedding(
            num_tasks=config.get('num_tasks', 10),
            embedding_dim=self.d_model
        )
        
        # Advanced normalization and regularization
        self.layer_norm = snt.LayerNorm(
            axis=-1, 
            create_scale=True, 
            create_offset=True
        )
        self.dropout = snt.Dropout(self.dropout_rate)
    
    def __call__(self, 
                 inputs: Dict[str, jnp.ndarray], 
                 training: bool = False,
                 task_id: Optional[int] = None) -> Dict[str, jnp.ndarray]:
        """
        Advanced forward pass supporting multi-modal inputs
        
        Args:
            inputs: Dictionary of input tensors by modality
            training: Training mode flag
            task_id: Optional task-specific embedding
        
        Returns:
            Dictionary of output tensors by modality
        """
        # Process multi-modal inputs
        processed_inputs = {}
        for modality, input_tensor in inputs.items():
            # Embed each modality
            x = self.modality_embeddings[modality](input_tensor)
            
            # Add positional encoding
            seq_len = input_tensor.shape[1]
            pos_ids = jnp.arange(seq_len)[jnp.newaxis, :]
            pos_emb = self._get_positional_encoding(seq_len)
            x += pos_emb
            
            # Inject meta-learning embedding if task_id provided
            if task_id is not None:
                task_emb = self.meta_embedding(task_id)
                x += task_emb
            
            processed_inputs[modality] = x
        
        # Multi-modal processing
        intermediate_outputs = {}
        for modality, x in processed_inputs.items():
            # Scale and normalize
            x *= jnp.sqrt(self.d_model)
            x = self.layer_norm(x)
            
            # Adaptive dropout
            if training:
                x = self.dropout(x, deterministic=False)
            
            # Progressive encoder layers
            layer_outputs = [x]
            for layer in self.encoder_layers:
                x = layer(x, training)
                layer_outputs.append(x)
            
            # Multi-scale aggregation
            intermediate_outputs[modality] = jnp.mean(
                jnp.stack(layer_outputs), axis=0
            )
        
        # Output projection for each modality
        outputs = {}
        for modality, x in intermediate_outputs.items():
            outputs[modality] = self.output_projections[modality](x)
        
        return outputs
    
    def _get_positional_encoding(self, seq_len: int) -> jnp.ndarray:
        """
        Advanced positional encoding with learnable components
        
        Args:
            seq_len: Sequence length
        
        Returns:
            Positional encoding tensor
        """
        position = jnp.arange(seq_len)[:, jnp.newaxis]
        div_term = jnp.exp(
            jnp.arange(0, self.d_model, 2) * 
            -(jnp.log(10000.0) / self.d_model)
        )
        
        pos_encoding = jnp.zeros((seq_len, self.d_model))
        pos_encoding = pos_encoding.at[:, 0::2].set(
            jnp.sin(position * div_term)
        )
        pos_encoding = pos_encoding.at[:, 1::2].set(
            jnp.cos(position * div_term)
        )
        
        return pos_encoding[jnp.newaxis, ...]

class AdaptiveMultiModalEncoderLayer(snt.Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 dff: int, 
                 dropout_rate: float = 0.1,
                 num_modalities: int = 1,
                 name: Optional[str] = None):
        super().__init__(name=name)
        
        # Cross-modal attention mechanism
        self.cross_modal_attention = CrossModalAttention(
            d_model, num_heads, num_modalities
        )
        
        # Enhanced feed-forward network
        self.ffn = AdaptiveFeedForward(d_model, dff)
        
        # Advanced normalization
        self.layernorm1 = snt.LayerNorm(
            axis=-1, 
            create_scale=True, 
            create_offset=True
        )
        self.layernorm2 = snt.LayerNorm(
            axis=-1, 
            create_scale=True, 
            create_offset=True
        )
        
        self.dropout_rate = dropout_rate
    
    def __call__(self, x, training, key=None):
        # Cross-modal self-attention
        attn_output = self.cross_modal_attention(x, x, x)
        
        # Adaptive dropout with proper key handling
        if training:
            if key is None:
                key = jax.random.PRNGKey(0)
            dropout_key1, dropout_key2 = jax.random.split(key)
            attn_output = hk.dropout(dropout_key1, self.dropout_rate, attn_output)
        
        # Residual connection and normalization
        out1 = self.layernorm1(x + attn_output)
        
        # Feed-forward processing
        ffn_output = self.ffn(out1)
        
        # Adaptive dropout with proper key handling
        if training:
            ffn_output = hk.dropout(dropout_key2, self.dropout_rate, ffn_output)
        
        # Final layer normalization
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

class CrossModalAttention(snt.Module):
    def __init__(self, 
                 d_model: int, 
                 num_heads: int, 
                 num_modalities: int,
                 name: Optional[str] = None):
        super().__init__(name=name)
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.num_modalities = num_modalities
        self.depth = d_model // num_heads
        
        # Modal-specific linear projections
        self.modal_projections = [
            (snt.Linear(d_model), snt.Linear(d_model), snt.Linear(d_model))
            for _ in range(num_modalities)
        ]
        
        self.output_projection = snt.Linear(d_model)
    
    def __call__(self, 
                 q: jnp.ndarray, 
                 k: jnp.ndarray, 
                 v: jnp.ndarray) -> jnp.ndarray:
        batch_size = q.shape[0]
        
        # Multi-modal attention computation
        attention_outputs = []
        for i in range(self.num_modalities):
            # Modal-specific projections
            wq, wk, wv = self.modal_projections[i]
            
            # Compute attention for each modality
            modal_q = self._split_heads(wq(q))
            modal_k = self._split_heads(wk(k))
            modal_v = self._split_heads(wv(v))
            
            # Scaled dot-product attention
            modal_attn = self._scaled_dot_product_attention(
                modal_q, modal_k, modal_v
            )
            attention_outputs.append(modal_attn)
        
        # Aggregate multi-modal attention
        combined_attention = jnp.mean(
            jnp.stack(attention_outputs), axis=0
        )
        
        # Reshape and project
        combined_attention = combined_attention.transpose((0, 2, 1, 3))
        concat_attention = combined_attention.reshape(
            (batch_size, -1, self.d_model)
        )
        
        return self.output_projection(concat_attention)
    
    def _split_heads(self, x):
        batch_size = x.shape[0]
        x = x.reshape((batch_size, -1, self.num_heads, self.depth))
        return x.transpose((0, 2, 1, 3))
    
    def _scaled_dot_product_attention(self, q, k, v):
        matmul_qk = jnp.matmul(q, k.transpose((0, 1, 3, 2)))
        scale = jnp.sqrt(jnp.float32(self.depth))
        scaled_attention_logits = matmul_qk / scale
        attention_weights = jax.nn.softmax(scaled_attention_logits, axis=-1)
        return jnp.matmul(attention_weights, v)

class MetaLearningEmbedding(snt.Module):
    def __init__(self, 
                 num_tasks: int, 
                 embedding_dim: int,
                 name: Optional[str] = None):
        super().__init__(name=name)
        
        # Learnable task embeddings
        self.task_embeddings = snt.Embed(num_tasks, embedding_dim)
    
    def __call__(self, task_id: int) -> jnp.ndarray:
        """
        Generate task-specific embedding
        
        Args:
            task_id: Identifier for the specific task
        
        Returns:
            Task-specific embedding vector
        """
        return self.task_embeddings(task_id)

class AdaptiveFeedForward(snt.Module):
    def __init__(self, 
                 d_model: int, 
                 dff: int, 
                 activation: str = 'gelu',
                 name: Optional[str] = None):
        super().__init__(name=name)
        
        # Configurable feed-forward network
        self.dense1 = snt.Linear(dff)
        self.dense2 = snt.Linear(d_model)
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, activation: str):
        activation_map = {
            'gelu': jax.nn.gelu,
            'swish': jax.nn.silu,
            'relu': jax.nn.relu,
            'sigmoid': jax.nn.sigmoid
        }
        return activation_map.get(activation, jax.nn.gelu)
    
    def __call__(self, x):
        x = self.dense1(x)
        x = self.activation(x)
        return self.dense2(x)

def create_advanced_learning_rate_schedule(
    base_lr: float = 1e-3,
    warmup_steps: int = 4000,
    decay_steps: int = 100000,
    min_lr: float = 1e-5
) -> optax.Schedule:
    """
    Advanced learning rate schedule with warmup, decay, and lower bound
    
    Args:
        base_lr: Initial learning rate
        warmup_steps: Number of warmup steps
        decay_steps: Steps after which learning rate decays
        min_lr: Minimum learning rate
    """
    def schedule(step):
        # Linear warmup
        warmup_lr = base_lr * (step / warmup_steps)
        
        # Cosine decay
        decay_lr = min_lr + 0.5 * (base_lr - min_lr) * (
            1 + jnp.cos(jnp.pi * jnp.minimum(step, decay_steps) / decay_steps)
        )
        
        # Combine warmup and decay
        lr = jnp.where(step < warmup_steps, warmup_lr, decay_lr)
        return jnp.maximum(lr, min_lr)
    
    return schedule

def main():
    # Advanced configuration
    config = {
        'num_layers': 6,
        'd_model': 512,
        'num_heads': 8,
        'dff': 2048,
        'dropout_rate': 0.1,
        'input_modalities': ['text', 'audio'],
        'output_modalities': ['text'],
        'text_vocab_size': 10000,
        'audio_vocab_size': 1024,
        'num_tasks': 5
    }
    
    # Create advanced transformer
    model = Transformer(config)
    
    # Simulate multi-modal input
    key = jax.random.PRNGKey(0)
    inputs = {
        'text': jax.random.randint(key, (32, 128), 0, config['text_vocab_size']),
        'audio': jax.random.randint(key, (32, 64), 0, config['audio_vocab_size'])
    }
    
    # Forward pass with task-specific embedding
    outputs = model(inputs, training=True, task_id=2)
    
    for modality, output in outputs.items():
        print(f"{modality.capitalize()} Output shape:", output.shape)

if __name__ == "__main__":
    main()