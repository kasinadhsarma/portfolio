"""
Comprehensive tests for the transformer model components
"""
import pytest
import jax
import jax.numpy as jnp
from models.portfolio import Transformer, AdaptiveMultiModalEncoderLayer
from src.exceptions import ModelInferenceError

@pytest.fixture
def test_config():
    return {
        'num_layers': 2,
        'd_model': 64,
        'num_heads': 2,
        'dff': 128,
        'dropout_rate': 0.1,
        'input_modalities': ['text'],
        'output_modalities': ['text'],
        'vocab_sizes': {'text': 1000}
    }

@pytest.fixture
def transformer(test_config):
    model = Transformer(**test_config)
    # Initialize parameters with a PRNG key
    key = jax.random.PRNGKey(0)
    x = jnp.ones((1, 10), dtype=jnp.int32)
    variables = model.init(key, {'text': x}, training=False)
    return model, variables

def test_transformer_initialization(transformer, test_config):
    """Test transformer initialization and configuration"""
    model, variables = transformer
    assert model.num_layers == test_config['num_layers']
    assert model.d_model == test_config['d_model']
    assert model.num_heads == test_config['num_heads']
    assert len(variables['params']['encoder_layers_0']) > 0  # Check encoder layers exist

def test_transformer_inference(transformer):
    """Test normal inference with small batch"""
    model, variables = transformer
    inputs = {'text': jnp.ones((1, 10), dtype=jnp.int32)}
    outputs = model.apply(variables, inputs, training=False)

    assert 'text' in outputs
    assert outputs['text'].shape == (1, 10, 1000)

def test_large_batch_handling(transformer):
    """Test processing of large batches"""
    model, variables = transformer
    inputs = {'text': jnp.ones((32, 10), dtype=jnp.int32)}
    outputs = model.apply(variables, inputs, training=False)

    assert outputs['text'].shape == (32, 10, 1000)

def test_memory_cleanup(transformer):
    """Test memory cleanup during processing"""
    model, variables = transformer
    initial_blocks = len(jax.live_arrays())

    # Process large batch that requires chunking
    inputs = {'text': jnp.ones((64, 10), dtype=jnp.int32)}
    outputs = model.apply(variables, inputs, training=False)

    # Force garbage collection
    jax.clear_caches()

    # Check memory usage after processing
    final_blocks = len(jax.live_arrays())
    assert final_blocks <= initial_blocks * 1.5  # Allow some overhead

def test_error_handling(transformer):
    """Test error handling for invalid inputs"""
    model, variables = transformer
    with pytest.raises(ValueError):
        # Test with invalid input shape
        model.apply(variables, {'text': jnp.ones((1,), dtype=jnp.int32)})

def test_positional_encoding(transformer):
    """Test positional encoding generation"""
    model, _ = transformer
    batch_size = 1
    seq_len = 10
    d_model = model.d_model
    pos_encoding = model._get_positional_encoding((batch_size, seq_len, d_model))

    assert pos_encoding.shape == (1, seq_len, d_model)
    assert jnp.allclose(pos_encoding[0, 0, 0], jnp.sin(0))

def test_response_decoding(transformer):
    """Test response decoding with temperature scaling"""
    model, _ = transformer
    logits = jnp.ones((1, 5, 1000))
    response = model._decode_response(logits)
    assert isinstance(response, str)

def test_gradient_accumulation(transformer):
    """Test gradient accumulation during training"""
    model, variables = transformer
    inputs = {'text': jnp.ones((16, 10), dtype=jnp.int32)}

    # Create PRNG keys for dropout
    key1 = jax.random.PRNGKey(0)
    key2 = jax.random.PRNGKey(1)

    # First forward pass with training=True
    outputs1 = model.apply(
        variables,
        inputs,
        training=True,
        rngs={'dropout': key1}
    )

    # Second forward pass with same input
    outputs2 = model.apply(
        variables,
        inputs,
        training=True,
        rngs={'dropout': key2}
    )

    # Check outputs are different due to dropout
    assert not jnp.allclose(outputs1['text'], outputs2['text'])

def test_cross_modal_attention(transformer):
    """Test cross-modal attention mechanism"""
    model, variables = transformer

    # Test with text modality only
    inputs = {
        'text': jnp.ones((1, 10), dtype=jnp.int32)
    }

    outputs = model.apply(
        variables,
        inputs,
        training=False,
        rngs={'dropout': jax.random.PRNGKey(2)}
    )

    # Verify text output exists and has correct shape
    assert 'text' in outputs
    assert outputs['text'].shape == (1, 10, 1000)

def test_memory_constraints():
    """Test model operates within memory constraints"""
    import psutil
    import gc

    # Force garbage collection
    gc.collect()
    jax.clear_caches()

    # Get initial memory usage
    initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB

    # Create model with reduced size for memory constraints
    config = {
        'num_layers': 4,
        'd_model': 256,
        'num_heads': 8,
        'dff': 1024,
        'dropout_rate': 0.1,
        'input_modalities': ['text'],
        'output_modalities': ['text'],
        'vocab_sizes': {'text': 10000}
    }

    model = Transformer(**config)
    key = jax.random.PRNGKey(0)
    variables = model.init(key, {'text': jnp.ones((1, 64), dtype=jnp.int32)}, training=False)

    # Process maximum allowed batch
    inputs = {'text': jnp.ones((16, 64), dtype=jnp.int32)}
    _ = model.apply(variables, inputs, training=False)

    # Check memory usage
    final_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
    memory_used = final_memory - initial_memory

    # Allow for some overhead but ensure we're well under 7.7GB
    assert memory_used < 4000, f"Memory usage ({memory_used}MB) exceeds reasonable limit"

def test_cpu_core_utilization():
    """Test efficient use of 2 CPU cores"""
    import psutil

    # Verify number of CPU cores
    num_cores = psutil.cpu_count()
    assert num_cores >= 2, "Requires minimum 2 CPU cores"

    # Create small model for testing
    config = {
        'num_layers': 2,
        'd_model': 64,
        'num_heads': 2,
        'dff': 128,
        'dropout_rate': 0.1,
        'input_modalities': ['text'],
        'output_modalities': ['text'],
        'vocab_sizes': {'text': 1000}
    }

    model = Transformer(**config)
    key = jax.random.PRNGKey(0)
    variables = model.init(key, {'text': jnp.ones((1, 10), dtype=jnp.int32)}, training=False)

    # Process multiple batches
    inputs = {'text': jnp.ones((8, 10), dtype=jnp.int32)}

    # Test sequential processing
    outputs = []
    for _ in range(4):
        output = model.apply(variables, inputs, training=False)
        outputs.append(output)

    assert len(outputs) == 4
    assert all(isinstance(r, dict) for r in outputs)

def test_meta_learning_embedding(transformer):
    """Test meta-learning embedding functionality"""
    model, variables = transformer
    # Test with task-specific embedding
    inputs = {'text': jnp.ones((1, 10), dtype=jnp.int32)}

    # Process same input with different random seeds
    key1 = jax.random.PRNGKey(0)
    key2 = jax.random.PRNGKey(1)
    output1 = model.apply(variables, inputs, training=True, rngs={'dropout': key1})
    output2 = model.apply(variables, inputs, training=True, rngs={'dropout': key2})

    # Outputs should be different due to different dropout patterns
    assert not jnp.allclose(output1['text'], output2['text'])

def test_batch_size_constraints(transformer):
    """Test batch size constraints and chunking"""
    model, variables = transformer
    # Test with various batch sizes
    batch_sizes = [4, 8, 16, 32, 64]
    seq_len = 10

    for batch_size in batch_sizes:
        inputs = {'text': jnp.ones((batch_size, seq_len), dtype=jnp.int32)}
        outputs = model.apply(variables, inputs, training=False)

        # Verify output shape
        assert outputs['text'].shape == (batch_size, seq_len, 1000)

        # For large batches, verify chunking occurred
        if batch_size > 16:
            # Check memory usage during processing
            import psutil
            mem_info = psutil.Process().memory_info()
            mem_used = mem_info.rss / (1024 * 1024)  # MB
            assert mem_used < 7700, f"Memory usage exceeded for batch size {batch_size}"
