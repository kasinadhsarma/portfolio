import sonnet as snt
import jax.numpy as np
from data import training_data, utils
from models.portfolio import (
    Transformer,
    MetaLearningEmbedding,
    CrossModalAttention,
    AdaptiveFeedForward,
)

class Portfolio:
    def __init__(self):
        self.sonnet = snt
        self.transformer = Transformer()
        self.attention = CrossModalAttention()
        self.feed_forward = AdaptiveFeedForward()
        self.embeddings = MetaLearningEmbedding()
        self.data = training_data
        self.utils = utils
        self.np = np
        self.chatbot_model = Transformer(
            embedding=self.embeddings,
            attention=self.attention,
            feed_forward=self.feed_forward
        )
