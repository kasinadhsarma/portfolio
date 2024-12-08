from sonnet import Sonnet
from models.portfolio import (
    Transformer,
    AdaptiveMultiModalEncoderLayer,
    MetaLearningEmbedding,
    CrossModalAttention,
    AdaptiveFeedForward,
    MetaLearningEmbedding
)
from data import data, utils
import jax as np

class Portfolio:
    def __init__(self):
        self.sonnet = Sonnet
        self.transformer = Transformer()
        self.attention = CrossModalAttention()
        self.feed_forward = AdaptiveFeedForward()
        self.embeddings = MetaLearningEmbedding()
        self.data = data
        self.utils = utils
        self.np = np