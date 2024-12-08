"""Logging configuration for the VisionAI model."""
import logging
import sys

# Configure logger
logger = logging.getLogger('visionai')
logger.setLevel(logging.INFO)

# Create console handler
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)

# Add handler to logger
logger.addHandler(handler)
