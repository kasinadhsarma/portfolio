"""
Custom exceptions for the VisionAI chatbot
"""

class ModelInferenceError(Exception):
    """Raised when model inference fails"""
    pass

class InputValidationError(Exception):
    """Raised when input validation fails"""
    pass

class TokenizationError(Exception):
    """Raised when text tokenization fails"""
    pass

class DataProcessingError(Exception):
    """Raised when data processing fails"""
    pass
