"""
Tests for API endpoints
"""
import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_chat_endpoint():
    message = {"text": "Hello"}
    response = client.post("/chat", json=message)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "emotion" in data
    assert isinstance(data["response"], str)
    assert isinstance(data["emotion"], str)

def test_chat_endpoint_empty_message():
    message = {"text": ""}
    response = client.post("/chat", json=message)
    assert response.status_code == 200

def test_chat_endpoint_invalid_request():
    message = {"invalid": "data"}
    response = client.post("/chat", json=message)
    assert response.status_code == 422  # Validation error
