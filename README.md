# Portfolio Chatbot

A sophisticated chatbot implementation using JAX and Sonnet for machine learning processing, featuring multi-modal transformer architecture and emotion-aware responses.

## Setup Instructions

1. Create and activate a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Start the FastAPI server:
```bash
uvicorn src.api:app --reload
```

4. Open index.html in a web browser to interact with the chatbot

## Architecture Overview

### Frontend
- JavaScript-based interactive chat interface
- Dynamic emoji rendering for emotional expression
- Real-time message processing
- Responsive design

### Backend
- FastAPI server for API endpoints
- JAX/Sonnet ML pipeline
- Custom data processing module
- Multi-modal transformer model

## Dependencies

### Python Packages
- JAX/Sonnet: Machine learning framework
- FastAPI: Backend API framework
- Pytest: Testing framework
- Uvicorn: ASGI server

### JavaScript
- Modern web browser with ES6+ support

## Environment Requirements

- Python 3.12+
- Node.js 18+ (for development)
- Modern web browser
- 2+ CPU cores recommended
- 8GB+ RAM recommended

## Development

Run tests:
```bash
pytest tests/
```

## License

MIT License
