# Quick Start Guide

The experimental product is the CLI, not this server demo:

```bash
rag-debugger repair TRACE --retriever module:function --judge overlap
```

See README.md. The steps below start the older upload/ask API.

## 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

## 2. Set OpenAI API Key

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## 3. Start the Server

**Option A: Using the startup script**
```powershell
python start_server.py
```

**Option B: Using uvicorn directly**
```powershell
uvicorn app:app --reload
```

The server will start at `http://localhost:8000`

## 4. Test the API

### Upload a document:
```powershell
curl -X POST "http://localhost:8000/upload" -F "files=@your_document.pdf"
```

### Ask a question:
```powershell
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d '{\"question\": \"What is this document about?\"}'
```

### View API docs:
Open `http://localhost:8000/docs` in your browser for interactive API documentation.

## Troubleshooting

- **"OPENAI_API_KEY not set"**: Make sure you've set the environment variable in the same terminal session
- **Import errors**: Make sure all dependencies are installed: `pip install -r requirements.txt`
- **Port already in use**: Change the port in `start_server.py` or use `uvicorn app:app --port 8001`
