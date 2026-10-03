# Quick start

Install the package and point it at your retriever:

```bash
pip install ragflip
ragflip check traces/ --retriever myapp.search:retrieve --k 8
ragflip repair failure.json --retriever myapp.search:retrieve --judge overlap
```

The trace format and the retriever signature are in [README.md](README.md).

## Local server

```bash
pip install -r requirements.txt
```

Set a key only if you want the OpenAI judge or the ask endpoint:

```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Start it:

```bash
ragflip serve
```

The server listens on `http://localhost:8000`. Interactive docs are at `http://localhost:8000/docs`.

```bash
curl -X POST "http://localhost:8000/upload" -F "files=@your_document.pdf"
curl -X POST "http://localhost:8000/ask" -H "Content-Type: application/json" -d "{\"question\": \"What is this document about?\"}"
```

If port 8000 is taken:

```bash
uvicorn app:app --port 8001
```
