# Guardrail — AI Hallucination Audit Suite

A professional hallucination detection API for AI responses.

## Architecture (Cascading Audit Pipeline)

1. **Extractor (SpaCy)** — Named entity mismatch detection (local, ~10ms)
2. **Scorer (Cross-Encoder via ONNX)** — Semantic faithfulness score (local, ~10ms)
3. **Judge (Groq Llama 3.3 70B)** — Natural language explanation (API, ~1s)
4. **Memory (Supabase)** — Audit log persistence + leaderboard

## Stack

- **Backend:** FastAPI + Python 3.13
- **NLP:** SpaCy (NER) + fastembed (ONNX cross-encoder, no PyTorch)
- **LLM:** Groq Llama-3.3-70b-versatile (zero cost, sub-second)
- **Database:** Supabase (Postgres)
- **Hosting:** Render (free tier — 512 MB RAM)

## Endpoints

| Method | Path          | Description                   |
| ------ | ------------- | ----------------------------- |
| GET    | /health       | Service health check          |
| POST   | /audit        | Run full hallucination audit  |
| GET    | /leaderboard  | Aggregated model scores       |

## Local Development

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env  # fill in your keys
uvicorn main:app --reload
```

Open `http://localhost:8000/docs` for the interactive API docs.

## License

MIT
