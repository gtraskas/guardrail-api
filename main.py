"""
Guardrail — AI Hallucination Audit Suite
Cascading pipeline: SpaCy NER -> Cross-Encoder -> Groq Llama 3 -> Supabase
"""

import time
import os
import logging
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastembed.rerank.cross_encoder import TextCrossEncoder
import spacy
from supabase import create_client, Client
from dotenv import load_dotenv
from langchain_groq import ChatGroq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

# --- App ---------------------------------------------------------------------

app = FastAPI(
    title="Guardrail — Hallucination Audit API",
    description="Cascading NER + Cross-Encoder + LLM hallucination detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Startup: load heavy models once -----------------------------------------

logger.info("Loading Cross-Encoder (ONNX) model...")
# fastembed runs the same ms-marco MiniLM cross-encoder via ONNX runtime,
# avoiding the ~2GB torch+CUDA dependency tree that broke Render free-tier deploys.
cross_encoder = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
logger.info("Cross-Encoder loaded.")

logger.info("Loading SpaCy NER model...")
nlp = spacy.load("en_core_web_sm")
logger.info("SpaCy loaded.")

supabase: Client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"],
)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.environ["GROQ_API_KEY"],
    temperature=0.1,
)

# --- Models ------------------------------------------------------------------


class AuditRequest(BaseModel):
    source_text: str
    ai_response: str
    model_name: str = "unknown"
    use_llm_judge: bool = True


class AuditResponse(BaseModel):
    score: float
    flag: str
    entity_mismatches: list[str]
    llm_explanation: str | None
    latency_ms: int
    model_name: str
    source_text: str
    ai_response: str


# --- Routes ------------------------------------------------------------------


@app.get("/health")
def health():
    """Uptime monitor ping target."""
    return {"status": "ok", "service": "guardrail-api"}


@app.post("/audit", response_model=AuditResponse)
def audit(req: AuditRequest):
    """
    Cascading hallucination audit pipeline:
      Stage 1 - SpaCy NER:      detect entity mismatches (names, dates, numbers)
      Stage 2 - Cross-Encoder:  semantic faithfulness score
      Stage 3 - Groq Llama 3:   natural language explanation (only if flagged)
      Stage 4 - Supabase:       persist result
    """
    start = time.time()
    logger.info("Audit started for model_name=%s", req.model_name)

    # Stage 1: SpaCy NER mismatch
    source_ents = {e.text.lower() for e in nlp(req.source_text).ents}
    response_ents = {e.text.lower() for e in nlp(req.ai_response).ents}
    entity_mismatches = sorted(response_ents - source_ents)
    logger.info("Stage 1 complete. Entity mismatches: %s", entity_mismatches)

    # Hard-fail shortcut: 2+ fabricated entities is a strong hallucination signal
    hard_fail = len(entity_mismatches) >= 2

    # Stage 2: Cross-Encoder score (ONNX via fastembed)
    # ms-marco cross-encoder raw range is roughly [-10, +10]; we use the raw score
    # for thresholds (well-separated for relevance) and a normalized 0-1 score for UI.
    raw_score = float(next(cross_encoder.rerank(req.source_text, [req.ai_response])))
    score = max(0.0, min(1.0, (raw_score + 10) / 20))
    logger.info("Stage 2 complete. raw=%.4f normalized=%.4f", raw_score, score)

    if hard_fail or raw_score < 5.0:
        flag = "hallucination"
    elif raw_score < 8.0:
        flag = "uncertain"
    else:
        flag = "grounded"

    # Stage 3: LLM judge (only if flagged)
    llm_explanation: str | None = None
    if req.use_llm_judge and flag in ("hallucination", "uncertain"):
        try:
            mismatch_str = (
                f"Entity mismatches found: {entity_mismatches}. "
                if entity_mismatches
                else ""
            )
            prompt = (
                "You are a hallucination detection expert. "
                f"The semantic faithfulness score between the source and the AI response "
                f"is {score:.2f} (scale 0-1). {mismatch_str}"
                f'Source text: "{req.source_text[:800]}"\n'
                f'AI response: "{req.ai_response[:800]}"\n\n'
                "In 2-3 concise sentences, explain exactly what was hallucinated "
                "or why the response diverges from the source."
            )
            llm_explanation = llm.invoke(prompt).content
            logger.info("Stage 3 complete. LLM explanation generated.")
        except Exception as exc:
            logger.warning("Stage 3 LLM call failed: %s", exc)
            llm_explanation = "LLM judge unavailable."

    latency_ms = int((time.time() - start) * 1000)

    # Stage 4: persist
    row = {
        "source_text": req.source_text,
        "ai_response": req.ai_response,
        "model_name": req.model_name,
        "score": score,
        "flag": flag,
        "entity_mismatches": entity_mismatches,
        "llm_explanation": llm_explanation,
        "latency_ms": latency_ms,
    }
    try:
        supabase.table("audit_logs").insert(row).execute()
        logger.info("Stage 4 complete. Row persisted to Supabase.")
    except Exception as exc:
        logger.error("Supabase insert failed: %s", exc)

    logger.info(
        "Audit complete. flag=%s score=%.4f latency=%dms",
        flag,
        score,
        latency_ms,
    )

    return AuditResponse(**row)


@app.get("/leaderboard")
def leaderboard():
    """Aggregate audit_logs by model_name. Worst hallucination rate first."""
    try:
        result = supabase.table("audit_logs").select("model_name, flag, score").execute()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}") from exc

    rows = result.data
    if not rows:
        return []

    stats: dict = defaultdict(
        lambda: {"total": 0, "hallucinations": 0, "uncertain": 0, "scores": []}
    )
    for r in rows:
        m = r["model_name"]
        stats[m]["total"] += 1
        stats[m]["scores"].append(r["score"])
        if r["flag"] == "hallucination":
            stats[m]["hallucinations"] += 1
        elif r["flag"] == "uncertain":
            stats[m]["uncertain"] += 1

    board = []
    for model, s in stats.items():
        total = s["total"]
        board.append(
            {
                "model_name": model,
                "total_audits": total,
                "hallucination_rate": round(s["hallucinations"] / total, 3),
                "uncertain_rate": round(s["uncertain"] / total, 3),
                "avg_score": round(sum(s["scores"]) / len(s["scores"]), 3),
            }
        )

    return sorted(board, key=lambda x: x["hallucination_rate"], reverse=True)
