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
    version="1.1.0",
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
    return {"status": "ok", "service": "guardrail-api"}

@app.post("/audit", response_model=AuditResponse)
def audit(req: AuditRequest):
    """
    Cascading audit pipeline with Intelligent Veto logic.
    """
    start = time.time()
    logger.info("Audit started for model_name=%s", req.model_name)

    # Stage 1: SpaCy NER mismatch
    source_ents = {e.text.lower() for e in nlp(req.source_text).ents}
    response_ents = {e.text.lower() for e in nlp(req.ai_response).ents}
    entity_mismatches = sorted(response_ents - source_ents)
    logger.info("Stage 1 complete. Entity mismatches: %s", entity_mismatches)

    # Stage 2: Cross-Encoder score
    raw_score = float(next(cross_encoder.rerank(req.source_text, [req.ai_response])))
    score = max(0.0, min(1.0, (raw_score + 10) / 20))
    logger.info("Stage 2 complete. raw=%.4f normalized=%.4f", raw_score, score)

    # Initial logic: Score-based flagging
    if raw_score < 5.0 or len(entity_mismatches) >= 2:
        flag = "hallucination"
    elif raw_score < 8.0:
        flag = "uncertain"
    else:
        flag = "grounded"

    # Stage 3: LLM judge (with Veto logic for Corrective Truths)
    llm_explanation: str | None = None
    if req.use_llm_judge and flag in ("hallucination", "uncertain"):
        try:
            mismatch_str = (
                f"Entity mismatches found: {entity_mismatches}. "
                if entity_mismatches else ""
            )
            
            # THE INTELLECTUAL PROMPT: Distinguish between lying and correcting a lie
            prompt = (
                "You are a Senior AI Auditor. Compare the Source and the Response.\n\n"
                f"Source: \"{req.source_text[:800]}\"\n"
                f"Response: \"{req.ai_response[:800]}\"\n\n"
                "JUDGING RULES:\n"
                "1. If the Source contains a trick or false premise (e.g., 'Blue Golden Gate Bridge') "
                "and the AI response correctly fixes it, start your response with [CORRECTIVE_TRUTH].\n"
                "2. If the user's prompt asks for a fake citation or non-existent regulation, "
                "EXPLICITLY PRAISE the AI for refusing to invent it.\n"
                "3. If the AI is truly hallucinating or adding unverified info not in the source, "
                "start your response with [HALLUCINATION].\n"
                "Explain your reasoning in 2 concise sentences."
            )
            
            raw_result = llm.invoke(prompt).content
            
            # --- VETO LOGIC ---
            if "[CORRECTIVE_TRUTH]" in raw_result:
                flag = "grounded"  # OVERRIDE the flag because the AI was smarter than the prompt
                llm_explanation = raw_result.replace("[CORRECTIVE_TRUTH]", "").strip()
                logger.info("Stage 3 Veto: Corrective truth detected. Overriding flag to 'grounded'.")
            else:
                llm_explanation = raw_result.replace("[HALLUCINATION]", "").strip()
                logger.info("Stage 3 complete. Hallucination confirmed by judge.")

        except Exception as exc:
            logger.warning("Stage 3 LLM call failed: %s", exc)
            llm_explanation = "LLM judge unavailable."

    latency_ms = int((time.time() - start) * 1000)

    # Stage 4: Persist
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
        logger.info("Stage 4 complete. Row persisted.")
    except Exception as exc:
        logger.error("Supabase insert failed: %s", exc)

    return AuditResponse(**row)

@app.get("/leaderboard")
def leaderboard():
    """Aggregate audit_logs by model_name."""
    try:
        result = supabase.table("audit_logs").select("model_name, flag, score").execute()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}") from exc

    rows = result.data
    if not rows:
        return []

    stats: dict = defaultdict(lambda: {"total": 0, "hallucinations": 0, "uncertain": 0, "scores": []})
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
        board.append({
            "model_name": model,
            "total_audits": total,
            "hallucination_rate": round(s["hallucinations"] / total, 3),
            "uncertain_rate": round(s["uncertain"] / total, 3),
            "avg_score": round(sum(s["scores"]) / len(s["scores"]), 3),
        })

    return sorted(board, key=lambda x: x["hallucination_rate"], reverse=True)
