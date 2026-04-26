# AI Portfolio Implementation Guide — 2026

---

## VS Code "Project Zero" Setup (Do This for Every Project)

Repeat these steps for every project folder you created (`/guardrail`, `/research-agent`, `/persona-factory`, `/billing-detective`). This ensures each project is self-contained and easy to manage.

### 1. Initialize the Environment

Open your terminal in the specific project folder and run:

```bash
python -m venv venv
source venv/bin/activate  # (Mac)
# or
.\venv\Scripts\activate  # (Windows)
```

### 2. Create the Configuration Files

Create these three empty files in the root of the folder:

- `main.py` (The Brain)
- `.env` (The Secrets)
- `requirements.txt` (The Needs)

### 3. The Copilot Initialization Prompt

Copy/paste this into Copilot Chat:

> Act as a Senior Python Developer. Scaffolding a FastAPI project for [Project Name].\n\nWrite a main.py with a /health check and a CORS setup for Lovable.\n\nWrite a requirements.txt including fastapi, uvicorn, python-dotenv, supabase, and langchain-groq.\n\nSetup a basic logger so I can see API calls in the Render logs.

---

## The "Cloud Master" Setup (Per Project)

Even if you use one Supabase account, you should treat the data separately for each project.

### 1. Supabase (The Memory)

- Don't mix your tables. Use Postgres Schemas or prefix your tables (e.g., `p1_audit_logs`, `p2_document_chunks`).
- **Master Prompt:**
  > Write the SQL to create a private schema for Project [Number] and the tables needed for [Project Goal]. Ensure RLS (Row Level Security) is enabled.

### 2. Render (The Server)

- Create four separate Web Services on Render (one per project).
- Each project will have its own unique URL (e.g., `guardrail-api.onrender.com`).
- For every service, set the Environment Variables for that specific project's Supabase keys and Groq key.

### 3. Lovable (The UI)

- In Lovable, create four separate projects.
- **Master Prompt:**
  > I am building the frontend for [Project Name]. I have a Render API at [Your-URL]. Create a base layout with a navigation bar, a 'Connection Status' indicator that pings my API's /health endpoint, and a theme that matches [Project Industry, e.g., Medical or Research].

---

## The "Portfolio Hub" Strategy

Once all four separate setups are done, you need a way to link them.

**The Professional Move:** Create a fifth, very simple "Landing Page" project in Lovable. This is your Main Portfolio. It should feature four "cards" — each card has a screenshot of the project and a link that opens the specific project's Lovable URL.

---

## Why Groq? (Consultant Rationale)

All four projects are architected to use Groq’s Llama 3 models for every LLM task. Groq delivers industry-leading speed (sub-second responses), zero API cost, and no vendor lock-in. This stack can be ported to a private server for data privacy, and avoids the “API tax” of paid providers. For embeddings, we use Hugging Face (free) or sentence-transformers locally, since Groq is text-only.

**Client pitch:**
> "I architected this using Llama 3 via Groq to demonstrate how to build high-performance AI tools without 'API Tax' or vendor lock-in. This stack allows for sub-second latency and ensures that the system can be ported to a private local server if data privacy becomes the primary concern."

---

## Project 1: Hallucination Guardrail Audit Suite

### Project 1: Copilot Logic Prompt

> Create a FastAPI main.py for an AI Hallucination Audit tool. Use the fastembed library (`TextCrossEncoder` with `Xenova/ms-marco-MiniLM-L-6-v2`) to score the similarity between a 'source_text' and an 'ai_response'. If the score is < 0.5, flag it as a 'hallucination'. Include a POST /audit endpoint that saves the result to a Supabase table called audit_logs and a GET /leaderboard endpoint that aggregates scores by model_name. Use Groq's Llama-3.3-70b-versatile as a secondary 'Judge' to explain WHY it might be a hallucination.

### Project 1: Lovable UI Prompt

> Build a professional AI Hallucination Guardrail dashboard. Features: 1. Two side-by-side textareas for 'Source' and 'AI Response'. 2. A 'Run Audit' button that shows a 'Waking up the engine' loading bar for 30 seconds on the first click. 3. A radial gauge showing the faithfulness score (0-100%). 4. A red/green diff view showing which words were hallucinated. 5. A secondary tab for a 'Model Leaderboard' that fetches data from my Render API.

### Project 1: What it does

**In plain English:**
Think of this project as a lie detector for AI. Even the smartest AI models sometimes "hallucinate"—they make up facts that aren't in the original text you gave them. This tool acts as a neutral third party that checks the AI's homework.

**How it works (Professional Audit Workflow):**

In a professional "Guardrail" pipeline, you don't just run everything at once. You follow a cascading order—moving from the cheapest/fastest tool to the most expensive/complex one. This ensures that if the first model finds a glaring error, you don't waste time or money running the others.

**The Audit Workflow:**

1. **The Extractor (SpaCy):** Fact-Checking. SpaCy pulls out every name, date, and dollar amount from both texts. If the AI response contains "$5,000" but the source says "$500," the audit can stop here with a "Hard Fail." (Local & near-instant)
2. **The Scorer (Cross-Encoder):** Semantic Integrity. If the facts match, the Cross-Encoder checks if the meaning was preserved. It catches logic errors and reversals. (High-precision, still local)
3. **The Judge (Groq Llama 3):** Explanation & Nuance. If needed, Llama 3 synthesizes the data into a clear explanation: "The AI hallucinated a different date and flipped the subject of the sentence." (External API, only called if needed)
4. **The Memory (Supabase):** Every check is saved in a database, so you can build a leaderboard to see which AI models are most honest.

**Why this order matters:**

- **Speed:** SpaCy runs on your server (0.01s). The Cross-Encoder is a local model (0.5s). Groq is an external API call (1-2s). Starting with the fast ones makes the app feel responsive.
- **Evidence-Based Judging:** By the time you reach Step 3 (Llama 3), you aren't just asking it "Did this hallucinate?"—you're giving it evidence: "The Extractor found a mismatch and the Scorer gave a 0.2. Explain why." This makes the Judge much more accurate.
- **Cost Optimization:** If SpaCy finds a 100% factual mismatch, you skip the expensive LLM call entirely. This is a senior-level, cost-aware design.

---

### Step 1 — Get the Dataset

**HaluEval** is a free Hugging Face dataset of human-labeled hallucinated vs grounded QA pairs.

```bash
pip install datasets pandas supabase-py
```

```python
from datasets import load_dataset
import pandas as pd

ds = load_dataset("pminervini/HaluEval", "qa_samples")
df = pd.DataFrame(ds["data"])

# HaluEval columns: knowledge, question, right_answer, hallucinated_answer
# Build 50 pass + 50 fail benchmark rows
pass_rows = df.sample(50).assign(label="pass")[["knowledge", "right_answer"]].rename(columns={"knowledge": "source_text", "right_answer": "ai_response"})
fail_rows = df.sample(50).assign(label="fail")[["knowledge", "hallucinated_answer"]].rename(columns={"knowledge": "source_text", "hallucinated_answer": "ai_response"})

benchmarks = pd.concat([pass_rows, fail_rows]).reset_index(drop=True)
benchmarks["model_name"] = "baseline"
print(benchmarks.head())
```

---

### Step 2 — Project 1: Supabase Schema

Run this SQL in your Supabase SQL editor:

```sql
create table benchmarks (
  id uuid primary key default gen_random_uuid(),
  source_text text not null,
  ai_response text not null,
  label text check (label in ('pass', 'fail')),
  model_name text default 'baseline',
  created_at timestamptz default now()
);

create table audit_logs (
  id uuid primary key default gen_random_uuid(),
  source_text text not null,
  ai_response text not null,
  model_name text not null,
  score float not null,
  flag text check (flag in ('grounded', 'uncertain', 'hallucination')),
  entity_mismatches jsonb default '[]',
  latency_ms int,
  created_at timestamptz default now()
);

alter table audit_logs enable row level security;
create policy "public read" on audit_logs for select using (true);
create policy "service insert" on audit_logs for insert with check (true);
```

---

### Step 3 — Project 1: Seed the Database

```python
import os
from supabase import create_client

supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

records = benchmarks.to_dict(orient="records")
supabase.table("benchmarks").insert(records).execute()
print(f"Seeded {len(records)} benchmark rows")
```

---

### Step 4 — Project 1: FastAPI Backend

```bash
pip install fastembed langchain-groq groq fastapi uvicorn python-dotenv supabase spacy
python -m spacy download en_core_web_sm
```

**`main.py`:**

```python
import time, os, json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastembed.rerank.cross_encoder import TextCrossEncoder
import spacy
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

nlp = spacy.load("en_core_web_sm")
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

# Load once at startup — Render caches this between requests
cross_encoder = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
nlp = spacy.load("en_core_web_sm")
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])

# Example: Use Groq Llama 3 for any LLM-based judging (if needed)
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama3-70b-8192", groq_api_key=os.environ["GROQ_API_KEY"])


class AuditRequest(BaseModel):
    source_text: str
    ai_response: str
    model_name: str = "unknown"


@app.post("/audit")
def audit(req: AuditRequest):
    start = time.time()

    # Cross-encoder score: 0.0 (unrelated) to 1.0 (fully supported)
    score = float(next(cross_encoder.rerank(req.source_text, [req.ai_response])))

    if score < 0.5:
        flag = "hallucination"
    elif score < 0.75:
        flag = "uncertain"
    else:
        flag = "grounded"

    # NER mismatch check
    source_ents = {e.text.lower() for e in nlp(req.source_text).ents}
    response_ents = {e.text.lower() for e in nlp(req.ai_response).ents}
    entity_mismatches = list(response_ents - source_ents)

    latency_ms = int((time.time() - start) * 1000)

    row = {
        "source_text": req.source_text,
        "ai_response": req.ai_response,
        "model_name": req.model_name,
        "score": score,
        "flag": flag,
        "entity_mismatches": entity_mismatches,
        "latency_ms": latency_ms,
    }
    supabase.table("audit_logs").insert(row).execute()

    return row


@app.get("/leaderboard")
def leaderboard():
    # Supabase doesn't do GROUP BY directly — pull all and aggregate in Python
    result = supabase.table("audit_logs").select("model_name, flag, score").execute()
    rows = result.data

    from collections import defaultdict
    stats = defaultdict(lambda: {"total": 0, "hallucinations": 0, "scores": []})
    for r in rows:
        m = r["model_name"]
        stats[m]["total"] += 1
        stats[m]["scores"].append(r["score"])
        if r["flag"] == "hallucination":
            stats[m]["hallucinations"] += 1

    leaderboard = []
    for model, s in stats.items():
        leaderboard.append({
            "model_name": model,
            "total_audits": s["total"],
            "hallucination_rate": round(s["hallucinations"] / s["total"], 3),
            "avg_score": round(sum(s["scores"]) / len(s["scores"]), 3),
        })

    return sorted(leaderboard, key=lambda x: x["hallucination_rate"], reverse=True)
```

**`requirements.txt`:**

```txt
fastapi
uvicorn
fastembed
langchain-groq
groq
spacy
supabase
python-dotenv
pydantic
```

**`Procfile` for Render:**

```txt
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

Set env vars in Render dashboard: `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`

---

### Step 5 — Project 1: Lovable Frontend

Tell Lovable to build the following components:

**Page layout:**

- Two textarea inputs side by side: "Source Text" (left), "AI Response" (right)
- A text input for "Model Name" above the textareas
- A submit button "Run Audit"
- Before the first successful backend response, show a loading state: `Waking up the AI engine...` with a slow progress bar that moves toward 90%
- Below: a score gauge (radial bar chart using recharts) colored red/amber/green based on score
- Below the gauge: a diff view using the `diff-match-patch` npm package — highlight tokens in AI response that do NOT appear in source text in red
- A list labeled "Entity Mismatches" showing the `entity_mismatches` array from the API response
- A draggable file upload zone that accepts .txt and .json files, parses them on client, populates the textareas

**Leaderboard page (`/leaderboard`):**

- Fetch GET `/leaderboard` from your Render URL on page load
- Render a sortable table: Rank | Model Name | Total Audits | Hallucination Rate | Avg Score
- Use Supabase client JS to subscribe to `audit_logs` INSERT events and refresh the table live
- Add a fixed bottom-right button "💰 ROI Case Study" that opens a modal with a client impact story

**Install in Lovable:**

```txt
diff-match-patch
recharts
```

**Supabase realtime in Lovable:**

```js
import { createClient } from "@supabase/supabase-js"
const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY)

supabase
  .channel("audit_logs")
  .on("postgres_changes", { event: "INSERT", schema: "public", table: "audit_logs" }, (payload) => {
    // refresh leaderboard
  })
  .subscribe()
```

---

## Project 2: RAG Document Intelligence Suite

### Project 2: Copilot Logic Prompt

> Build a FastAPI RAG backend. Use pdfplumber for text and table extraction. Use sentence-transformers (all-MiniLM-L6-v2) for embeddings—remember to set the Supabase vector dimension to 384. Create an /ingest endpoint to process PDFs and a /query endpoint that uses Groq's Llama-3-70b to answer questions based on the retrieved context. The answer must include page-number citations.

### Project 2: Lovable UI Prompt

> Create a document intelligence interface. Left sidebar: File upload zone and a list of uploaded PDFs with checkboxes. Right side: A full-height chat window. When the AI answers, include a small 'Sources' dropdown under the message that shows the exact text chunks used from the PDF and their page numbers. Use a streaming text effect for the AI response.

### Project 2: What it does

Ingests PDFs, DOCX, and TXT files into a pgvector knowledge base. Users chat with their documents and get answers with cited source chunks and page numbers. Multi-turn conversation memory included.

---

### Step 1 — Enable pgvector in Supabase

```sql
create extension if not exists vector;

create table documents (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id) on delete cascade,
  filename text not null,
  file_type text check (file_type in ('pdf', 'docx', 'txt')),
  page_count int,
  chunk_count int default 0,
  uploaded_at timestamptz default now()
);

create table document_chunks (
  id uuid primary key default gen_random_uuid(),
  document_id uuid references documents(id) on delete cascade,
  chunk_text text not null,
  embedding vector(384), -- all-MiniLM-L6-v2 uses 384 dims
  page_number int,
  chunk_index int,
  created_at timestamptz default now()
);

create or replace function match_documents(
  query_embedding vector(1536),
  match_count int,
  doc_ids uuid[]
) returns table (
  id uuid, chunk_text text, page_number int, document_id uuid, similarity float
) language plpgsql as $$
begin
  return query
  select
    dc.id, dc.chunk_text, dc.page_number, dc.document_id,
    1 - (dc.embedding <=> query_embedding) as similarity
  from document_chunks dc
  where dc.document_id = any(doc_ids)
  order by dc.embedding <=> query_embedding
  limit match_count;
end;
$$;

alter table documents enable row level security;
create policy "own docs" on documents using (auth.uid() = user_id);
alter table document_chunks enable row level security;
create policy "own chunks" on document_chunks
  using (document_id in (select id from documents where user_id = auth.uid()));
```

---

### Step 2 — Get Free Demo Documents

For your portfolio demo, pre-load 5 public domain PDFs so visitors can try without uploading:

- **Annual report:** Any public company 10-K from sec.gov/cgi-bin/browse-edgar
- **Legal contract:** Sample NDA from SEC EDGAR full-text search
- **Research paper:** arxiv.org
- **Product manual:** Open-source project documentation PDF
- **Policy document:** US government agency policy PDF from gao.gov or federalregister.gov

Name them clearly in Supabase as `is_demo = true` so the UI shows them as default options.

---

### Step 3 — Project 2: FastAPI Backend

```bash
pip install pymupdf pdfplumber python-docx langchain-groq groq sentence-transformers supabase tiktoken fastapi uvicorn python-dotenv
```

**`main.py`:**

```python
import os, json, uuid
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import fitz
import pdfplumber
from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
embeddings = SentenceTransformer("all-MiniLM-L6-v2")  # or use Hugging Face Inference API
llm = ChatGroq(model="llama3-70b-8192", groq_api_key=os.environ["GROQ_API_KEY"])
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)


def extract_text(file_bytes: bytes, filename: str) -> list[dict]:
    pages = []
    if filename.endswith(".pdf"):
        try:
          import io
          with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
              text = page.extract_text() or ""
              tables = page.extract_tables() or []
              table_blocks = []
              for table in tables:
                try:
                  rows = [row for row in table if row]
                  if not rows:
                    continue
                  cleaned_rows = [[str(cell or "").strip().replace("\n", " ") for cell in row] for row in rows]
                  header = "| " + " | ".join(cleaned_rows[0]) + " |"
                  separator = "| " + " | ".join(["---"] * len(cleaned_rows[0])) + " |"
                  body = ["| " + " | ".join(row) + " |" for row in cleaned_rows[1:]]
                  table_blocks.append("\n".join([header, separator] + body))
                except Exception:
                  # If table is too messy, skip Markdown conversion
                  continue
              if table_blocks:
                text = text + "\n\n" + "\n\n".join(table_blocks)
              if text.strip():
                pages.append({"text": text, "page_number": i + 1})
        except Exception:
          doc = fitz.open(stream=file_bytes, filetype="pdf")
          for i, page in enumerate(doc):
            pages.append({"text": page.get_text(), "page_number": i + 1})
    elif filename.endswith(".docx"):
        import io
        doc = DocxDocument(io.BytesIO(file_bytes))
        full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        pages.append({"text": full_text, "page_number": 1})
    elif filename.endswith(".txt"):
        pages.append({"text": file_bytes.decode("utf-8"), "page_number": 1})
    return pages


@app.post("/ingest")
async def ingest(file: UploadFile = File(...), user_id: str = Form(...)):
    file_bytes = await file.read()
    pages = extract_text(file_bytes, file.filename)

    doc_result = supabase.table("documents").insert({
        "user_id": user_id,
        "filename": file.filename,
        "file_type": file.filename.split(".")[-1],
        "page_count": len(pages),
    }).execute()
    doc_id = doc_result.data[0]["id"]

    all_chunks = []
    for page in pages:
        chunks = splitter.split_text(page["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append({"text": chunk, "page_number": page["page_number"], "chunk_index": i})

    chunk_count = 0

    for chunk in all_chunks:
      embedding_vector = embeddings.encode([chunk["text"]])[0]  # Hugging Face or sentence-transformers
      supabase.table("document_chunks").insert({
        "document_id": doc_id,
        "chunk_text": chunk["text"],
        "embedding": embedding_vector,
        "page_number": chunk["page_number"],
        "chunk_index": chunk["chunk_index"],
      }).execute()
      chunk_count += 1

    supabase.table("documents").update({"chunk_count": chunk_count}).eq("id", doc_id).execute()
    return {"document_id": doc_id, "chunk_count": chunk_count}


class QueryRequest(BaseModel):
    question: str
    document_ids: List[str]
    chat_history: Optional[List[dict]] = []
    top_k: int = 5


@app.post("/query")
async def query(req: QueryRequest):
    query_embedding = embeddings.embed_query(req.question)

    result = supabase.rpc("match_documents", {
        "query_embedding": query_embedding,
        "match_count": req.top_k,
        "doc_ids": req.document_ids,
    }).execute()

    chunks = result.data
    context = "\n\n".join([f"[Page {c['page_number']}]: {c['chunk_text']}" for c in chunks])

    prompt = f"""Answer the question using only the context below. Cite page numbers.
If the answer is not in the context, say \"I could not find this in the provided documents.\"

Context:
{context}

Question: {req.question}
Answer:"""


    def stream():
      for token in llm.stream(prompt):
        yield token.content

    sources = [{"chunk_text": c["chunk_text"], "page_number": c["page_number"],
                 "similarity": round(c["similarity"], 3)} for c in chunks]

    return StreamingResponse(stream(), media_type="text/plain",
                             headers={"X-Sources": json.dumps(sources)})


@app.get("/documents/{user_id}")
def list_documents(user_id: str):
    result = supabase.table("documents").select("*").eq("user_id", user_id).execute()
    return result.data
```

This table-aware extraction is the key upgrade that makes the demo feel more senior than a basic text-only RAG app.

---

### Step 4 — Lovable Frontend

Tell Lovable to build:

**Layout:**

- Left sidebar: list of uploaded documents with checkboxes to select which docs to query. At bottom of sidebar: drag-and-drop upload zone with a progress bar
- Right panel: full-height chat interface

**Chat component:**

- User messages on the right, AI responses on the left
- Streaming cursor while response is generating
- Expandable source accordion below each AI response
- Confidence badge on each answer based on source similarity

**Streaming fetch in Lovable:**

```js
const response = await fetch(`${API_URL}/query`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ question, document_ids: selectedDocs, chat_history }),
})

const sources = JSON.parse(response.headers.get("X-Sources") || "[]")
const reader = response.body.getReader()
const decoder = new TextDecoder()
let answer = ""

while (true) {
  const { done, value } = await reader.read()
  if (done) break
  answer += decoder.decode(value)
  setCurrentAnswer(answer)
}
```

---

## Project 3: Synthetic Customer Twin Factory

### Project 3: Copilot Logic Prompt

> Create a FastAPI backend that connects to the US Census API. When a user provides a ZIP code, fetch the demographic data and use Groq (Llama-3-8b) to generate 5 distinct 'Customer Personas' based on that data. Implement a WebSocket endpoint /ws/chat that allows the user to ask one question and receives 5 simultaneous streams—one from each persona—answering the question from their unique perspective.

### Project 3: Lovable UI Prompt

> Build a 'Synthetic Focus Group' UI. Input fields for ZIP Code and Product Category. Below that, 5 cards representing AI personas (use random avatars). When a question is asked, show the text typing into all 5 cards at once. Add a 'Sentiment Summary' chart at the bottom that aggregates the 5 responses into 'Positive', 'Skeptical', or 'Negative'.

### Project 3: What it does

Pulls US Census data for any ZIP code, generates 5 demographically-grounded AI personas, then lets users ask focus group questions and streams all 5 persona responses simultaneously via WebSocket.

---

### Step 1 — Get the Free Census API Key

1. Go to: `https://api.census.gov/data/key_signup.html`
2. Fill in your name and email
3. Key arrives in email within about 60 seconds
4. Rate limit: 500 requests/day on free tier

---

### Step 2 — Project 3: Supabase Schema

```sql
create table sessions (
  id uuid primary key default gen_random_uuid(),
  zip_code text not null,
  product_name text,
  product_category text,
  is_public boolean default false,
  created_at timestamptz default now()
);

create table personas (
  id uuid primary key default gen_random_uuid(),
  session_id uuid references sessions(id) on delete cascade,
  name text,
  age int,
  occupation text,
  city text,
  archetype text,
  system_prompt text,
  census_data jsonb,
  avatar_seed text,
  created_at timestamptz default now()
);

create table responses (
  id uuid primary key default gen_random_uuid(),
  session_id uuid references sessions(id) on delete cascade,
  persona_id uuid references personas(id),
  question text not null,
  answer text not null,
  sentiment text check (sentiment in ('positive', 'neutral', 'skeptical', 'negative')),
  created_at timestamptz default now()
);
```

---

### Step 3 — Free Occupation & Name Lists

For persona generation, use these free sources:

- Random names via the `names` Python package
- US Census or SSA common first names
- BLS occupational data or a hardcoded occupation map by income bracket

---

### Step 4 — FastAPI Backend with WebSocket

```bash
pip install fastapi uvicorn langchain-groq groq supabase python-dotenv httpx names
```

Build the backend with:

- Census API fetch by ZIP code
- persona generation by archetype
- WebSocket endpoint for asking all personas one question
- Groq async calls in parallel (Llama 3 8B for persona chat)
  - Add: Insert a short time.sleep(0.1) between persona calls to avoid Groq rate limits (429 errors) when running "Ask All".
- response storage in Supabase
- session replay endpoint

**Core archetypes:**

- early_adopter
- value_seeker
- brand_loyalist
- skeptic
- community_influencer

---

### Step 5 — Project 3: Lovable Frontend

Tell Lovable to build:

**Page layout:**

- ZIP code input
- product name input
- product category dropdown
- Generate Personas button
- Five persona cards with avatar, role, age, and live answer area
- Bottom question bar with Ask All button

**WebSocket behavior:**

- Send one question to all personas
- Stream all responses into their cards
- Show summary card with buy intent and sentiment breakdown
- Add export to PDF for a client-ready report
- If the WebSocket connection fails, automatically fall back to an HTTP POST request that returns all five responses at once

**Fallback backend idea:**

```python
@app.post("/focus-group/{session_id}")
async def focus_group_http(session_id: str, question: str):
    # Reuse the same persona-response logic as the WebSocket version
    # Return all persona responses in one JSON payload
    ...
    # When calling Groq for each persona:
    import time
    for persona in personas:
      # ...call Groq for persona...
      time.sleep(0.1)  # Prevent hitting Groq rate limits
```

**Fallback frontend idea:**

```js
ws.onerror = async () => {
  const response = await fetch(`${API_URL}/focus-group/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  })
  const data = await response.json()
  // render all responses at once
}
```

**Demo ZIP chips:**

- 10001 - Manhattan NY
- 60601 - Chicago IL
- 30301 - Atlanta GA
- 73101 - Oklahoma City OK
- 94102 - San Francisco CA
- 85001 - Phoenix AZ

---

## Project 4: Medical Billing Revenue Leak Detective

### Project 4: Copilot Logic Prompt

> Develop a medical billing audit engine in FastAPI. Use regex and Llama-3-70b to extract CPT codes from uploaded medical bill text. Compare these codes against a CSV of the 'CMS Medicare Fee Schedule' stored in Supabase. Flag any line item where the 'Billed Amount' is 20% higher than the 'Allowed Amount'. Calculate the 'Total Potential Recovery' for the entire bill.

### Project 4: Lovable UI Prompt

> Build a high-end Financial Audit dashboard for medical billing. 1. A PDF upload area. 2. A data table showing: CPT Code, Description, Billed, CMS Allowed, and Variance %. 3. High-visibility 'Metric Cards' at the top showing 'Total Billed' and 'Recoverable Revenue'. 4. A button to 'Export PDF Audit Report' using jsPDF.

### Project 4: What it does

Accepts a medical bill PDF, extracts CPT procedure codes, compares each against the CMS Medicare Fee Schedule, detects unbundling and upcoding errors, and generates a professional audit report with dollar recovery estimates.

---

### Step 1 — Get the Free Datasets

#### Dataset 1: CMS Medicare Physician Fee Schedule

- URL: `https://www.cms.gov/medicare/payment/fee-schedules/physician`
- Use the national CSV files
- Key columns: `HCPCS`, `DESCRIPTION`, `NON_FACILITY_PRICE`, `GLOB_DAYS`

#### Dataset 2: NCCI PTP Edits

- URL: `https://www.cms.gov/medicare/coding-billing/national-correct-coding-initiative-ncci-edits`
- Use Practitioner PTP Edits
- Build a set of forbidden CPT pairs for bundling checks

#### Dataset 3: Sample Medical Bills

- Generate synthetic bills in Python for demo purposes
- Do not use real patient data

---

### Step 2 — Project 4: Supabase Schema

```sql
create table bills (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id),
  clinic_name text,
  service_date date,
  total_billed numeric(10,2),
  total_allowed numeric(10,2),
  recovery_amount numeric(10,2),
  audit_status text default 'pending',
  uploaded_at timestamptz default now()
);

create table bill_line_items (
  id uuid primary key default gen_random_uuid(),
  bill_id uuid references bills(id) on delete cascade,
  cpt_code text not null,
  description text,
  billed_amount numeric(10,2),
  cms_allowed numeric(10,2),
  variance_pct numeric(5,2),
  flag_type text,
  flag_severity text check (flag_severity in ('high', 'medium', 'low', 'clean')),
  flag_explanation text,
  has_modifier boolean default false
);

create table audit_trail (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users(id),
  bill_id uuid references bills(id),
  action text not null,
  created_at timestamptz default now()
);
```

---

### Step 3 — Project 4: FastAPI Backend

```bash
pip install fastapi uvicorn pdfplumber PyMuPDF pytesseract Pillow pandas supabase python-dotenv
```

Build the backend with:

- PDF parsing via pdfplumber with OCR fallback
- CPT extraction with regex
- CMS price lookup
- NCCI bundling checks
- duplicate and upcoding detection
- bill summary and recovery estimate
- Supabase persistence for bills and line items

---

### Step 4 — Project 4: Lovable Frontend

Tell Lovable to build:

**Layout:**

- drag-and-drop PDF upload zone
- clinic name input
- data table for CPT line items
- sticky summary sidebar

**Data table columns:**

- CPT Code
- Description
- Billed $
- CMS Allowed $
- Variance %
- Status badge

**Summary sidebar:**

- Total Billed
- CMS Allowed
- Potential Recovery
- High / Medium / Clean counts

**Charts:**

- Pie chart for clean vs flagged value
- Bar chart for variance by CPT

**Export:**

- jsPDF audit report download

**Demo bills:**

- demo_clean.json
- demo_moderate.json
- demo_severe.json

---

## Deployment Checklist (All Projects)

## Zero-Cost, Sub-Second AI Architecture

All LLM calls use Groq (Llama 3) for blazing speed and zero cost. Embeddings use Hugging Face or sentence-transformers. This stack is portable, open, and avoids API tax.

### Render (FastAPI backend)

1. Create account at render.com
2. New Web Service → connect GitHub repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables in Render dashboard
6. Add a `/health` endpoint and ping it every 10 minutes with an uptime monitor

### Supabase

1. Create project at supabase.com
2. Run schema SQL in SQL Editor
3. Copy `SUPABASE_URL` and `SUPABASE_SERVICE_KEY`
4. Enable Realtime where needed

### Lovable

1. Create project at lovable.dev
2. Connect Supabase
3. Add your Render backend URL as an environment variable
4. Use `VITE_API_URL` in the frontend
5. Deploy and use the public URL in your portfolio

### Keep Render Free Tier Alive

```python
@app.get("/health")
def health():
    return {"status": "ok"}
```

Then add the `/health` URL to uptimerobot.com with a 10-minute ping interval.
