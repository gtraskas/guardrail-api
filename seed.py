"""
Seed the Supabase benchmarks table with 100 rows from the HaluEval dataset.
Run once: python seed.py
"""

import os

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"],
)

print("Downloading HaluEval dataset (first run may take ~30s)...")
ds = load_dataset("pminervini/HaluEval", "qa_samples")
df = pd.DataFrame(ds["data"])

# HaluEval qa_samples columns: knowledge, question, answer, hallucination ('yes'|'no')
grounded = df[df["hallucination"] == "no"]
hallucinated = df[df["hallucination"] == "yes"]

pass_rows = (
    grounded.sample(50, random_state=42)
    .assign(label="pass")[["knowledge", "answer", "label"]]
    .rename(columns={"knowledge": "source_text", "answer": "ai_response"})
)
fail_rows = (
    hallucinated.sample(50, random_state=99)
    .assign(label="fail")[["knowledge", "answer", "label"]]
    .rename(columns={"knowledge": "source_text", "answer": "ai_response"})
)

benchmarks = pd.concat([pass_rows, fail_rows]).reset_index(drop=True)
benchmarks["model_name"] = "baseline"

records = benchmarks.to_dict(orient="records")

print(f"Inserting {len(records)} benchmark rows into Supabase...")
supabase.table("benchmarks").insert(records).execute()
print(f"Done. Seeded {len(records)} rows.")
