"""
generate_demo_data.py
─────────────────────
Creates a set of realistic demo files that mirror the exact schemas expected
by AnnotationLens, using fully synthetic (but structurally faithful) data.

Run: python generate_demo_data.py
Output directory: demo_data/
"""

import os
import json
import random
import string
import csv

random.seed(42)

OUT_DIR = "demo_data"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Vocabulary for generating realistic-ish responses ─────────────────
TASK_TYPES = ["qa", "data2text", "summarization"]
HALLUCINATION_LABELS = ["Conflicting", "Unverifiable", "Baseless"]

PASSAGES = [
    ("The Eiffel Tower was built in 1889 as the entrance arch for the 1889 World's Fair. "
     "It stands 330 metres tall and is located in Paris, France. "
     "Gustave Eiffel's engineering company designed the structure. "
     "The tower receives approximately 7 million visitors annually. "
     "It was initially criticised by some of France's leading artists."),

    ("Photosynthesis is the process by which plants convert light energy into chemical energy. "
     "Chloroplasts contain chlorophyll, which absorbs sunlight. "
     "The process produces glucose and oxygen as byproducts. "
     "Most plants perform C3 photosynthesis, though C4 and CAM variants also exist. "
     "Without photosynthesis, most life on Earth would not be possible."),

    ("The Python programming language was created by Guido van Rossum and first released in 1991. "
     "Python emphasises code readability and uses significant indentation. "
     "It supports multiple programming paradigms including procedural and object-oriented. "
     "Python is widely used in data science, web development, and automation. "
     "The language name was inspired by Monty Python, not the snake."),

    ("The human genome contains approximately 3 billion base pairs of DNA. "
     "Less than 2 percent of the genome encodes proteins. "
     "The Human Genome Project was completed in 2003 after 13 years. "
     "Gene expression is regulated by complex mechanisms including epigenetics. "
     "Mutations in certain genes are associated with increased cancer risk."),

    ("Black holes are regions of spacetime where gravity is so strong that nothing can escape. "
     "They form when massive stars collapse at the end of their life cycle. "
     "The boundary of a black hole is called the event horizon. "
     "Stephen Hawking proposed that black holes emit thermal radiation. "
     "The first image of a black hole was captured in 2019 by the Event Horizon Telescope."),

    ("The Great Wall of China stretches over 21,000 kilometres across northern China. "
     "Construction began as early as the 7th century BC. "
     "The wall was built to protect Chinese states from nomadic invasions. "
     "Multiple dynasties contributed to its construction over centuries. "
     "Contrary to popular belief, the Great Wall is not visible from space with the naked eye."),

    ("Climate change refers to long-term shifts in global temperatures and weather patterns. "
     "Since the industrial revolution, human activities have been the main driver. "
     "The burning of fossil fuels releases carbon dioxide into the atmosphere. "
     "Rising sea levels, more frequent extreme weather, and biodiversity loss are key effects. "
     "International agreements such as the Paris Accord aim to limit global warming."),

    ("The Internet was developed from ARPANET, a US Defense Department project in the 1960s. "
     "Tim Berners-Lee invented the World Wide Web in 1989. "
     "HTTP and HTML are the foundational protocols of the web. "
     "The Internet connects billions of devices globally via packet-switched networks. "
     "Cloud computing has transformed how data is stored and processed over the Internet."),
]


def random_sentence_from(passage: str) -> tuple[str, int, int]:
    """Return (sentence, start, end) picking a non-trivial chunk from passage."""
    words = passage.split()
    if len(words) < 6:
        return passage, 0, len(passage)
    start_w = random.randint(0, max(0, len(words) - 8))
    end_w   = start_w + random.randint(4, min(12, len(words) - start_w))
    # Reconstruct offsets
    prefix = " ".join(words[:start_w])
    snippet = " ".join(words[start_w:end_w])
    char_start = len(prefix) + (1 if prefix else 0)
    char_end   = char_start + len(snippet)
    return snippet, char_start, min(char_end, len(passage))


def make_hallucinated_response(passage: str) -> tuple[str, list]:
    """
    Take a passage and inject a small amount of plausible hallucinated text.
    Returns (full_response_text, list_of_original_annotations)
    """
    sentences = [s.strip() for s in passage.split(".") if s.strip()]
    idx = random.randint(0, len(sentences) - 1)
    # Build response (passage as-is — the spans refer to offsets in this text)
    response = passage

    # Pick 0–2 random character spans as annotations
    annotations = []
    n_ann = random.randint(0, 2)
    span_set = set()
    for _ in range(n_ann):
        _, s, e = random_sentence_from(passage)
        if s >= e or (s, e) in span_set:
            continue
        span_set.add((s, e))
        annotations.append({
            "start": s, "end": e,
            "label": random.choice(HALLUCINATION_LABELS),
            "reason": "Demo annotation — synthetic span for illustration."
        })
    return response, annotations


# ── Generate 30 demo responses ─────────────────────────────────────────
N = 30
records = []
for i in range(N):
    passage = random.choice(PASSAGES)
    task    = random.choice(TASK_TYPES)
    resp_text, annotations = make_hallucinated_response(passage)
    rid = f"demo_resp_{i+1:04d}"
    records.append({
        "id": rid,
        "task_type": task,
        "response": resp_text,
        "labels": annotations,
    })

# ── response.jsonl ─────────────────────────────────────────────────────
jsonl_path = os.path.join(OUT_DIR, "response.jsonl")
with open(jsonl_path, "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")
print(f"  Wrote {jsonl_path}")

# ── messages.csv ───────────────────────────────────────────────────────
msg_rows = []
for i, r in enumerate(records):
    mid = f"msg_{i+1:04d}"
    msg_rows.append({
        "message_id": mid,
        "source_id": f"src_{i+1:04d}",
        "source_dataset": "demo",
        "source_dataset_id": r["id"],   # matches response.jsonl id
        "original_id": r["id"],
        "role": "assistant",
        "content": r["response"],
    })
    # Also add a preceding user message for completeness
    msg_rows.append({
        "message_id": f"msg_{i+1:04d}_user",
        "source_id": f"src_{i+1:04d}",
        "source_dataset": "demo",
        "source_dataset_id": r["id"],
        "original_id": r["id"],
        "role": "user",
        "content": "Please answer the following question.",
    })

csv_path = os.path.join(OUT_DIR, "messages.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=msg_rows[0].keys())
    w.writeheader(); w.writerows(msg_rows)
print(f"  Wrote {csv_path}")

# ── hallucination_spans.csv (RAGTruth++ — slightly expanded) ───────────
span_rows = []
span_id = 1
# mid -> message_id for assistant rows
mid_map = {r["id"]: f"msg_{i+1:04d}" for i, r in enumerate(records)}

for i, r in enumerate(records):
    mid  = mid_map[r["id"]]
    text = r["response"]
    # Start from original labels and optionally add/modify a span
    spans = list(r["labels"])  # copy

    # With probability 0.4 add an extra span not in original
    if random.random() < 0.4:
        _, s, e = random_sentence_from(text)
        if s < e:
            spans.append({
                "start": s, "end": e,
                "label": random.choice(HALLUCINATION_LABELS),
                "reason": "Additional span identified in re-annotation pass."
            })

    for sp in spans:
        span_rows.append({
            "span_id": f"sp_{span_id:05d}",
            "message_id": mid,
            "start": sp["start"],
            "end":   sp["end"],
            "label": sp["label"],
            "reason": sp.get("reason", ""),
        })
        span_id += 1

csv_path = os.path.join(OUT_DIR, "hallucination_spans.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["span_id","message_id","start","end","label","reason"])
    w.writeheader(); w.writerows(span_rows)
print(f"  Wrote {csv_path}")

# ── lettucedetect_predictions.csv ─────────────────────────────────────
pred_rows = []
for r in records:
    text = r["response"]
    # Generate 0–3 predictions per response (overlapping somewhat with truth)
    n_preds = random.randint(0, 3)
    pred_spans = set()
    for _ in range(n_preds):
        _, s, e = random_sentence_from(text)
        if s >= e or (s, e) in pred_spans:
            continue
        pred_spans.add((s, e))
        pred_rows.append({
            "response_id": r["id"],
            "start": s,
            "end": e,
            "label": "Hallucination",
            "score": round(random.uniform(0.55, 0.99), 4),
        })

csv_path = os.path.join(OUT_DIR, "lettucedetect_predictions.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["response_id","start","end","label","score"])
    w.writeheader(); w.writerows(pred_rows)
print(f"  Wrote {csv_path}")

# ── manual_audit_notes.csv (optional) ─────────────────────────────────
audit_cats = ["Agreement", "Partial Disagreement", "Full Disagreement", "Ambiguous"]
audit_rows = []
for r in random.sample(records, 10):
    audit_rows.append({
        "response_id": r["id"],
        "category": random.choice(audit_cats),
        "finding": "Demo audit finding — spans reviewed manually.",
        "assessment": random.choice(["Confirmed", "Borderline", "Disputed"]),
        "reason": "Synthetic audit note generated for demonstration purposes.",
    })

csv_path = os.path.join(OUT_DIR, "manual_audit_notes.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["response_id","category","finding","assessment","reason"])
    w.writeheader(); w.writerows(audit_rows)
print(f"  Wrote {csv_path}")

print(f"\nDemo data generated in ./{OUT_DIR}/ — {N} responses.")
print("Upload these files in AnnotationLens → Upload Data page.")
