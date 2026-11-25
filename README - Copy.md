# AnnotationLens
### RAG Hallucination Annotation Analysis Platform

A fully-local, interactive web application for exploring and evaluating annotation differences in RAG hallucination detection research.

---

## Research Context

This tool supports the investigation of:

> **"How do differences between original RAGTruth and RAGTruth++ span annotations affect the measured performance of a fixed hallucination detector on matched assistant responses?"**

It compares:
- **Original RAGTruth** hallucination annotations (character-level spans from `response.jsonl`)
- **RAGTruth++** expanded / re-annotated spans (from `hallucination_spans.csv`)
- **LettuceDetect** predictions from `KRLabsOrg/lettucedect-base-modernbert-en-v1` (from `lettucedetect_predictions.csv`)

---

## Features

| Page | Description |
|---|---|
| **Upload** | Validate and load your 4 required project files |
| **Dashboard** | Summary statistics, 4 Chart.js visualisations |
| **Response Explorer** | Per-response highlighted text with toggleable annotation layers |
| **Annotation Comparison** | Exact / partial overlap analysis between RAGTruth and RAGTruth++ |
| **Detector Evaluation** | Character-level Precision / Recall / F1 vs both reference sets |
| **Audit Explorer** | Browse and filter manual audit records (optional) |
| **About** | Research explanation page |

---

## Requirements

- Python 3.9+
- No GPU, no cloud API, no paid services required

---

## Installation

```powershell
# 1. Navigate to the project folder
cd "C:\Users\Dell\Desktop\AnnotationLens"

# 2. (Recommended) Create a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the Application

```powershell
python app.py
```

Then open your browser at: **http://127.0.0.1:5000**

---

## Generating Demo Data (optional)

If you don't have the project files yet, generate synthetic demo data:

```powershell
python generate_demo_data.py
```

This creates a `demo_data/` folder with:
- `response.jsonl`
- `messages.csv`
- `hallucination_spans.csv`
- `lettucedetect_predictions.csv`
- `manual_audit_notes.csv`

Upload these on the Upload page to explore all features with synthetic data.

---

## Required File Schemas

### `messages.csv`
| Column | Description |
|---|---|
| `message_id` | Unique identifier for the message |
| `source_id` | Source record identifier |
| `source_dataset` | Dataset name |
| `source_dataset_id` | ID in the source dataset |
| `role` | `assistant` or `user` |
| `content` | Full text of the message |
| `original_id` *(optional)* | If present, used to match RAGTruth records; otherwise `source_dataset_id` is used |

### `hallucination_spans.csv`
| Column | Description |
|---|---|
| `span_id` | Unique span identifier |
| `message_id` | References `message_id` in messages.csv |
| `start` | Character start offset (0-indexed, inclusive) |
| `end` | Character end offset (exclusive) |
| `label` | Hallucination type label |
| `reason` *(optional)* | Annotation rationale |

### `lettucedetect_predictions.csv`
| Column | Description |
|---|---|
| `response_id` | References `id` in response.jsonl |
| `start` | Character start offset |
| `end` | Character end offset |
| `label` *(optional)* | Predicted label |
| `score` *(optional)* | Confidence score |

### `response.jsonl`
One JSON object per line with:
| Field | Description |
|---|---|
| `id` or `response_id` | Unique response identifier |
| `response` or `content` | Full assistant response text |
| `labels` or `annotations` | List of annotation objects with `start`, `end`, `label`, `reason` |
| `task_type` *(optional)* | Task category |

### Audit CSV *(optional)*
Any of: `manual_audit_notes.csv`, `disagreement_audit_30.csv`, `complete_audit_review.csv`
Required columns: `response_id`, `category`, `finding`, `assessment`

---

## Project Structure

```
AnnotationLens/
├── app.py                     # Flask backend + all data processing
├── generate_demo_data.py      # Synthetic demo data generator
├── requirements.txt
├── README.md
├── templates/
│   ├── base.html              # Layout with sidebar navigation
│   ├── upload.html            # File upload page
│   ├── dashboard.html         # Analytics dashboard
│   ├── explorer.html          # Response Explorer (main feature)
│   ├── comparison.html        # Annotation Comparison
│   ├── evaluation.html        # Detector Evaluation
│   ├── audit.html             # Audit Explorer
│   └── about.html             # Research explanation
├── static/
│   ├── css/main.css           # Dark navy design system
│   └── js/main.js             # Global JS utilities
└── demo_data/                 # Created by generate_demo_data.py
```

---

## Demo Sequence (for Research Reviewer)

1. **Start the app**: `python app.py` → open http://127.0.0.1:5000
2. **Upload page**: Upload your 4 project files (or demo data). Watch validation output.
3. **Dashboard**: Review summary cards and 4 live charts (categories, span totals, metrics, distribution).
4. **Response Explorer**: Select any response ID. Toggle annotation layers on/off. Hover spans for labels.
5. **Annotation Comparison**: Select a response with both original and expanded spans. See exact matches vs partial overlaps vs unique spans.
6. **Detector Evaluation**: Run evaluation in "Both Datasets" mode. Compare Precision / Recall / F1 side-by-side. Expand per-response breakdown.
7. **Audit Explorer** *(if audit file uploaded)*: Filter by category, search findings.
8. **About**: Walk the reviewer through the research context and methodology.

---

## Accuracy & Integrity Notes

- All statistics are computed from uploaded files — nothing is hardcoded.
- Character offsets from source files are validated before display; invalid offsets are skipped with a warning.
- Span text is extracted directly from the response text using the supplied offsets.
- Evaluation metrics use character-level Boolean masks — overlapping spans are handled correctly.
- Missing annotations are not treated as proof that a response is correct.
- The detector performs no live inference — predictions come from the uploaded predictions file.
- Annotation disagreement between RAGTruth and RAGTruth++ does not mean either dataset is wrong.
