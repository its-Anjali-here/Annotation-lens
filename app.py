"""
AnnotationLens — RAG Hallucination Annotation Analysis Platform
Main Flask application
"""

import os
import json
import re
import io
import traceback
from functools import lru_cache

import numpy as np
import pandas as pd
from flask import (
    Flask, render_template, request, jsonify, session,
    redirect, url_for, flash
)
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "annotationlens-dev-secret-2024")
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB

# ─────────────────────────────────────────────
# In-memory data store (per-process singleton)
# ─────────────────────────────────────────────
DATA = {
    "loaded": False,
    "messages": None,           # pd.DataFrame
    "spans": None,              # pd.DataFrame
    "predictions": None,        # pd.DataFrame
    "ragtruth_responses": None, # dict  id -> record
    "audit": None,              # pd.DataFrame or None
    # processed
    "matched": None,            # pd.DataFrame – final merged view
    "stats": None,              # dict
}

# ─────────────────────────────────────────────
# Schema definitions
# ─────────────────────────────────────────────
REQUIRED_COLS = {
    "messages": ["message_id", "source_id", "source_dataset", "source_dataset_id",
                 "role", "content"],
    "spans": ["span_id", "message_id", "start", "end", "label"],
    "predictions": ["response_id", "start", "end"],
}

AUDIT_COLS = ["response_id", "category", "finding", "assessment"]  # minimum

# ─────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────

def _normalize_ws(text: str) -> str:
    """Collapse all whitespace runs to a single space and strip."""
    return re.sub(r"\s+", " ", text or "").strip()


def _read_csv_safe(file_storage) -> pd.DataFrame:
    raw = file_storage.read()
    file_storage.seek(0)
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise ValueError(f"Cannot parse CSV: {exc}")
    return df


def _check_columns(df: pd.DataFrame, required: list, label: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label}: missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def _validate_offsets(text: str, start: int, end: int) -> bool:
    if not isinstance(start, (int, np.integer)) or not isinstance(end, (int, np.integer)):
        return False
    if start < 0 or end < 0 or start >= end:
        return False
    if end > len(text):
        return False
    return True


def _char_mask(text_len: int, spans: list) -> np.ndarray:
    """Boolean mask of length text_len marking span positions."""
    mask = np.zeros(text_len, dtype=bool)
    for s, e in spans:
        s = max(0, int(s))
        e = min(text_len, int(e))
        if s < e:
            mask[s:e] = True
    return mask


def _mask_metrics(ref_mask: np.ndarray, pred_mask: np.ndarray) -> dict:
    tp = int(np.sum(ref_mask & pred_mask))
    fp = int(np.sum(~ref_mask & pred_mask))
    fn = int(np.sum(ref_mask & ~pred_mask))
    tn = int(np.sum(~ref_mask & ~pred_mask))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}


# ─────────────────────────────────────────────
# Data loading pipeline
# ─────────────────────────────────────────────

def _load_and_merge(messages_file, spans_file, preds_file, ragtruth_file,
                    audit_file=None):
    errors = []

    # ── 1. messages.csv ────────────────────────────────────────────────
    messages = _read_csv_safe(messages_file)
    _check_columns(messages, REQUIRED_COLS["messages"], "messages.csv")
    # keep only assistant messages from the expanded dataset
    asst = messages[messages["role"] == "assistant"].copy()
    asst = asst.dropna(subset=["content"])
    asst["content_norm"] = asst["content"].apply(_normalize_ws)
    # Derive original_id: prefer a column called original_id or source_dataset_id
    if "original_id" in asst.columns:
        asst["original_id"] = asst["original_id"].astype(str).str.strip()
    else:
        asst["original_id"] = asst["source_dataset_id"].astype(str).str.strip()

    # ── 2. hallucination_spans.csv ─────────────────────────────────────
    spans = _read_csv_safe(spans_file)
    _check_columns(spans, REQUIRED_COLS["spans"], "hallucination_spans.csv")
    spans["start"] = pd.to_numeric(spans["start"], errors="coerce")
    spans["end"]   = pd.to_numeric(spans["end"],   errors="coerce")
    spans = spans.dropna(subset=["start", "end", "message_id"])
    spans["start"] = spans["start"].astype(int)
    spans["end"]   = spans["end"].astype(int)
    spans["message_id"] = spans["message_id"].astype(str)

    # ── 3. lettucedetect_predictions.csv ──────────────────────────────
    preds = _read_csv_safe(preds_file)
    _check_columns(preds, REQUIRED_COLS["predictions"], "lettucedetect_predictions.csv")
    preds["start"] = pd.to_numeric(preds["start"], errors="coerce")
    preds["end"]   = pd.to_numeric(preds["end"],   errors="coerce")
    preds = preds.dropna(subset=["start", "end", "response_id"])
    preds["start"] = preds["start"].astype(int)
    preds["end"]   = preds["end"].astype(int)
    preds["response_id"] = preds["response_id"].astype(str).str.strip()

    # ── 4. response.jsonl ─────────────────────────────────────────────
    ragtruth_raw = ragtruth_file.read().decode("utf-8", errors="replace")
    ragtruth_records = {}
    orig_spans_by_id = {}  # original_id -> list of {start,end,label,...}
    for line_no, line in enumerate(ragtruth_raw.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"response.jsonl line {line_no}: JSON error – {exc}")
            continue
        rid = str(rec.get("id", rec.get("response_id", ""))).strip()
        if not rid:
            errors.append(f"response.jsonl line {line_no}: missing id field")
            continue
        ragtruth_records[rid] = rec
        # collect original annotations
        ann_list = []
        for ann in rec.get("labels", rec.get("annotations", [])):
            s = ann.get("start")
            e = ann.get("end")
            if s is None or e is None:
                continue
            ann_list.append({
                "start": int(s), "end": int(e),
                "label": ann.get("label", ann.get("hallucination_type", "Hallucination")),
                "reason": ann.get("reason", ann.get("explanation", ""))
            })
        orig_spans_by_id[rid] = ann_list

    if not ragtruth_records:
        raise ValueError("response.jsonl contained no parseable records.")

    # ── 5. Match expanded messages to RAGTruth records ─────────────────
    matched_rows = []
    unmatched_ids = []

    for _, row in asst.iterrows():
        oid = row["original_id"]
        rt_rec = ragtruth_records.get(oid)
        if rt_rec is None:
            unmatched_ids.append(oid)
            continue
        rt_text = _normalize_ws(rt_rec.get("response", rt_rec.get("content", "")))
        exp_text = row["content_norm"]
        if rt_text != exp_text and not (rt_text in exp_text or exp_text in rt_text):
            # allow slight mismatch – warn but keep
            errors.append(
                f"Text mismatch for original_id={oid}: "
                f"RAGTruth length={len(rt_text)}, messages.csv length={len(exp_text)}"
            )
        matched_rows.append({
            "message_id": str(row["message_id"]),
            "original_id": oid,
            "content": row["content"],
            "task_type": rt_rec.get("task_type", rt_rec.get("source_type", "")),
            "rt_text": rt_rec.get("response", rt_rec.get("content", row["content"])),
        })

    if not matched_rows:
        raise ValueError(
            f"No expanded messages matched any RAGTruth record. "
            f"Checked {len(asst)} assistant messages against {len(ragtruth_records)} "
            f"RAGTruth records. First few unmatched original_ids: {unmatched_ids[:5]}"
        )

    matched = pd.DataFrame(matched_rows)

    # ── 6. Attach expanded spans (RAGTruth++) ─────────────────────────
    matched_mids = set(matched["message_id"].astype(str))
    exp_spans_by_mid = {}
    bad_offset_count = 0
    for _, sp in spans.iterrows():
        mid = str(sp["message_id"])
        if mid not in matched_mids:
            continue
        row_m = matched[matched["message_id"] == mid].iloc[0]
        txt = row_m["content"]
        if not _validate_offsets(txt, sp["start"], sp["end"]):
            bad_offset_count += 1
            continue
        _span_label = str(sp["label"]) if "label" in sp.index else "Hallucination"
        _span_reason = ""
        if "reason" in sp.index and sp["reason"] == sp["reason"]:  # not NaN
            _span_reason = str(sp["reason"])
        elif "explanation" in sp.index and sp["explanation"] == sp["explanation"]:
            _span_reason = str(sp["explanation"])
        exp_spans_by_mid.setdefault(mid, []).append({
            "start": int(sp["start"]), "end": int(sp["end"]),
            "label": _span_label,
            "reason": _span_reason,
        })
    if bad_offset_count:
        errors.append(
            f"Skipped {bad_offset_count} expanded spans with invalid offsets."
        )

    # ── 7. Attach original RAGTruth spans ──────────────────────────────
    orig_spans_validated = {}
    orig_bad = 0
    for oid, ann_list in orig_spans_by_id.items():
        rt_rec = ragtruth_records.get(oid, {})
        txt = rt_rec.get("response", rt_rec.get("content", ""))
        valid = []
        for ann in ann_list:
            if _validate_offsets(txt, ann["start"], ann["end"]):
                valid.append(ann)
            else:
                orig_bad += 1
        orig_spans_validated[oid] = valid
    if orig_bad:
        errors.append(
            f"Skipped {orig_bad} original RAGTruth spans with invalid offsets."
        )

    # ── 8. Attach detector predictions ────────────────────────────────
    pred_cols = set(preds.columns)
    pred_by_oid = {}
    pred_bad = 0
    for _, pr in preds.iterrows():
        oid = str(pr["response_id"])
        if oid not in ragtruth_records:
            continue
        rt_rec = ragtruth_records[oid]
        txt = rt_rec.get("response", rt_rec.get("content", ""))
        if not _validate_offsets(txt, int(pr["start"]), int(pr["end"])):
            pred_bad += 1
            continue
        if "label" in pred_cols:
            label = str(pr["label"]) if pr["label"] == pr["label"] else "Hallucination"
        elif "predicted_label" in pred_cols:
            label = str(pr["predicted_label"]) if pr["predicted_label"] == pr["predicted_label"] else "Hallucination"
        else:
            label = "Hallucination"
        score = float(pr["score"]) if "score" in pred_cols and pr["score"] == pr["score"] else None
        pred_by_oid.setdefault(oid, []).append({
            "start": int(pr["start"]), "end": int(pr["end"]),
            "label": label,
            "score": score,
        })
    if pred_bad:
        errors.append(
            f"Skipped {pred_bad} detector predictions with invalid offsets."
        )

    # ── 9. Audit (optional) ────────────────────────────────────────────
    audit_df = None
    if audit_file is not None:
        try:
            audit_df = _read_csv_safe(audit_file)
            missing_a = [c for c in AUDIT_COLS if c not in audit_df.columns]
            if missing_a:
                errors.append(
                    f"Audit file missing columns {missing_a}; audit section disabled."
                )
                audit_df = None
        except Exception as exc:
            errors.append(f"Could not parse audit file: {exc}")

    # ── 10. Compute summary statistics ────────────────────────────────
    n_matched = len(matched)
    orig_span_total = sum(
        len(orig_spans_validated.get(r["original_id"], []))
        for _, r in matched.iterrows()
    )
    exp_span_total = sum(
        len(exp_spans_by_mid.get(r["message_id"], []))
        for _, r in matched.iterrows()
    )
    pred_span_total = sum(
        len(pred_by_oid.get(r["original_id"], []))
        for _, r in matched.iterrows()
    )

    both = 0; only_exp = 0; only_orig = 0; none_ = 0
    for _, r in matched.iterrows():
        has_orig = len(orig_spans_validated.get(r["original_id"], [])) > 0
        has_exp  = len(exp_spans_by_mid.get(r["message_id"], [])) > 0
        if has_orig and has_exp:   both += 1
        elif has_exp:              only_exp += 1
        elif has_orig:             only_orig += 1
        else:                      none_ += 1

    stats = {
        "n_matched": n_matched,
        "n_orig_spans": orig_span_total,
        "n_exp_spans": exp_span_total,
        "n_pred_spans": pred_span_total,
        "both": both,
        "only_exp": only_exp,
        "only_orig": only_orig,
        "none": none_,
    }

    return {
        "loaded": True,
        "messages": messages,
        "spans": spans,
        "predictions": preds,
        "ragtruth_responses": ragtruth_records,
        "audit": audit_df,
        "matched": matched,
        "orig_spans": orig_spans_validated,      # oid -> list
        "exp_spans": exp_spans_by_mid,           # mid -> list
        "pred_spans": pred_by_oid,               # oid -> list
        "stats": stats,
        "warnings": errors,
    }


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/")
def index():
    if DATA["loaded"]:
        return redirect(url_for("dashboard"))
    return redirect(url_for("upload"))


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "GET":
        return render_template("upload.html", loaded=DATA["loaded"])

    # POST — handle file upload
    required_keys = ["messages_csv", "spans_csv", "preds_csv", "ragtruth_jsonl"]
    for key in required_keys:
        if key not in request.files or request.files[key].filename == "":
            return jsonify({"success": False,
                            "error": f"Missing required file: {key}"}), 400

    audit_file = request.files.get("audit_csv")
    if audit_file and audit_file.filename == "":
        audit_file = None

    try:
        result = _load_and_merge(
            messages_file=request.files["messages_csv"],
            spans_file=request.files["spans_csv"],
            preds_file=request.files["preds_csv"],
            ragtruth_file=request.files["ragtruth_jsonl"],
            audit_file=audit_file,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        app.logger.error(tb)
        return jsonify({"success": False, "error": str(exc)}), 400

    # Persist in global DATA
    DATA.update(result)

    warnings = result.get("warnings", [])
    return jsonify({
        "success": True,
        "stats": result["stats"],
        "warnings": warnings,
        "redirect": url_for("dashboard"),
    })


@app.route("/dashboard")
def dashboard():
    if not DATA["loaded"]:
        return redirect(url_for("upload"))
    return render_template("dashboard.html", stats=DATA["stats"])


@app.route("/api/dashboard_stats")
def api_dashboard_stats():
    if not DATA["loaded"]:
        return jsonify({"error": "No data loaded"}), 400
    return jsonify(DATA["stats"])


@app.route("/explorer")
def explorer():
    if not DATA["loaded"]:
        return redirect(url_for("upload"))
    matched = DATA["matched"]
    response_ids = sorted(matched["original_id"].unique().tolist())
    return render_template("explorer.html", response_ids=response_ids)


@app.route("/api/response/<response_id>")
def api_response(response_id):
    if not DATA["loaded"]:
        return jsonify({"error": "No data loaded"}), 400

    matched = DATA["matched"]
    row = matched[matched["original_id"] == response_id]
    if row.empty:
        return jsonify({"error": f"Response ID '{response_id}' not found"}), 404
    row = row.iloc[0]

    mid = str(row["message_id"])
    oid = str(row["original_id"])
    rt_rec = DATA["ragtruth_responses"].get(oid, {})
    response_text = rt_rec.get("response", rt_rec.get("content", row["content"]))

    orig_spans = DATA["orig_spans"].get(oid, [])
    exp_spans  = DATA["exp_spans"].get(mid, [])
    pred_spans = DATA["pred_spans"].get(oid, [])

    return jsonify({
        "original_id": oid,
        "message_id": mid,
        "task_type": str(row.get("task_type", "")),
        "response_text": response_text,
        "orig_spans": orig_spans,
        "exp_spans": exp_spans,
        "pred_spans": pred_spans,
    })


@app.route("/comparison")
def comparison():
    if not DATA["loaded"]:
        return redirect(url_for("upload"))
    matched = DATA["matched"]
    response_ids = sorted(matched["original_id"].unique().tolist())
    return render_template("comparison.html", response_ids=response_ids)


@app.route("/api/comparison/<response_id>")
def api_comparison(response_id):
    if not DATA["loaded"]:
        return jsonify({"error": "No data loaded"}), 400

    matched = DATA["matched"]
    row = matched[matched["original_id"] == response_id]
    if row.empty:
        return jsonify({"error": f"Response ID '{response_id}' not found"}), 404
    row = row.iloc[0]

    mid = str(row["message_id"])
    oid = str(row["original_id"])
    rt_rec = DATA["ragtruth_responses"].get(oid, {})
    response_text = rt_rec.get("response", rt_rec.get("content", row["content"]))
    text_len = len(response_text)

    orig_spans = DATA["orig_spans"].get(oid, [])
    exp_spans  = DATA["exp_spans"].get(mid, [])
    pred_spans = DATA["pred_spans"].get(oid, [])

    # Pairwise overlap analysis
    def _spans_overlap(a, b):
        return a["start"] < b["end"] and b["start"] < a["end"]

    exact_matches = []
    partial_overlaps = []
    orig_only = []
    exp_only = []

    matched_exp_indices = set()
    for os_ in orig_spans:
        found_exact = False
        found_partial = False
        for ei, es in enumerate(exp_spans):
            if os_["start"] == es["start"] and os_["end"] == es["end"]:
                exact_matches.append({"orig": os_, "exp": es})
                matched_exp_indices.add(ei)
                found_exact = True
                break
            elif _spans_overlap(os_, es):
                partial_overlaps.append({"orig": os_, "exp": es})
                matched_exp_indices.add(ei)
                found_partial = True
        if not found_exact and not found_partial:
            orig_only.append(os_)

    for ei, es in enumerate(exp_spans):
        if ei not in matched_exp_indices:
            exp_only.append(es)

    return jsonify({
        "original_id": oid,
        "response_text": response_text,
        "orig_spans": orig_spans,
        "exp_spans": exp_spans,
        "pred_spans": pred_spans,
        "comparison": {
            "exact_matches": exact_matches,
            "partial_overlaps": partial_overlaps,
            "orig_only": orig_only,
            "exp_only": exp_only,
        }
    })


@app.route("/evaluation")
def evaluation():
    if not DATA["loaded"]:
        return redirect(url_for("upload"))
    matched = DATA["matched"]
    response_ids = sorted(matched["original_id"].unique().tolist())
    return render_template("evaluation.html", response_ids=response_ids)


@app.route("/api/evaluation", methods=["POST"])
def api_evaluation():
    """
    Body: { "response_ids": [...] or "all", "mode": "orig" | "exp" | "both" }
    """
    if not DATA["loaded"]:
        return jsonify({"error": "No data loaded"}), 400

    body = request.get_json(force=True, silent=True) or {}
    selected = body.get("response_ids", "all")
    mode = body.get("mode", "both")

    matched = DATA["matched"]
    if selected == "all":
        rows = matched
    else:
        rows = matched[matched["original_id"].isin(selected)]

    if rows.empty:
        return jsonify({"error": "No matched responses for the selected IDs"}), 400

    def _compute(use_orig: bool):
        agg_tp = agg_fp = agg_fn = agg_tn = 0
        per_response = []
        for _, r in rows.iterrows():
            oid = str(r["original_id"])
            mid = str(r["message_id"])
            rt_rec = DATA["ragtruth_responses"].get(oid, {})
            text = rt_rec.get("response", rt_rec.get("content", r["content"]))
            tlen = len(text)
            if tlen == 0:
                continue
            ref_spans = (DATA["orig_spans"].get(oid, []) if use_orig
                         else DATA["exp_spans"].get(mid, []))
            pred_spans = DATA["pred_spans"].get(oid, [])
            ref_mask  = _char_mask(tlen, [(s["start"], s["end"]) for s in ref_spans])
            pred_mask = _char_mask(tlen, [(s["start"], s["end"]) for s in pred_spans])
            m = _mask_metrics(ref_mask, pred_mask)
            agg_tp += m["tp"]; agg_fp += m["fp"]
            agg_fn += m["fn"]; agg_tn += m["tn"]
            per_response.append({"response_id": oid, **m})

        prec = agg_tp / (agg_tp + agg_fp) if (agg_tp + agg_fp) > 0 else 0.0
        rec  = agg_tp / (agg_tp + agg_fn) if (agg_tp + agg_fn) > 0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        return {
            "tp": agg_tp, "fp": agg_fp, "fn": agg_fn, "tn": agg_tn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "n_responses": len(per_response),
            "per_response": per_response,
        }

    result = {}
    if mode in ("orig", "both"):
        result["orig"] = _compute(use_orig=True)
    if mode in ("exp", "both"):
        result["exp"] = _compute(use_orig=False)

    return jsonify(result)


@app.route("/audit")
def audit():
    if not DATA["loaded"]:
        return redirect(url_for("upload"))
    has_audit = DATA.get("audit") is not None
    return render_template("audit.html", has_audit=has_audit)


@app.route("/api/audit")
def api_audit():
    if not DATA["loaded"]:
        return jsonify({"error": "No data loaded"}), 400
    audit_df = DATA.get("audit")
    if audit_df is None:
        return jsonify({"error": "No audit data loaded"}), 404
    category = request.args.get("category", "")
    df = audit_df.copy()
    if category:
        df = df[df["category"].astype(str) == category]
    records = df.fillna("").to_dict(orient="records")
    categories = sorted(audit_df["category"].dropna().unique().tolist())
    return jsonify({"records": records, "categories": categories})


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/api/response_ids")
def api_response_ids():
    if not DATA["loaded"]:
        return jsonify({"error": "No data loaded"}), 400
    matched = DATA["matched"]
    return jsonify({"response_ids": sorted(matched["original_id"].unique().tolist())})


@app.route("/api/span_distribution")
def api_span_distribution():
    """Distribution of span counts per response for chart."""
    if not DATA["loaded"]:
        return jsonify({"error": "No data loaded"}), 400
    matched = DATA["matched"]
    buckets_orig = {}
    buckets_exp = {}
    for _, r in matched.iterrows():
        oid = str(r["original_id"])
        mid = str(r["message_id"])
        n_orig = len(DATA["orig_spans"].get(oid, []))
        n_exp  = len(DATA["exp_spans"].get(mid, []))
        buckets_orig[n_orig] = buckets_orig.get(n_orig, 0) + 1
        buckets_exp[n_exp]   = buckets_exp.get(n_exp, 0) + 1

    max_k = max(max(buckets_orig.keys(), default=0),
                max(buckets_exp.keys(), default=0))
    labels = list(range(0, max_k + 1))
    orig_counts = [buckets_orig.get(k, 0) for k in labels]
    exp_counts  = [buckets_exp.get(k,  0) for k in labels]
    return jsonify({"labels": labels,
                    "orig_counts": orig_counts,
                    "exp_counts": exp_counts})


@app.route("/api/status")
def api_status():
    return jsonify({"loaded": DATA["loaded"],
                    "stats": DATA.get("stats")})


if __name__ == "__main__":
    print("=" * 60)
    print("  AnnotationLens — RAG Hallucination Analysis Platform")
    print("  Running at: http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, host="127.0.0.1", port=5000)
