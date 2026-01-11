"""
llm_sentiment_openrouter.py

FULL RUN with safe resume + non-arbitrary handling.

Principles:
- Never invent values if the LLM output is incomplete/malformed.
- If a batch fails, retry by splitting into smaller batches (25 -> 12/13 -> ... -> 1).
- Persist results incrementally so you can resume after crashes without losing progress.

Input:
- data_processed/dataset_es_classic.parquet (post_id, created_at, text)

Outputs:
- data_processed/features_llm_all.parquet         (labels only, post_id + features)
- data_processed/features_llm_all_raw.jsonl       (audit log: sent payload + raw response + parsed items OR error)
- data_processed/features_llm_progress.json       (resume state: completed_post_ids)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
from tqdm import tqdm
from openai import OpenAI


# ---------------------------
# Config
# ---------------------------

@dataclass(frozen=True)
class Config:
    model: str = "google/gemini-2.5-flash-lite-preview-09-2025"
    base_url: str = "https://openrouter.ai/api/v1"

    batch_size: int = 25
    temperature: float = 0.0
    max_output_tokens: int = 6000  # per-call cap (NOT total spend)

    out_parquet_name: str = "features_llm_all.parquet"
    out_raw_name: str = "features_llm_all_raw.jsonl"
    progress_name: str = "features_llm_progress.json"

    # Safety: if the model keeps failing, we stop splitting at singletons and skip
    max_split_depth: int = 10


ALLOWED_TOPICS = {
    "none",
    "monetary_policy",
    "fiscal_tax",
    "trade_tariffs",
    "geopolitics",
    "foreign_policy",
    "company_specific",
    "energy_commodities",
    "stock_market",
    "economy_macro",
    "immigration",
    "media",
    "entertainment_celebrity",
    "other",
}

REQUIRED_KEYS = [
    "post_id",
    "market_relevance",
    "market_topic",
    "uncertainty_shock",
    "policy_surprise",
    "novelty",
    "tone_valence",
    "tail_risk_severity",
]


# ---------------------------
# Helpers
# ---------------------------

def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_posts(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path)

    needed = {"post_id", "created_at", "text"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"dataset_es_classic.parquet missing required columns: {sorted(missing)}")

    df = df.copy()
    df["post_id"] = df["post_id"].astype(str)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["text"] = df["text"].astype(str)

    df["__text_stripped"] = df["text"].str.strip()
    df = df.loc[df["__text_stripped"].ne("")].drop(columns="__text_stripped")
    df = df.dropna(subset=["created_at"])
    df = df.sort_values("created_at").reset_index(drop=True)
    return df


def _make_batches(items: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    if not isinstance(text, str):
        raise ValueError("Model response is not a string.")

    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)

    start = t.find("[")
    end = t.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON array found or truncated. Response starts:\n{t[:500]}")

    candidate = t[start : end + 1].strip()

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*([\]}])", r"\1", candidate)
        data = json.loads(cleaned)

    if not isinstance(data, list):
        raise ValueError(f"Extracted JSON is not a list. Got type={type(data)}")
    for obj in data:
        if not isinstance(obj, dict):
            raise ValueError("JSON array must contain objects (dicts).")
    return data


def _clean_keys(obj: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k).strip(): v for k, v in obj.items()}


def _as_int(x: Any) -> int:
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)):
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if re.fullmatch(r"-?\d+", s):
            return int(s)
    raise ValueError(f"Cannot coerce to int: {x!r}")


def _normalize_item_strict(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    STRICT normalization: if anything is missing/malformed, raise.
    No defaults, no arbitrary fills.
    """
    obj = _clean_keys(obj)

    for k in REQUIRED_KEYS:
        if k not in obj:
            raise ValueError(f"Missing key '{k}' in item: {obj}")

    post_id = str(obj["post_id"]).strip()

    mr = _as_int(obj["market_relevance"])
    topic = str(obj["market_topic"]).strip()
    us = _as_int(obj["uncertainty_shock"])
    ps = _as_int(obj["policy_surprise"])
    nov = _as_int(obj["novelty"])
    tv = _as_int(obj["tone_valence"])
    trs = _as_int(obj["tail_risk_severity"])

    # Validate ranges
    if mr not in (0, 1):
        raise ValueError(f"market_relevance must be 0/1, got {mr}")
    if us not in (0, 1):
        raise ValueError(f"uncertainty_shock must be 0/1, got {us}")
    if ps not in (0, 1):
        raise ValueError(f"policy_surprise must be 0/1, got {ps}")
    if not (0 <= nov <= 10):
        raise ValueError(f"novelty must be 0..10, got {nov}")
    if not (0 <= tv <= 10):
        raise ValueError(f"tone_valence must be 0..10, got {tv}")
    if not (0 <= trs <= 3):
        raise ValueError(f"tail_risk_severity must be 0..3, got {trs}")

    # Enforce irrelevant => none/zeros (deterministic rule, not arbitrary)
    if mr == 0:
        topic = "none"
        us = 0
        ps = 0
        trs = 0

    if topic not in ALLOWED_TOPICS:
        topic = "other"

    return {
        "post_id": post_id,
        "market_relevance": mr,
        "market_topic": topic,
        "uncertainty_shock": us,
        "policy_surprise": ps,
        "novelty": nov,
        "tone_valence": tv,
        "tail_risk_severity": trs,
    }


def _build_messages(batch_inputs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    system = (
        "You are a strict JSON generator. Output only valid JSON parseable by json.loads. "
        "No markdown, no backticks, no extra text."
    )
    topic_list = ", ".join(sorted(ALLOWED_TOPICS))

    user = (
        "Return a JSON array with exactly one object per input post, same order.\n"
        "Each object must contain ONLY these keys:\n"
        "- post_id (string)\n"
        "- market_relevance (0 or 1)\n"
        f"- market_topic (one of: {topic_list})\n"
        "- uncertainty_shock (0 or 1)\n"
        "- policy_surprise (0 or 1)\n"
        "- novelty (integer 0..10)\n"
        "- tone_valence (integer 0..10)\n"
        "- tail_risk_severity (integer 0..3)\n\n"
        "Rules:\n"
        "- Always include ALL keys for EVERY object.\n"
        "- Same order as inputs.\n"
        "- If market_relevance=0 then set market_topic='none', uncertainty_shock=0, policy_surprise=0, tail_risk_severity=0.\n"
        "- Output JSON only.\n\n"
        "Input posts:\n"
        + json.dumps(batch_inputs, ensure_ascii=False)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def annotate_batch(client: OpenAI, cfg: Config, batch_inputs: List[Dict[str, str]]) -> Dict[str, Any]:
    messages = _build_messages(batch_inputs)

    resp = client.chat.completions.create(
        model=cfg.model,
        messages=messages,
        temperature=cfg.temperature,
        max_tokens=cfg.max_output_tokens,
    )

    raw_text = (resp.choices[0].message.content or "").strip()
    parsed = _extract_json_array(raw_text)
    items = [_normalize_item_strict(obj) for obj in parsed]

    # Order/length checks (strict)
    if len(items) != len(batch_inputs):
        raise ValueError(f"Model returned {len(items)} items but batch had {len(batch_inputs)} inputs.")

    def norm_id(x: Any) -> str:
        return str(x).strip()

    sent_ids = [norm_id(x["post_id"]) for x in batch_inputs]
    got_ids = [norm_id(x["post_id"]) for x in items]
    if sent_ids != got_ids:
        # pinpoint first mismatch
        for i, (s, g) in enumerate(zip(sent_ids, got_ids)):
            if s != g:
                raise ValueError(f"post_id order mismatch at i={i}: sent={s!r}, got={g!r}")
        raise ValueError("post_id order mismatch (length or late mismatch).")

    return {"raw_text": raw_text, "items": items}


def annotate_with_split_retry(
    client: OpenAI,
    cfg: Config,
    batch_inputs: List[Dict[str, str]],
    out_raw: Path,
    batch_no: int,
    depth: int = 0,
) -> List[Dict[str, Any]]:
    """
    Strict: never fabricate values.
    Retry strategy: if batch fails, split into smaller batches and retry.
    Returns: list of normalized items (same order).
    If depth exceeds max or single item keeps failing, logs and SKIPS that item.
    """
    try:
        result = annotate_batch(client, cfg, batch_inputs)

        with open(out_raw, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "batch": batch_no,
                "sent": batch_inputs,
                "raw_response": result["raw_text"],
                "items": result["items"],
                "split_depth": depth,
            }, ensure_ascii=False) + "\n")

        return result["items"]

    except Exception as e:
        # log failure
        with open(out_raw, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "batch": batch_no,
                "sent": batch_inputs,
                "error_type": type(e).__name__,
                "error": str(e),
                "split_depth": depth,
            }, ensure_ascii=False) + "\n")

        # If singleton and still failing: skip (no arbitrary values)
        if len(batch_inputs) == 1 or depth >= cfg.max_split_depth:
            # skip this item
            return []

        # split and retry
        mid = len(batch_inputs) // 2
        left = batch_inputs[:mid]
        right = batch_inputs[mid:]

        left_items = annotate_with_split_retry(client, cfg, left, out_raw, batch_no, depth + 1)
        right_items = annotate_with_split_retry(client, cfg, right, out_raw, batch_no, depth + 1)

        return left_items + right_items


def _load_progress(progress_path: Path) -> Set[str]:
    if not progress_path.exists():
        return set()
    data = json.loads(progress_path.read_text(encoding="utf-8"))
    done = data.get("completed_post_ids", [])
    return {str(x).strip() for x in done}


def _save_progress(progress_path: Path, completed: Set[str]) -> None:
    progress_path.write_text(
        json.dumps({"completed_post_ids": sorted(completed)}, ensure_ascii=False),
        encoding="utf-8",
    )


def _append_parquet(out_parquet: Path, df_new: pd.DataFrame) -> None:
    """
    Append by reading existing parquet, concatenating, dropping duplicates on post_id.
    (Simple + safe for ~21k rows.)
    """
    if out_parquet.exists():
        df_old = pd.read_parquet(out_parquet, engine="pyarrow")
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["post_id"], keep="last")
    else:
        df_all = df_new

    df_all.to_parquet(out_parquet, index=False, engine="pyarrow")


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    cfg = Config()

    root = _project_root()
    data_path = root / "data_processed" / "dataset_es_classic.parquet"
    out_dir = root / "data_processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_parquet = out_dir / cfg.out_parquet_name
    out_raw = out_dir / cfg.out_raw_name
    progress_path = out_dir / cfg.progress_name

    # You demanded this line not be removed:
    api_key = "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"

    client = OpenAI(api_key=api_key, base_url=cfg.base_url)

    df = _load_posts(data_path)

    # Resume: skip already completed post_ids
    completed = _load_progress(progress_path)

    # Build inputs for items not yet completed
    inputs_all = [{"post_id": str(r["post_id"]), "text": str(r["text"])} for _, r in df.iterrows()]
    inputs = [x for x in inputs_all if str(x["post_id"]).strip() not in completed]

    print(f"[info] Loaded posts: {len(df)}")
    print(f"[info] Already completed: {len(completed)}")
    print(f"[info] Remaining to annotate: {len(inputs)} | batch_size={cfg.batch_size} | model={cfg.model}")

    batches = _make_batches(inputs, cfg.batch_size)

    # Ensure raw log exists (do not wipe on resume)
    out_raw.touch(exist_ok=True)

    # Process
    newly_completed: Set[str] = set()
    buffer: List[Dict[str, Any]] = []

    for b_idx, batch_inputs in enumerate(tqdm(batches, desc="Annotating (full)", total=len(batches)), start=1):
        items = annotate_with_split_retry(client, cfg, batch_inputs, out_raw, batch_no=b_idx)

        # items may be empty if all singletons failed (rare)
        buffer.extend(items)

        # mark completed only for items we actually got valid labels for
        for it in items:
            newly_completed.add(str(it["post_id"]).strip())

        # flush every ~500 items to parquet + progress (tune if you want)
        if len(buffer) >= 500:
            df_new = pd.DataFrame(buffer)

            # enforce types
            df_new["post_id"] = df_new["post_id"].astype(str)
            df_new["market_relevance"] = pd.to_numeric(df_new["market_relevance"], errors="raise").astype("Int64")
            df_new["uncertainty_shock"] = pd.to_numeric(df_new["uncertainty_shock"], errors="raise").astype("Int64")
            df_new["policy_surprise"] = pd.to_numeric(df_new["policy_surprise"], errors="raise").astype("Int64")
            df_new["novelty"] = pd.to_numeric(df_new["novelty"], errors="raise").astype("Int64")
            df_new["tone_valence"] = pd.to_numeric(df_new["tone_valence"], errors="raise").astype("Int64")
            df_new["tail_risk_severity"] = pd.to_numeric(df_new["tail_risk_severity"], errors="raise").astype("Int64")
            df_new["market_topic"] = df_new["market_topic"].astype(str)

            _append_parquet(out_parquet, df_new)

            completed |= newly_completed
            _save_progress(progress_path, completed)

            print(f"[progress] wrote {len(df_new)} rows | total completed now {len(completed)}")

            buffer.clear()
            newly_completed.clear()

    # final flush
    if buffer:
        df_new = pd.DataFrame(buffer)
        df_new["post_id"] = df_new["post_id"].astype(str)
        df_new["market_relevance"] = pd.to_numeric(df_new["market_relevance"], errors="raise").astype("Int64")
        df_new["uncertainty_shock"] = pd.to_numeric(df_new["uncertainty_shock"], errors="raise").astype("Int64")
        df_new["policy_surprise"] = pd.to_numeric(df_new["policy_surprise"], errors="raise").astype("Int64")
        df_new["novelty"] = pd.to_numeric(df_new["novelty"], errors="raise").astype("Int64")
        df_new["tone_valence"] = pd.to_numeric(df_new["tone_valence"], errors="raise").astype("Int64")
        df_new["tail_risk_severity"] = pd.to_numeric(df_new["tail_risk_severity"], errors="raise").astype("Int64")
        df_new["market_topic"] = df_new["market_topic"].astype(str)

        _append_parquet(out_parquet, df_new)

        completed |= {str(x["post_id"]).strip() for x in buffer}
        _save_progress(progress_path, completed)

        print(f"[progress] final write {len(df_new)} rows | total completed {len(completed)}")

    # Summary
    if out_parquet.exists():
        chk = pd.read_parquet(out_parquet, engine="pyarrow")
        print(f"[done] Parquet: {out_parquet}")
        print(f"[done] Rows in parquet: {len(chk)}")
        print(chk.head().to_string(index=False))
    else:
        print("[done] No parquet written (all items failed).")


if __name__ == "__main__":
    main()
