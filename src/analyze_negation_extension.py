#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_negation_extension.py

Reads SigExt run outputs (test_dataset.jsonl + test_predictions.json)
and computes:

1) Negation exposure (how often selected keywords are negated in the source)
2) Faithfulness metric: negation flip rate
3) Optional: prints examples of flips for qualitative analysis

Works for BOTH:
- baseline runs (sigext_topk)
- neg-aware runs (sigext_topk_neg) with styles: NOT(...), [NEG], (negated)

Usage:
python analyze_negation_extension.py \
  --run_dir /content/drive/MyDrive/experiments/cnn_sigext_mistral_k15 \
  --max_examples 500 \
  --print_top_flips 10
"""

import argparse
import json
import math
import os
import pathlib
import re
from collections import Counter, defaultdict

import jsonlines
from rapidfuzz import fuzz


# ----------------------------
# Negation cues (same spirit as your generation script)
# ----------------------------
NEG_SINGLE = {
    "no", "not", "never", "none", "without",
    "cannot", "can't", "won't", "isn't", "aren't", "wasn't", "weren't",
}
NEG_MULTI = [
    "do not", "does not", "did not",
    "has not", "have not", "had not",
    "will not", "would not", "should not", "could not", "must not",
    "not expected to", "not likely to", "no plans to",
    "deny", "denies", "denied",
    "fail to", "fails to", "failed to",
    "lack of", "rule out", "ruled out", "no evidence of",
]
FALSE_POS = ["not only"]


def find_sentence_bounds(text: str, idx: int):
    if idx is None or idx < 0:
        return 0, len(text)
    left = text.rfind("\n", 0, idx)
    for ch in [".", "!", "?", ";", ":"]:
        left = max(left, text.rfind(ch, 0, idx))

    right_candidates = [text.find("\n", idx)]
    for ch in [".", "!", "?", ";", ":"]:
        right_candidates.append(text.find(ch, idx))
    right_candidates = [c for c in right_candidates if c != -1]
    right = min(right_candidates) if right_candidates else len(text)
    if right < len(text):
        right = min(len(text), right + 1)

    left = 0 if left == -1 else left + 1
    return left, right


def is_negated_in_context(text: str, phrase_start: int, phrase: str) -> bool:
    if not text or phrase_start is None or phrase_start < 0:
        return False
    phrase = (phrase or "").strip()
    if not phrase:
        return False

    left, right = find_sentence_bounds(text, phrase_start)
    sent = text[left:right].lower()

    local_start = max(0, phrase_start - left)
    local_end = min(len(sent), local_start + len(phrase))

    pre = sent[max(0, local_start - 140): local_start]
    post = sent[local_end: min(len(sent), local_end + 100)]
    around = sent[max(0, local_start - 50): min(len(sent), local_end + 50)]

    fp_hit = any(fp in around for fp in FALSE_POS)

    for cue in NEG_MULTI:
        if cue in pre or cue in around or cue in post:
            if fp_hit and cue == "not":
                continue
            return True

    # single cues closer
    for cue in NEG_SINGLE:
        if cue in pre or cue in around:
            if fp_hit and cue == "not":
                continue
            return True

    return False


# ----------------------------
# Keyword extraction from prompt_input
# ----------------------------
def extract_keywords_from_prompt(prompt: str):
    """
    Extract the <keywords> list that was injected into the prompt.
    We rely on the shared substring "Consider include the following information:" from prompts.py
    Then split on ';'.
    """
    if not isinstance(prompt, str):
        return []

    marker = "Consider include the following information:"
    if marker not in prompt:
        return []

    tail = prompt.split(marker, 1)[1].strip()

    # remove closing tokens like [/INST] or trailing punctuation
    tail = tail.replace("[/INST]", " ").strip()

    # keywords usually end with a period
    # but sometimes there's no period. We'll just strip trailing whitespace/punct.
    tail = tail.strip()
    tail = tail.rstrip()

    # If there's extra text after keywords, try to cut at the end of the keyword list.
    # Commonly it ends at the first newline.
    tail = tail.split("\n")[0].strip()

    # Drop trailing '.' if present
    if tail.endswith("."):
        tail = tail[:-1].strip()

    # Split on ';'
    parts = [p.strip() for p in tail.split(";")]
    parts = [p for p in parts if p]
    return parts


def normalize_keyword(kw: str):
    kw = (kw or "").strip()

    # Handle styles:
    # NOT(phrase)
    m = re.match(r"^NOT\((.*)\)$", kw.strip(), flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(), True

    # [NEG] phrase
    m = re.match(r"^\[NEG\]\s*(.*)$", kw.strip(), flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(), True

    # phrase (negated)
    m = re.match(r"^(.*)\s*\(negated\)$", kw.strip(), flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(), True

    return kw, False


# ----------------------------
# Matching keyword in text / summary
# ----------------------------
def find_phrase_start(text: str, phrase: str):
    """Return char index of first occurrence, or -1."""
    if not text or not phrase:
        return -1
    return text.lower().find(phrase.lower())


def phrase_in_summary(summary: str, phrase: str, min_fuzzy=90):
    """
    Check if phrase appears in summary by:
    - substring
    - fallback fuzzy match on windows
    Returns (matched: bool, match_start_char: int or -1)
    """
    if not summary or not phrase:
        return False, -1

    s = summary.lower()
    p = phrase.lower()

    idx = s.find(p)
    if idx != -1:
        return True, idx

    # Fuzzy fallback (cheap): scan a few windows
    # Limit work: if phrase is long, windows are sized around it.
    L = len(p)
    if L < 6:
        return False, -1

    # sample windows every ~20 chars
    step = 20
    best = 0
    best_i = -1
    for i in range(0, max(1, len(s) - L + 1), step):
        chunk = s[i:i+L]
        sc = fuzz.ratio(chunk, p)
        if sc > best:
            best = sc
            best_i = i
    if best >= min_fuzzy:
        return True, best_i
    return False, -1


def has_local_negation(summary: str, match_start: int, phrase: str, window_tokens=5):
    """
    Check if there is a negation cue near the matched phrase location in the summary.
    We approximate using token windows around the character match.
    """
    if not summary or match_start < 0:
        return False

    # crude tokenization
    tokens = re.findall(r"\w+|[^\w\s]", summary.lower(), flags=re.UNICODE)
    if not tokens:
        return False

    # map char position -> token index (approx)
    # We'll do a simple rebuild of token spans.
    spans = []
    pos = 0
    s = summary
    for t in tokens:
        # find next occurrence from pos
        j = s.lower().find(t.lower(), pos)
        if j == -1:
            continue
        spans.append((j, j + len(t), t))
        pos = j + len(t)

    # find token index closest to match_start
    best_k = None
    best_dist = 10**9
    for k, (a, b, t) in enumerate(spans):
        dist = abs(a - match_start)
        if dist < best_dist:
            best_dist = dist
            best_k = k
    if best_k is None:
        return False

    lo = max(0, best_k - window_tokens)
    hi = min(len(spans), best_k + window_tokens + 1)
    local = " ".join([spans[k][2] for k in range(lo, hi)])

    # handle false positive "not only"
    if "not only" in local:
        # still could be negation elsewhere, so don't early-return True
        pass

    # multi cues
    for cue in NEG_MULTI:
        if cue in local:
            return True
    # single
    for cue in NEG_SINGLE:
        if cue in local:
            if cue == "not" and "not only" in local:
                continue
            return True

    return False


# ----------------------------
# Main analysis
# ----------------------------
def load_run(run_dir: str):
    run_dir = pathlib.Path(run_dir)
    ds_path = run_dir / "test_dataset.jsonl"
    pr_path = run_dir / "test_predictions.json"

    if not ds_path.exists():
        raise FileNotFoundError(f"Missing {ds_path}")
    if not pr_path.exists():
        raise FileNotFoundError(f"Missing {pr_path}")

    with jsonlines.open(str(ds_path)) as f:
        data = list(f)

    with open(str(pr_path), "r") as f:
        preds = json.load(f)

    if len(preds) != len(data):
        # still continue but warn by truncation
        n = min(len(preds), len(data))
        data = data[:n]
        preds = preds[:n]

    return data, preds


def analyze_run(run_dir: str, max_examples=None, print_top_flips=0):
    data, preds = load_run(run_dir)
    if max_examples is not None:
        data = data[:max_examples]
        preds = preds[:max_examples]

    K_counts = []
    neg_counts = []
    exposure = 0

    flip_examples = 0
    flip_total = 0
    neg_total = 0

    flip_records = []  # for qualitative display

    for ex, summary in zip(data, preds):
        text = ex.get("trunc_input", "")
        prompt = ex.get("prompt_input", "")

        kws_raw = extract_keywords_from_prompt(prompt)
        kws = []
        kw_marked_neg = 0
        for k in kws_raw:
            base, marked = normalize_keyword(k)
            if marked:
                kw_marked_neg += 1
            if base:
                kws.append(base)

        K = len(kws)
        if K == 0:
            continue

        K_counts.append(K)

        ex_neg_count = 0
        ex_flip = 0

        for phrase in kws:
            # locate phrase in source
            start = find_phrase_start(text, phrase)
            if start == -1:
                # if we can't find it, skip for negation exposure (conservative)
                continue

            neg = is_negated_in_context(text, start, phrase)
            if neg:
                ex_neg_count += 1
                neg_total += 1

                # check if summary mentions it
                present, match_start = phrase_in_summary(summary, phrase)
                if present:
                    # check local negation in summary
                    local_neg = has_local_negation(summary, match_start, phrase)
                    if not local_neg:
                        ex_flip += 1
                        flip_total += 1
                        if print_top_flips > 0:
                            flip_records.append({
                                "phrase": phrase,
                                "source_snippet": text[max(0, start-120): min(len(text), start+len(phrase)+120)],
                                "summary": summary,
                            })

        neg_counts.append(ex_neg_count)
        if ex_neg_count > 0:
            exposure += 1
        if ex_flip > 0:
            flip_examples += 1

    n = len(neg_counts) if neg_counts else 1

    # exposure distribution buckets
    buckets = Counter()
    for c in neg_counts:
        if c == 0:
            buckets["0"] += 1
        elif c == 1:
            buckets["1"] += 1
        elif 2 <= c <= 3:
            buckets["2-3"] += 1
        else:
            buckets["4+"] += 1

    results = {
        "run_dir": str(run_dir),
        "n_examples": n,
        "avg_K": sum(K_counts)/len(K_counts) if K_counts else 0.0,
        "exposure_rate": exposure / n,
        "avg_negated_kw": sum(neg_counts)/n,
        "dist_negated_kw": dict(buckets),
        "flip_example_rate": flip_examples / n,
        "flips_per_100": (flip_total / n) * 100.0,
        "flips_per_neg_kw": (flip_total / neg_total) if neg_total > 0 else 0.0,
        "neg_total": int(neg_total),
        "flip_total": int(flip_total),
    }

    # Print top flips for qualitative section
    if print_top_flips > 0 and flip_records:
        print("\n=== TOP FLIP EXAMPLES (for qualitative analysis) ===")
        for rec in flip_records[:print_top_flips]:
            print("\n---")
            print("PHRASE:", rec["phrase"])
            print("SOURCE SNIPPET:", rec["source_snippet"].replace("\n", " "))
            print("SUMMARY:", rec["summary"].replace("\n", " "))

    return results


def fmt_pct(x): 
    return f"{100.0*x:.2f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, type=str, help="One run folder containing test_dataset.jsonl + test_predictions.json")
    ap.add_argument("--max_examples", default=None, type=int)
    ap.add_argument("--print_top_flips", default=0, type=int)
    args = ap.parse_args()

    res = analyze_run(args.run_dir, max_examples=args.max_examples, print_top_flips=args.print_top_flips)

    print("\n=== Negation Extension Analysis ===")
    print("Run:", res["run_dir"])
    print("N:", res["n_examples"])
    print("avg K:", f"{res['avg_K']:.2f}")
    print("Exposure rate (>=1 negated kw in source):", fmt_pct(res["exposure_rate"]))
    print("Avg negated keywords:", f"{res['avg_negated_kw']:.3f}")
    print("Negated kw distribution:", res["dist_negated_kw"])
    print("Flip@Example (>=1 flip):", fmt_pct(res["flip_example_rate"]))
    print("Flips / 100 examples:", f"{res['flips_per_100']:.2f}")
    print("Flips / negated kw:", f"{res['flips_per_neg_kw']:.3f}")
    print("Totals: neg_kw =", res["neg_total"], "| flips =", res["flip_total"])


if __name__ == "__main__":
    main()
