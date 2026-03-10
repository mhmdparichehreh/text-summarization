#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_negation_extension.py (STRICT NEGATION SCOPE VERSION)

Reads SigExt run outputs (test_dataset.jsonl + test_predictions.json)
and computes:

1) Negation exposure (how often selected keywords are negated in the source)
2) Faithfulness metric: negation flip rate
3) Optional: prints examples of flips for qualitative analysis

Supports BOTH:
- baseline runs (sigext_topk)
- neg-aware runs (sigext_topk_neg) with styles: NOT(...), [NEG], (negated)

Key upgrade:
- Adds strict negation scope detection (token-window + high-precision patterns)
- You can choose via --scope {loose, strict}

Usage:
python analyze_negation_extension.py \
  --run_dir /content/drive/MyDrive/experiments/cnn_sigext_mistral_k15 \
  --scope strict \
  --max_examples 500 \
  --print_top_flips 10
"""

import argparse
import json
import os
import pathlib
import re
from collections import Counter

import jsonlines
from rapidfuzz import fuzz


# ----------------------------
# Negation cues
# ----------------------------
NEG_SINGLE = {
    "no", "not", "never", "none", "without",
    "cannot", "can't", "won't", "isn't", "aren't", "wasn't", "weren't",
    "don't", "doesn't", "didn't",
    "hasn't", "haven't", "hadn't",
    "shouldn't", "couldn't", "wouldn't", "mustn't",
}

NEG_MULTI = [
    "do not", "does not", "did not",
    "has not", "have not", "had not",
    "will not", "would not", "should not", "could not", "must not",
    "not expected to", "not likely to", "no plans to",
    "deny", "denies", "denied",
    "fail to", "fails to", "failed to",
    "lack of", "rule out", "ruled out", "no evidence of", "no sign of",
]

FALSE_POS = ["not only"]


# ----------------------------
# Sentence bounds (same idea as before)
# ----------------------------
def find_sentence_bounds(text: str, idx: int):
    if idx is None or idx < 0 or not text:
        return 0, len(text) if text else 0

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


# ----------------------------
# LOOSE negation (your original)
# ----------------------------
def is_negated_in_context_loose(text: str, phrase_start: int, phrase: str) -> bool:
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

    for cue in NEG_SINGLE:
        if cue in pre or cue in around:
            if cue == "not" and fp_hit:
                continue
            return True

    return False


# ----------------------------
# STRICT negation scope (new)
# ----------------------------
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

STRICT_PRE_WIN = 6
STRICT_POST_WIN = 3

STRICT_PATTERNS_PRE = [
    "no evidence of",
    "no sign of",
    "lack of",
]

STRICT_PATTERNS_VERB = [
    "rule out", "ruled out",
    "deny", "denies", "denied",
    "fail to", "fails to", "failed to",
]

def tok(text: str):
    return TOKEN_RE.findall((text or "").lower())

def find_sublist(hay, needle):
    if not needle or not hay or len(needle) > len(hay):
        return None
    for i in range(len(hay) - len(needle) + 1):
        if hay[i:i+len(needle)] == needle:
            return (i, i+len(needle))
    return None

def is_negated_in_sentence_strict(sentence: str, phrase: str,
                                 pre_win: int = STRICT_PRE_WIN,
                                 post_win: int = STRICT_POST_WIN) -> bool:
    """
    Strict negation: phrase is negated only if:
    - a negation cue occurs within a small token window around phrase span, OR
    - high-precision negation patterns occur close to phrase mention
    """
    if not sentence or not phrase:
        return False

    stoks = tok(sentence)
    ptoks = tok(phrase)
    span = find_sublist(stoks, ptoks)
    if span is None:
        # conservative: if we can't align phrase tokens, don't call it negated
        return False

    s, e = span
    left = max(0, s - pre_win)
    right = min(len(stoks), e + post_win)
    window = stoks[left:right]
    window_str = " ".join(window)

    # false-positive guard: "not only"
    if "not only" in window_str:
        # treat as not-negation unless another cue exists besides "not"
        other_cues = [w for w in window if w in NEG_SINGLE and w != "not"]
        if not other_cues:
            return False

    if any(w in NEG_SINGLE for w in window):
        return True

    sent_low = sentence.lower()
    phrase_low = phrase.lower()
    idx = sent_low.find(phrase_low)

    # pre-patterns must appear before phrase mention
    if idx != -1:
        prefix = sent_low[max(0, idx - 200): idx]
        for p in STRICT_PATTERNS_PRE:
            if p in prefix:
                return True

        # verb patterns can be around phrase (±120 chars)
        around = sent_low[max(0, idx - 120): min(len(sent_low), idx + len(phrase_low) + 120)]
        for p in STRICT_PATTERNS_VERB:
            if p in around:
                return True

    return False

def is_negated_in_context_strict(text: str, phrase_start: int, phrase: str) -> bool:
    if not text or phrase_start is None or phrase_start < 0:
        return False
    phrase = (phrase or "").strip()
    if not phrase:
        return False
    left, right = find_sentence_bounds(text, phrase_start)
    sentence = text[left:right]
    return is_negated_in_sentence_strict(sentence, phrase)


# ----------------------------
# Keyword extraction from prompt_input
# ----------------------------
def extract_keywords_from_prompt(prompt: str):
    """
    Extract the <keywords> list injected into the prompt.
    Relies on substring "Consider include the following information:" from prompts.py
    Splits on ';'
    """
    if not isinstance(prompt, str):
        return []
    marker = "Consider include the following information:"
    if marker not in prompt:
        return []

    tail = prompt.split(marker, 1)[1].strip()
    tail = tail.replace("[/INST]", " ").strip()
    tail = tail.split("\n")[0].strip()

    if tail.endswith("."):
        tail = tail[:-1].strip()

    parts = [p.strip() for p in tail.split(";")]
    return [p for p in parts if p]


def normalize_keyword(kw: str):
    kw = (kw or "").strip()

    m = re.match(r"^NOT\((.*)\)$", kw.strip(), flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(), True

    m = re.match(r"^\[NEG\]\s*(.*)$", kw.strip(), flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(), True

    m = re.match(r"^(.*)\s*\(negated\)$", kw.strip(), flags=re.IGNORECASE)
    if m:
        return m.group(1).strip(), True

    return kw, False


# ----------------------------
# Matching keyword in text / summary
# ----------------------------
def find_phrase_start(text: str, phrase: str):
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

    L = len(p)
    if L < 6:
        return False, -1

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


def strict_local_negation_in_summary(summary: str, phrase: str, match_start_char: int, win: int = 5) -> bool:
    """
    Strict summary-side: tokenize the containing sentence, find phrase token span,
    then look for negation cues within +/- win tokens.
    """
    if not summary or match_start_char < 0 or not phrase:
        return False

    # extract containing sentence roughly
    l, r = find_sentence_bounds(summary, match_start_char)
    sent = summary[l:r]

    stoks = tok(sent)
    ptoks = tok(phrase)
    span = find_sublist(stoks, ptoks)
    if span is None:
        # fallback: if can't align, do a small-window string check around match
        around = sent.lower()[max(0, (match_start_char - l) - 60): min(len(sent), (match_start_char - l) + len(phrase) + 60)]
        if "not only" in around:
            # do not treat "not only" as negation
            return any(cue in around for cue in ["no ", "never", "without", "denied", "ruled out", "rule out"])
        return any(cue in around for cue in NEG_MULTI) or any(cue in around.split() for cue in NEG_SINGLE)

    s, e = span
    left = max(0, s - win)
    right = min(len(stoks), e + win)
    window = stoks[left:right]
    window_str = " ".join(window)

    # guard for "not only"
    if "not only" in window_str:
        other = [w for w in window if w in NEG_SINGLE and w != "not"]
        if not other:
            return False

    if any(w in NEG_SINGLE for w in window):
        return True

    # multi cues in window string
    for cue in NEG_MULTI:
        if cue in window_str:
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
        n = min(len(preds), len(data))
        data = data[:n]
        preds = preds[:n]

    return data, preds


def analyze_run(run_dir: str, scope: str = "strict", max_examples=None, print_top_flips=0):
    data, preds = load_run(run_dir)
    if max_examples is not None:
        data = data[:max_examples]
        preds = preds[:max_examples]

    if scope not in {"loose", "strict"}:
        raise ValueError("scope must be one of: loose, strict")

    K_counts = []
    neg_counts = []
    exposure = 0

    flip_examples = 0
    flip_total = 0
    neg_total = 0

    flip_records = []

    for ex, summary in zip(data, preds):
        text = ex.get("trunc_input", "")
        prompt = ex.get("prompt_input", "")

        kws_raw = extract_keywords_from_prompt(prompt)
        kws = []
        for k in kws_raw:
            base, _marked = normalize_keyword(k)
            if base:
                kws.append(base)

        K = len(kws)
        if K == 0:
            continue
        K_counts.append(K)

        ex_neg_count = 0
        ex_flip = 0

        for phrase in kws:
            start = find_phrase_start(text, phrase)
            if start == -1:
                continue

            if scope == "loose":
                neg = is_negated_in_context_loose(text, start, phrase)
            else:
                neg = is_negated_in_context_strict(text, start, phrase)

            if neg:
                ex_neg_count += 1
                neg_total += 1

                present, match_start = phrase_in_summary(summary, phrase)
                if present:
                    # strict local negation check in summary (recommended)
                    local_neg = strict_local_negation_in_summary(summary, phrase, match_start, win=5)
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

    res = {
        "run_dir": str(run_dir),
        "scope": scope,
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

    if print_top_flips > 0 and flip_records:
        print("\n=== TOP FLIP EXAMPLES (for qualitative analysis) ===")
        for rec in flip_records[:print_top_flips]:
            print("\n---")
            print("PHRASE:", rec["phrase"])
            print("SOURCE SNIPPET:", rec["source_snippet"].replace("\n", " "))
            print("SUMMARY:", str(rec["summary"]).replace("\n", " "))

    return res


def fmt_pct(x):
    return f"{100.0*x:.2f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, type=str,
                    help="One run folder containing test_dataset.jsonl + test_predictions.json")
    ap.add_argument("--scope", default="strict", choices=["loose", "strict"],
                    help="Negation scope: loose (sentence cue) vs strict (token window + patterns)")
    ap.add_argument("--max_examples", default=None, type=int)
    ap.add_argument("--print_top_flips", default=0, type=int)
    args = ap.parse_args()

    res = analyze_run(
        args.run_dir,
        scope=args.scope,
        max_examples=args.max_examples,
        print_top_flips=args.print_top_flips,
    )

    print("\n=== Negation Extension Analysis ===")
    print("Run:", res["run_dir"])
    print("Scope:", res["scope"])
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
