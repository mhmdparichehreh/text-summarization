#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pathlib
import re
import statistics
from collections import Counter

import jsonlines

MARKER = "Consider include the following information:"


def extract_keywords_from_prompt(prompt: str):
    if not isinstance(prompt, str) or MARKER not in prompt:
        return []

    tail = prompt.split(MARKER, 1)[1].strip()
    tail = tail.replace("[/INST]", " ").strip()
    tail = tail.split("\n")[0].strip()
    if tail.endswith("."):
        tail = tail[:-1].strip()

    parts = [p.strip() for p in tail.split(";")]
    return [p for p in parts if p]


def normalize_kw(kw: str):
    kw = (kw or "").strip()

    m = re.match(r"^NOT\((.*)\)$", kw, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    m = re.match(r"^\[NEG\]\s*(.*)$", kw, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    m = re.match(r"^(.*)\s*\(negated\)$", kw, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    return kw


def load_k_values(run_dir: str, split: str = "test"):
    run_dir = pathlib.Path(run_dir)
    ds_path = run_dir / f"{split}_dataset.jsonl"

    if not ds_path.exists():
        raise FileNotFoundError(f"Missing {ds_path}")

    ks = []
    with jsonlines.open(str(ds_path)) as f:
        for ex in f:
            prompt = ex.get("prompt_input", "")
            kws = extract_keywords_from_prompt(prompt)
            kws = [normalize_kw(k) for k in kws]
            kws = [k for k in kws if k]
            ks.append(len(kws))
    return ks


def summarize(ks):
    c = Counter(ks)
    avg = sum(ks) / len(ks)
    mn, mx = min(ks), max(ks)
    med = statistics.median(ks)

    if len(ks) >= 4:
        q = statistics.quantiles(ks, n=4)
        p25, p75 = q[0], q[2]
    else:
        p25, p75 = mn, mx

    mode = c.most_common(1)[0][0]
    return {
        "n": len(ks),
        "avg": avg,
        "min": mn,
        "max": mx,
        "median": med,
        "p25": p25,
        "p75": p75,
        "mode": mode,
        "counts": dict(sorted(c.items())),
    }


def save_plot(ks, run_name, out_path):
    import matplotlib.pyplot as plt

    mn, mx = min(ks), max(ks)
    bins = list(range(mn, mx + 2))  # integer bins

    s = summarize(ks)

    plt.figure(figsize=(7.2, 4.0))  # readable in IEEE 2-col if scaled
    plt.hist(ks, bins=bins, edgecolor="black", linewidth=0.6)

    plt.xlabel("Injected keyword budget $K$")
    plt.ylabel("#Examples")
    plt.title(f"Adaptive-K distribution: {run_name}")

    # integer ticks
    plt.xticks(list(range(mn, mx + 1)))

    # grid (light)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    # mean/median markers
    plt.axvline(s["avg"], linestyle="--", linewidth=1.0, label=f"Mean={s['avg']:.2f}")
    plt.axvline(s["median"], linestyle="-", linewidth=1.0, label=f"Median={s['median']:.1f}")
    plt.legend(loc="upper right", frameon=True)

    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, type=str)
    ap.add_argument("--split", default="test", choices=["test", "validation"])
    ap.add_argument("--plot_out", default=None, type=str)
    args = ap.parse_args()

    ks = load_k_values(args.run_dir, split=args.split)
    s = summarize(ks)

    print("\n=== Adaptive-K Budget Stats ===")
    print("Run:", args.run_dir)
    print("Split:", args.split)
    print("N:", s["n"])
    print(f"Avg K: {s['avg']:.2f}")
    print(f"Min–Max: {s['min']}–{s['max']}")
    print(f"Median: {s['median']} | P25–P75: {s['p25']}–{s['p75']}")
    print(f"Mode: {s['mode']}")
    print("K counts:", s["counts"])

    if args.plot_out:
        run_name = pathlib.Path(args.run_dir).name
        save_plot(ks, run_name, args.plot_out)
        print("Saved plot:", args.plot_out)


if __name__ == "__main__":
    main()
