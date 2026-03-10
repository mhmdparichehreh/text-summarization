#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
zs_summarization_2stage.py

Implements the "2-stage" baseline for SigExt:
  Stage 1: Extract top-N salient sentences from the document.
  Stage 2: Rewrite the extracted sentences into a fluent summary.

Assumes dataset jsonl has:
  - trunc_input (document text)
  - raw_output  (reference summary)

Outputs (like zs_summarization.py):
  - {split}_dataset.jsonl      (copied inference data)
  - {split}_predictions.json   (list of generated summaries)
  - {split}_metrics.json       (ROUGE metrics)

Works with your existing bedrock_utils.py:
  - predict_one_eg_claude_instant expects <summary> tags (it also auto-adds the instruction)
  - predict_one_eg_mistral returns raw text
"""

import argparse
import json
import logging
import pathlib
from collections import defaultdict

import jsonlines
import nltk
import numpy as np
import tqdm
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer

from bedrock_utils import predict_one_eg_mistral, predict_one_eg_claude_instant


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels


def compute_rouge_score(inference_data, preds):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"],
        use_stemmer=True
    )

    labels = [item["raw_output"] for item in inference_data]
    decoded_preds, decoded_labels = postprocess_text(preds, labels)

    result_element = defaultdict(list)
    for pred, label in zip(decoded_preds, decoded_labels):
        score = scorer.score(target=label, prediction=pred)
        for metric_name, value in score.items():
            result_element[f"{metric_name}p"].append(value.precision)
            result_element[f"{metric_name}r"].append(value.recall)
            result_element[f"{metric_name}f"].append(value.fmeasure)

    result = {k: float(np.mean(v)) for k, v in result_element.items()}
    result = {k: round(v * 100, 4) for k, v in result.items()}

    prediction_lens = [len(word_tokenize(pred)) for pred in preds]
    result["gen_len"] = float(np.mean(prediction_lens)) if prediction_lens else 0.0
    return result


def _predict(model_name: str, prompt_input: str) -> str:
    x = {"prompt_input": prompt_input}
    if model_name == "claude":
        return predict_one_eg_claude_instant(x) or ""
    elif model_name == "mistral":
        return predict_one_eg_mistral(x) or ""
    else:
        raise ValueError(f"Unsupported model_name={model_name}")


def build_stage1_extract_prompt(doc: str, n_sent: int) -> str:
    # IMPORTANT:
    # - Claude wrapper *automatically* appends: "Write your summary in <summary> XML tags."
    # - So we keep stage-1 output inside <summary> too (as a list of extracted sentences).
    return f"""
You are given a document. Extract the {n_sent} most important sentences that together cover the key information.
Rules:
- Copy sentences verbatim from the document (no paraphrasing).
- One sentence per line.
- Do not add any new information not present in the document.
- If the document has fewer than {n_sent} good sentences, extract as many as possible.

Document:
{doc}
""".strip()


def build_stage2_rewrite_prompt(extracted_sents: str, dataset: str) -> str:
    # Keep it generic; you can add dataset-specific length constraints if you want.
    # Again: Claude wrapper will enforce <summary> tags, which is exactly what we want for final output.
    return f"""
You are given extracted key sentences from a document. Write a coherent, fluent summary based ONLY on these sentences.
Rules:
- Do not introduce facts that are not present in the extracted sentences.
- Merge/rewrite for readability.
- Keep it concise and focused.

Extracted sentences:
{extracted_sents}

Dataset: {dataset}
""".strip()


def run_2stage(model_name: str,
              dataset: str,
              dataset_dir: str,
              output_dir: str,
              inference_on_split: str,
              extract_sentences: int):

    dataset_dir = pathlib.Path(dataset_dir)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_file = dataset_dir / f"{inference_on_split}.jsonl"
    if not dataset_file.exists():
        raise FileNotFoundError(f"Missing dataset file: {dataset_file}")

    # Ensure punkt exists for nltk.sent_tokenize
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    with jsonlines.open(str(dataset_file)) as f:
        inference_data = list(f)

    logging.info(f"Loaded {len(inference_data)} examples from {dataset_file}")

    preds = []
    stage1_outputs = []

    for item in tqdm.tqdm(inference_data, desc="2-stage inference"):
        doc = (item.get("trunc_input") or "").strip()

        # --- Stage 1: extraction ---
        p1 = build_stage1_extract_prompt(doc, extract_sentences)
        extracted = _predict(model_name, p1).strip()

        # fallback: if extraction failed, just use the document itself (last-resort)
        if not extracted:
            extracted = doc[:2000].strip()

        stage1_outputs.append(extracted)

        # --- Stage 2: rewrite ---
        p2 = build_stage2_rewrite_prompt(extracted, dataset)
        summary = _predict(model_name, p2).strip()

        preds.append(summary)

    # Save artifacts (same style as zs_summarization.py)
    with jsonlines.open(str(output_dir / f"{inference_on_split}_dataset.jsonl"), "w") as f:
        f.write_all(inference_data)

    with open(str(output_dir / f"{inference_on_split}_predictions.json"), "w") as f:
        json.dump(preds, f, indent=2)

    # Optional: keep stage1 evidence (very useful for debugging)
    with open(str(output_dir / f"{inference_on_split}_stage1_extracted.json"), "w") as f:
        json.dump(stage1_outputs, f, indent=2)

    metrics = compute_rouge_score(inference_data, preds)
    with open(str(output_dir / f"{inference_on_split}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logging.info(f"Done. Metrics: {metrics}")
    print(json.dumps(metrics, indent=2))


def main():
    logging.basicConfig(level=logging.INFO)

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True, choices=["claude", "mistral"])
    ap.add_argument("--dataset", required=True, type=str)
    ap.add_argument("--dataset_dir", required=True, type=str)
    ap.add_argument("--output_dir", required=True, type=str)
    ap.add_argument("--inference_on_split", default="test", type=str)
    ap.add_argument("--extract_sentences", default=8, type=int,
                    help="How many sentences to extract in Stage 1 (2-stage baseline).")
    args = ap.parse_args()

    run_2stage(
        model_name=args.model_name,
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        inference_on_split=args.inference_on_split,
        extract_sentences=args.extract_sentences,
    )


if __name__ == "__main__":
    main()
