#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
zs_summarization_adaptive.py

Extension: Adaptive Keyphrase Budget (Dynamic K) for SigExt prompting.

Adds a new kw_strategy:
  - sigext_adaptive: choose K per document (instead of fixed --kw_model_top_k)

Dynamic K modes:
  - len  : K = clamp(ceil(a + b * log(n_tokens)), k_min, k_max)
  - mass : choose smallest K such that sum(sigmoid(score_i)) >= tau, then clamp

Also stores per-example debug info (k_doc, n_tokens, etc.) in `other_info`.
"""

import argparse
import json
import logging
import math
import os
import pathlib
from collections import defaultdict
from multiprocessing import Pool

import jsonlines
import nltk
import numpy as np
import tqdm
from nltk.tokenize import word_tokenize
from rapidfuzz import fuzz
from rouge_score import rouge_scorer

from bedrock_utils import predict_one_eg_mistral, predict_one_eg_claude_instant
from prompts import (
    ZS_NAIVE_PROMPT_STR_FOR_MISTRAL,
    ZS_NAIVE_PROMPT_STR_FOR_CLAUDE,
    ZS_KEYWORD_PROMPT_STR_FOR_MISTRAL,
    ZS_KEYWORD_PROMPT_STR_FOR_CLAUDE,
)

ZS_NAIVE_PROMPT_STR = {
    "mistral": ZS_NAIVE_PROMPT_STR_FOR_MISTRAL,
    "claude": ZS_NAIVE_PROMPT_STR_FOR_CLAUDE,
}

ZS_KEYWORD_PROMPT_STR = {
    "mistral": ZS_KEYWORD_PROMPT_STR_FOR_MISTRAL,
    "claude": ZS_KEYWORD_PROMPT_STR_FOR_CLAUDE,
}


def estimate_logits_threshold(dataset_file, percentile_threshold):
    """
    Threshold computed on validation set: take percentile over all phrase scores.
    (This is the same as your original file.)
    """
    if not os.path.exists(dataset_file):
        logging.warning("validation set not found for logits threshold.")
        return -1

    with jsonlines.open(dataset_file) as f:
        data = list(f)

    if not data or "input_kw_model" not in data[0]:
        logging.warning("input_kw_model not found in the file. Use -1 as threshold.")
        return -1

    logits = []
    for item in data:
        for kw_info_model in item.get("input_kw_model", []):
            logits.append(kw_info_model["score"])
    if not logits:
        return -1
    return float(np.percentile(logits, percentile_threshold))


def compute_dynamic_k(
    example,
    dataset_name,
    mode="len",
    k_min=None,
    k_max=None,
    a=2.0,
    b=2.0,
    tau=0.85,
):
    """
    Compute per-document K.

    Defaults for k_min/k_max follow the paper's K search ranges:
      - cnn/samsum/meetingbank: [10, 20]
      - arxiv: [30, 40]
    :contentReference[oaicite:0]{index=0}
    """
    if k_min is None or k_max is None:
        if dataset_name == "arxiv":
            k_min = 30
            k_max = 40
        else:
            k_min = 10
            k_max = 20

    text = example.get("trunc_input", "") or ""
    n_tokens = max(1, len(text.split()))

    if mode == "len":
        # K = clamp(ceil(a + b*log(n_tokens)), k_min, k_max)
        k = int(math.ceil(a + b * math.log(n_tokens)))
        return max(int(k_min), min(int(k_max), k))

    if mode == "mass":
        # Choose smallest K such that sum(sigmoid(score_i)) >= tau, then clamp.
        # Note: caller should pass candidate scores sorted desc for efficiency/consistency.
        cum = 0.0
        k = 0
        for kw in example.get("input_kw_model", []):
            s = float(kw.get("score", 0.0))
            # sigmoid
            p = 1.0 / (1.0 + math.exp(-s))
            cum += p
            k += 1
            if cum >= float(tau):
                break
        return max(int(k_min), min(int(k_max), k))

    raise ValueError(f"Unknown adaptive_mode={mode}")


class NaivePrompt(object):
    def __init__(self, model_name, dataset_name, customized_prompt=None):
        self.prompt = customized_prompt or ZS_NAIVE_PROMPT_STR[model_name][dataset_name]

    def __call__(self, example):
        return self.prompt.replace("<text>", example["trunc_input"])


def remove_duplicate_top_k(candidates, top_k, threshold=70):
    """
    Keep up to top_k, removing near-duplicates by fuzzy string ratio.
    (This is your original logic.)
    """
    ret = []
    for candidate in candidates:
        to_delete = set()
        to_skip = False

        if len(ret) >= top_k:
            break

        for added_kw in ret:
            if fuzz.ratio(added_kw["phrase"].lower(), candidate["phrase"].lower()) >= threshold:
                if len(added_kw["phrase"]) <= len(candidate["phrase"]):
                    to_delete.add(added_kw["phrase"])
                else:
                    to_skip = True

        ret = [item for item in ret if item["phrase"] not in to_delete]

        if not to_skip:
            ret.append(candidate)

    return ret


class SegExtTopK(object):
    """
    Fixed-K SigExt prompting (original), plus optional adaptive K.
    """

    def __init__(
        self,
        model_name,
        dataset_name,
        top_k,
        deduplicate=True,
        logits_threshold=-1,
        use_rank=False,
        customized_prompt=None,
        adaptive=False,
        adaptive_mode="len",
        k_min=None,
        k_max=None,
        k_a=2.0,
        k_b=2.0,
        k_tau=0.85,
    ):
        self.prompt = customized_prompt or ZS_KEYWORD_PROMPT_STR[model_name][dataset_name]
        self.dataset_name = dataset_name

        self.top_k = int(top_k)
        self.deduplicate = bool(deduplicate)
        self.logits_threshold = float(logits_threshold) if logits_threshold is not None else -1
        self.use_rank = bool(use_rank)

        self.adaptive = bool(adaptive)
        self.adaptive_mode = adaptive_mode
        self.k_min = k_min
        self.k_max = k_max
        self.k_a = float(k_a)
        self.k_b = float(k_b)
        self.k_tau = float(k_tau)

    def __call__(self, example):
        # Build ranked candidate list
        if self.use_rank:
            # rank-based baseline (kept for parity)
            selected_keywords = sorted(example["trunc_input_phrases"], key=lambda x: x["rank"])
            # store rank index as score
            selected_keywords_scored = []
            for i, kw in enumerate(selected_keywords):
                ph = dict(kw)  # avoid mutating original
                ph["score"] = float(i)
                selected_keywords_scored.append(ph)
            selected_keywords = selected_keywords_scored
        else:
            # model-score-based
            selected_keywords = []
            for kw_info in sorted(example.get("input_kw_model", []), key=lambda x: x["score"], reverse=True):
                if kw_info["score"] < self.logits_threshold:
                    break
                kw_index = kw_info["kw_index"]
                if kw_index >= len(example["trunc_input_phrases"]):
                    continue
                ph = dict(example["trunc_input_phrases"][kw_index])  # IMPORTANT: copy
                ph["score"] = float(kw_info["score"])
                selected_keywords.append(ph)

        # Decide K per document
        if self.adaptive:
            k_doc = compute_dynamic_k(
                example=example,
                dataset_name=self.dataset_name,
                mode=self.adaptive_mode,
                k_min=self.k_min,
                k_max=self.k_max,
                a=self.k_a,
                b=self.k_b,
                tau=self.k_tau,
            )
        else:
            k_doc = self.top_k

        # Dedup + truncate to K_doc
        if self.deduplicate:
            selected_keywords = remove_duplicate_top_k(selected_keywords, top_k=k_doc)
        else:
            selected_keywords = selected_keywords[:k_doc]

        formatted_keywords = "; ".join([item["phrase"] for item in selected_keywords]) + "."

        prompt = (
            self.prompt.replace("<text>", example["trunc_input"])
            .replace("<keywords>", formatted_keywords)
        )

        # Return prompt plus metadata so run_inference can store it in other_info
        other_info = {
            "k_doc": int(k_doc),
            "adaptive": bool(self.adaptive),
            "adaptive_mode": self.adaptive_mode if self.adaptive else None,
            "n_tokens": int(max(1, len((example.get("trunc_input") or "").split()))),
            "n_candidates_after_thr": int(len(selected_keywords)),
            "logits_threshold": float(self.logits_threshold),
        }
        return prompt, other_info


def get_prompt_fn(
    model_name,
    dataset,
    kw_strategy,
    kw_model_top_k,
    logits_threshold,
    adaptive_mode="len",
    k_min=None,
    k_max=None,
    k_a=2.0,
    k_b=2.0,
    k_tau=0.85,
):
    if kw_strategy == "disable":
        return NaivePrompt(model_name, dataset)

    if kw_strategy == "sigext_topk":
        return SegExtTopK(
            model_name,
            dataset,
            top_k=kw_model_top_k,
            logits_threshold=logits_threshold,
            adaptive=False,
        )

    if kw_strategy == "sigext_adaptive":
        return SegExtTopK(
            model_name,
            dataset,
            top_k=kw_model_top_k,  # fallback; unused when adaptive=True
            logits_threshold=logits_threshold,
            adaptive=True,
            adaptive_mode=adaptive_mode,
            k_min=k_min,
            k_max=k_max,
            k_a=k_a,
            k_b=k_b,
            k_tau=k_tau,
        )

    raise RuntimeError("unknown kw strategy.")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels


def compute_rouge_score(inference_data, preds):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
    labels = [item["raw_output"] for item in inference_data]
    decoded_preds, decoded_labels = postprocess_text(preds, labels)

    result_element = defaultdict(list)
    for pred, label in zip(decoded_preds, decoded_labels):
        score = scorer.score(target=label, prediction=pred)
        for metric_name, value in score.items():
            result_element[f"{metric_name}p"].append(value.precision)
            result_element[f"{metric_name}r"].append(value.recall)
            result_element[f"{metric_name}f"].append(value.fmeasure)

    result = {metric_name: float(np.mean(values)) for metric_name, values in result_element.items()}
    result = {k: round(v * 100, 4) for k, v in result.items()}

    prediction_lens = [len(word_tokenize(pred)) for pred in preds]
    result["gen_len"] = float(np.mean(prediction_lens)) if prediction_lens else 0.0
    return result


def run_inference(
    model_name,
    kw_strategy,
    kw_model_top_k,
    dataset,
    dataset_dir,
    output_dir,
    inference_on_split="test",
    adaptive_mode="len",
    k_min=None,
    k_max=None,
    k_a=2.0,
    k_b=2.0,
    k_tau=0.85,
):
    dataset_dir = pathlib.Path(dataset_dir)

    logits_threshold = estimate_logits_threshold(str(dataset_dir.joinpath("validation.jsonl")), 75)
    logging.info(f"logits threshold is {logits_threshold}")

    if model_name == "mistral":
        predict_one_eg_fn = predict_one_eg_mistral
    elif model_name == "claude":
        predict_one_eg_fn = predict_one_eg_claude_instant
    else:
        raise ValueError(f"invalid model name {model_name}")

    prompting_fn = get_prompt_fn(
        model_name=model_name,
        dataset=dataset,
        kw_strategy=kw_strategy,
        kw_model_top_k=kw_model_top_k,
        logits_threshold=logits_threshold,
        adaptive_mode=adaptive_mode,
        k_min=k_min,
        k_max=k_max,
        k_a=k_a,
        k_b=k_b,
        k_tau=k_tau,
    )
    assert not isinstance(prompting_fn, dict)

    dataset_filename = str(dataset_dir.joinpath(f"{inference_on_split}.jsonl"))
    with jsonlines.open(dataset_filename) as f:
        inference_data = list(f)

    # Build prompts
    with Pool(1) as p:
        all_prompt = list(tqdm.tqdm(p.imap(prompting_fn, inference_data), total=len(inference_data)))

    for i in range(len(inference_data)):
        if isinstance(all_prompt[i], str):
            inference_data[i]["prompt_input"] = all_prompt[i]
        else:
            inference_data[i]["prompt_input"] = all_prompt[i][0]
            inference_data[i]["other_info"] = all_prompt[i][1]

    # LLM calls
    with Pool(1) as p:
        all_res = list(tqdm.tqdm(p.imap(predict_one_eg_fn, inference_data), total=len(inference_data)))
    all_res = [item for item in all_res]

    output_path = str(pathlib.Path(output_dir).expanduser())
    os.makedirs(output_path, exist_ok=True)

    with jsonlines.open(os.path.join(output_path, f"{inference_on_split}_dataset.jsonl"), "w") as f:
        f.write_all(inference_data)

    with open(os.path.join(output_path, f"{inference_on_split}_predictions.json"), "w") as f:
        json.dump(all_res, f, indent=2)

    test_metrics = compute_rouge_score(inference_data, all_res)
    with open(str(pathlib.Path(output_dir).joinpath(f"{inference_on_split}_metrics.json")), "w") as f:
        json.dump(test_metrics, f, indent=2)

    logging.info("Wrote metrics to %s", str(pathlib.Path(output_dir).joinpath(f"{inference_on_split}_metrics.json")))


def main():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("transformers.generation").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="mistral", choices=["claude", "mistral"], help="llm name")
    parser.add_argument(
        "--kw_strategy",
        choices=["disable", "sigext_topk", "sigext_adaptive"],
        required=True,
        help="keyword strategy.",
    )
    parser.add_argument("--kw_model_top_k", default=20, type=int, help="Fixed top-K (used for sigext_topk).")

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["arxiv", "pubmed", "cnn", "samsum", "meetingbank"],
        help="Select from supported datasets.",
    )
    parser.add_argument("--dataset_dir", required=True, type=str, help="directory of data (jsonl files).")
    parser.add_argument("--output_dir", required=True, type=str, help="directory to save experiment.")
    parser.add_argument("--inference_on_split", default="test", type=str, help="split_to_run_inference")

    # Adaptive-K options (used only for kw_strategy=sigext_adaptive)
    parser.add_argument("--adaptive_mode", default="len", choices=["len", "mass"], help="Dynamic K mode.")
    parser.add_argument("--k_min", default=None, type=int, help="Min K for adaptive mode (optional).")
    parser.add_argument("--k_max", default=None, type=int, help="Max K for adaptive mode (optional).")
    parser.add_argument("--k_a", default=2.0, type=float, help="a for K=ceil(a+b*log(tokens))")
    parser.add_argument("--k_b", default=2.0, type=float, help="b for K=ceil(a+b*log(tokens))")
    parser.add_argument("--k_tau", default=0.85, type=float, help="tau mass threshold for adaptive_mode=mass")

    args = parser.parse_args()
    run_inference(**vars(args))


if __name__ == "__main__":
    main()
