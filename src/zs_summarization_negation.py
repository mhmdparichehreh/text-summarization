# zs_summarization_negation.py
# Based on your ORIGINAL zs_summarization.py, with ONE extension:
#   - adds kw_strategy = "sigext_topk_neg"
#   - detects negation around selected keyphrases (domain-agnostic cues for cnn/arxiv/meetingbank/samsum)
#   - annotates negated keyphrases in <keywords> as NOT(phrase) (or other styles)
#
# Everything else (prompt templates, ROUGE, IO format) stays consistent with the original script.

import argparse
import json
import logging
import os
import pathlib
from collections import defaultdict
from multiprocessing import Pool
from typing import Tuple

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
    if not os.path.exists(dataset_file):
        logging.warning("validation set not found for logits threshold.")
        return -1

    with jsonlines.open(dataset_file) as f:
        data = list(f)

    if not data:
        logging.warning("validation set is empty for logits threshold.")
        return -1

    if "input_kw_model" not in data[0]:
        logging.warning("input_kw_model not found in the file. Use -1 as threshold.")
        return -1

    logits = []
    for item in data:
        for kw_info_model in item.get("input_kw_model", []):
            logits.append(kw_info_model["score"])

    if not logits:
        logging.warning("no logits found in validation. Use -1 as threshold.")
        return -1

    return float(np.percentile(logits, percentile_threshold))


class NaivePrompt(object):
    def __init__(self, model_name, dataset_name, customized_prompt=None):
        self.prompt = customized_prompt or ZS_NAIVE_PROMPT_STR[model_name][dataset_name]

    def __call__(self, example):
        return self.prompt.replace("<text>", example["trunc_input"])


def remove_duplicate_top_k(candidates, top_k, threshold=70):
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


# ----------------------------
# Negation detection (domain-agnostic cues)
# ----------------------------

# Single-token cues / contractions
NEGATION_CUES_SINGLE = [
    "no",
    "not",
    "never",
    "none",
    "without",
    "cannot",
    "can't",
    "won't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
]

# Multiword cues (higher precision)
NEGATION_CUES_MULTI = [
    "do not",
    "does not",
    "did not",
    "has not",
    "have not",
    "had not",
    "will not",
    "would not",
    "should not",
    "could not",
    "must not",
    "not expected to",
    "not likely to",
    "no plans to",
    "deny",
    "denies",
    "denied",
    "fail to",
    "fails to",
    "failed to",
    "lack of",
    "rule out",
    "ruled out",
    "no evidence of",
]

# Basic false-positive guard
NEGATION_FALSE_POSITIVE_PATTERNS = [
    "not only",  # "not only X but also Y"
]


def _find_sentence_bounds(text: str, idx: int) -> Tuple[int, int]:
    """
    Rough sentence boundaries around idx using punctuation/newlines (no extra dependencies).
    """
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
    right = len(text) if right == -1 else right
    return left, right


def is_negated_in_context(text: str, phrase_start: int, phrase: str) -> bool:
    """
    Conservative rule-based negation detection.
    Checks for negation cues within the same rough sentence and near the phrase.
    """
    if not text or phrase_start is None or phrase_start < 0:
        return False

    phrase = (phrase or "").strip()
    if not phrase:
        return False

    left, right = _find_sentence_bounds(text, phrase_start)
    sent = text[left:right].lower()

    local_start = max(0, phrase_start - left)
    local_end = min(len(sent), local_start + len(phrase))

    pre_window = sent[max(0, local_start - 140): local_start]
    post_window = sent[local_end: min(len(sent), local_end + 100)]
    around_window = sent[max(0, local_start - 50): min(len(sent), local_end + 50)]

    # False positive guard
    fp_hit = any(fp in around_window for fp in NEGATION_FALSE_POSITIVE_PATTERNS)

    # Multiword cues: look in pre/around/post (captures "ruled out" after phrase)
    for cue in NEGATION_CUES_MULTI:
        if cue in pre_window or cue in around_window or cue in post_window:
            # avoid "not only" being treated as negation for "not"
            if fp_hit and cue == "not":
                continue
            return True

    # Single cues: closer proximity only (pre/around)
    for cue in NEGATION_CUES_SINGLE:
        if cue in pre_window or cue in around_window:
            if fp_hit and cue == "not":
                continue
            return True

    return False


def format_phrase_with_negation(phrase: str, negated: bool, style: str) -> str:
    """
    style:
      - NOT: NOT(phrase)
      - TAG: [NEG] phrase
      - PAREN: phrase (negated)
    """
    phrase = (phrase or "").strip()
    if not negated:
        return phrase

    s = (style or "NOT").upper()
    if s == "TAG":
        return f"[NEG] {phrase}"
    if s == "PAREN":
        return f"{phrase} (negated)"
    return f"NOT({phrase})"


class SegExtTopK(object):
    def __init__(
        self,
        model_name,
        dataset_name,
        top_k,
        deduplicate=True,
        logits_threshold=-1,
        use_rank=False,
        customized_prompt=None,
    ):
        self.prompt = customized_prompt or ZS_KEYWORD_PROMPT_STR[model_name][dataset_name]
        self.top_k = top_k
        self.deduplicate = deduplicate
        self.logits_threshold = logits_threshold
        self.use_rank = use_rank

    def __call__(self, example):
        if self.use_rank:
            selected_keywords = sorted(example["trunc_input_phrases"], key=lambda x: x["rank"])
            for i in range(len(selected_keywords)):
                selected_keywords[i]["score"] = i
        else:
            selected_keywords = []
            for kw_info in sorted(example["input_kw_model"], key=lambda x: x["score"], reverse=True):
                if kw_info["score"] < self.logits_threshold:
                    break
                selected_keywords.append(example["trunc_input_phrases"][kw_info["kw_index"]])
                selected_keywords[-1]["score"] = kw_info["score"]

        if self.deduplicate:
            selected_keywords = remove_duplicate_top_k(selected_keywords, top_k=self.top_k)
        else:
            selected_keywords = selected_keywords[: self.top_k]

        formatted_keywords = "; ".join([item["phrase"] for item in selected_keywords]) + "."
        return self.prompt.replace("<text>", example["trunc_input"]).replace("<keywords>", formatted_keywords)


class SegExtTopKNegationAware(object):
    """
    Same selection as SegExtTopK, but marks selected phrases that are negated in the source.
    Returns (prompt, other_info) so we can log negated phrase count per example.
    """
    def __init__(
        self,
        model_name,
        dataset_name,
        top_k,
        deduplicate=True,
        logits_threshold=-1,
        use_rank=False,
        neg_style="NOT",
        customized_prompt=None,
    ):
        self.prompt = customized_prompt or ZS_KEYWORD_PROMPT_STR[model_name][dataset_name]
        self.top_k = top_k
        self.deduplicate = deduplicate
        self.logits_threshold = logits_threshold
        self.use_rank = use_rank
        self.neg_style = neg_style

    def __call__(self, example):
        # ---- select keywords exactly like the original SegExtTopK ----
        if self.use_rank:
            selected_keywords = sorted(example["trunc_input_phrases"], key=lambda x: x["rank"])
            for i in range(len(selected_keywords)):
                selected_keywords[i]["score"] = i
        else:
            selected_keywords = []
            for kw_info in sorted(example["input_kw_model"], key=lambda x: x["score"], reverse=True):
                if kw_info["score"] < self.logits_threshold:
                    break
                selected_keywords.append(example["trunc_input_phrases"][kw_info["kw_index"]])
                selected_keywords[-1]["score"] = kw_info["score"]

        if self.deduplicate:
            selected_keywords = remove_duplicate_top_k(selected_keywords, top_k=self.top_k)
        else:
            selected_keywords = selected_keywords[: self.top_k]

        # ---- detect negation and format ----
        text = example["trunc_input"]
        formatted = []
        neg_cc = 0

        for item in selected_keywords:
            # trunc_input_phrases entries usually have {"phrase":..., "index":...}
            idx = item.get("index", None)
            neg = False
            if idx is not None:
                try:
                    neg = is_negated_in_context(text, int(idx), item.get("phrase", ""))
                except Exception:
                    neg = False

            if neg:
                neg_cc += 1
            formatted.append(format_phrase_with_negation(item["phrase"], negated=neg, style=self.neg_style))

        formatted_keywords = "; ".join(formatted) + "."
        prompt = self.prompt.replace("<text>", text).replace("<keywords>", formatted_keywords)

        other_info = {
            "negated_kw_count": neg_cc,
            "neg_style": self.neg_style,
        }
        return prompt, other_info


def get_prompt_fn(model_name, dataset, kw_strategy, kw_model_top_k, logits_threshold, neg_style="NOT"):
    if kw_strategy == "disable":
        return NaivePrompt(model_name, dataset)
    elif kw_strategy == "sigext_topk":
        return SegExtTopK(model_name, dataset, top_k=kw_model_top_k, logits_threshold=logits_threshold)
    elif kw_strategy == "sigext_topk_neg":
        return SegExtTopKNegationAware(
            model_name,
            dataset,
            top_k=kw_model_top_k,
            logits_threshold=logits_threshold,
            neg_style=neg_style,
        )
    else:
        raise RuntimeError("unknown kw strategy.")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
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

    result = {}
    for metric_name, values in result_element.items():
        result[metric_name] = np.mean(values)

    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [len(word_tokenize(pred)) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result


def run_inference(
    model_name,
    kw_strategy,
    kw_model_top_k,
    dataset,
    dataset_dir,
    output_dir,
    inference_on_split="test",
    neg_style="NOT",
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
        model_name,
        dataset,
        kw_strategy,
        kw_model_top_k=kw_model_top_k,
        logits_threshold=logits_threshold,
        neg_style=neg_style,
    )
    assert not isinstance(prompting_fn, dict)
    dataset_dir = pathlib.Path(dataset_dir)

    dataset_filename = str(dataset_dir.joinpath(f"{inference_on_split}.jsonl"))
    with jsonlines.open(dataset_filename) as f:
        inference_data = list(f)

    with Pool(1) as p:
        all_prompt = list(tqdm.tqdm(p.imap(prompting_fn, inference_data), total=len(inference_data)))

    for i in range(len(inference_data)):
        if isinstance(all_prompt[i], str):
            inference_data[i]["prompt_input"] = all_prompt[i]
        else:
            inference_data[i]["prompt_input"] = all_prompt[i][0]
            inference_data[i]["other_info"] = all_prompt[i][1]

    with Pool(1) as p:
        all_res = list(tqdm.tqdm(p.imap(predict_one_eg_fn, inference_data), total=len(inference_data)))
    all_res = [item for item in all_res]

    output_path = str(pathlib.Path(output_dir).expanduser())
    os.makedirs(output_path, exist_ok=True)

    with jsonlines.open(output_path + f"/{inference_on_split}_dataset.jsonl", "w") as f:
        f.write_all(inference_data)

    with open(output_path + f"/{inference_on_split}_predictions.json", "w") as f:
        json.dump(all_res, f, indent=2)

    test_metrics = compute_rouge_score(inference_data, all_res)
    with open(str(pathlib.Path(output_dir).joinpath(f"{inference_on_split}_metrics.json")), "w") as f:
        json.dump(test_metrics, f, indent=2)


def main():
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("transformers.generation").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="mistral", choices=["claude", "mistral"], help="llm name")
    parser.add_argument(
        "--kw_strategy",
        choices=["disable", "sigext_topk", "sigext_topk_neg"],
        help="keyword strategy.",
        required=True,
    )
    parser.add_argument("--kw_model_top_k", default=20, type=int, help="keyword strategy.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["arxiv", "pubmed", "cnn", "samsum", "meetingbank"],
        help="Select from supported datasets.",
    )
    parser.add_argument("--dataset_dir", required=True, type=str, help="directory of train and validation data.")
    parser.add_argument("--output_dir", required=True, type=str, help="directory to save experiment.")
    parser.add_argument("--inference_on_split", default="test", type=str, help="split_to_run_inference")
    parser.add_argument(
        "--neg_style",
        default="NOT",
        choices=["NOT", "TAG", "PAREN"],
        help="How to mark negated keyphrases in <keywords> (only used for sigext_topk_neg).",
    )

    args = parser.parse_args()

    run_inference(
        model_name=args.model_name,
        kw_strategy=args.kw_strategy,
        kw_model_top_k=args.kw_model_top_k,
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        inference_on_split=args.inference_on_split,
        neg_style=args.neg_style,
    )


if __name__ == "__main__":
    main()
