# two_stage_summarization.py
import argparse, json, os, pathlib
import jsonlines
import nltk
import numpy as np
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize

from bedrock_utils import predict_one_eg_mistral, predict_one_eg_claude_instant

def postprocess_text(preds, labels):
    preds = [p.strip() for p in preds]
    labels = [l.strip() for l in labels]
    preds = ["\n".join(nltk.sent_tokenize(p)) for p in preds]
    labels = ["\n".join(nltk.sent_tokenize(l)) for l in labels]
    return preds, labels

def compute_rouge(inference_data, preds):
    scorer = rouge_scorer.RougeScorer(["rouge1","rougeL","rougeLsum"], use_stemmer=True)
    labels = [x["raw_output"] for x in inference_data]
    decoded_preds, decoded_labels = postprocess_text(preds, labels)
    res = {"rouge1f":[], "rougeLf":[], "rouge1r":[], "rougeLr":[]}
    for p,l in zip(decoded_preds, decoded_labels):
        s = scorer.score(target=l, prediction=p)
        res["rouge1f"].append(s["rouge1"].fmeasure)
        res["rougeLf"].append(s["rougeL"].fmeasure)
        res["rouge1r"].append(s["rouge1"].recall)
        res["rougeLr"].append(s["rougeL"].recall)
    out = {
        "rouge1f": round(float(np.mean(res["rouge1f"]))*100, 4),
        "rougeLf": round(float(np.mean(res["rougeLf"]))*100, 4),
        "rouge1r": round(float(np.mean(res["rouge1r"]))*100, 4),
        "gen_len": float(np.mean([len(word_tokenize(p)) for p in preds])),
    }
    return out

def build_prompts(model_name, dataset_name, text, max_sents):
    # Pass 1: extraction
    p1 = f"""You are helping with summarization.
From the following document, extract the {max_sents} most important sentences (verbatim).
Return ONLY the extracted sentences, one per line.

DOCUMENT:
{text}
"""
    # Pass 2: abstraction
    p2 = """Rewrite the extracted sentences into a coherent abstractive summary.
Keep key facts, remove redundancy, and keep it concise.
Return ONLY the summary.
"""
    return p1, p2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", choices=["mistral","claude"], required=True)
    parser.add_argument("--dataset", choices=["cnn","samsum","arxiv","meetingbank","pubmed"], required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--inference_on_split", default="test")
    parser.add_argument("--max_sents", type=int, default=12)
    args = parser.parse_args()

    if args.model_name == "mistral":
        predict = predict_one_eg_mistral
    else:
        predict = predict_one_eg_claude_instant

    dataset_path = pathlib.Path(args.dataset_dir).joinpath(f"{args.inference_on_split}.jsonl")
    with jsonlines.open(str(dataset_path)) as f:
        data = list(f)

    preds = []
    cache = []

    for ex in data:
        p1, p2 = build_prompts(args.model_name, args.dataset, ex["trunc_input"], args.max_sents)

        ex1 = {"prompt_input": p1}
        extracted = predict(ex1).strip()

        ex2 = {"prompt_input": p2 + "\n\nEXTRACTED SENTENCES:\n" + extracted}
        summary = predict(ex2).strip()

        preds.append(summary)
        cache.append({"extracted": extracted, "summary": summary})

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"{args.inference_on_split}_predictions.json", "w") as f:
        json.dump(preds, f, indent=2)

    with open(out_dir / f"{args.inference_on_split}_two_stage_cache.json", "w") as f:
        json.dump(cache, f, indent=2)

    metrics = compute_rouge(data, preds)
    with open(out_dir / f"{args.inference_on_split}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
