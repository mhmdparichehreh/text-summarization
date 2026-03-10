#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import glob
import logging
import pathlib

import jsonlines
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from train_longformer_extractor_context import KWDatasetContext, KeywordExtractorClf


def _safe_tensor(t):
    import torch

    if not isinstance(t, torch.Tensor):
        return t
    t = t.float()
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return t


def find_best_checkpoint(checkpoint_dir: str) -> str:
    candidates = sorted(glob.glob(f"{checkpoint_dir}/lightning_logs/*/checkpoints/*.ckpt"))
    if not candidates:
        raise RuntimeError(f"Candidates not found under: {checkpoint_dir}/lightning_logs/*/checkpoints/*.ckpt")

    best_score = float("-inf")
    best_checkpoint = None

    for item in candidates:
        stem = pathlib.Path(item).stem
        tokens = stem.split("-")

        info = {}
        ok = True
        for tok in tokens:
            if "_" not in tok:
                continue
            key, value = tok.split("_", 1)
            info[key] = value

        if "recall20" not in info:
            # Skip checkpoints that don't encode recall20 in filename
            continue

        try:
            score = float(info["recall20"])
        except ValueError:
            ok = False

        if not ok:
            continue

        if score >= best_score:
            best_score = score
            best_checkpoint = item

    if best_checkpoint is None:
        # fallback: just take the latest checkpoint by sort order
        best_checkpoint = candidates[-1]
        logging.warning(
            "No checkpoint with 'recall20_...' found in filename. Falling back to last checkpoint: %s",
            best_checkpoint,
        )
    return best_checkpoint


def parse_result(raw_dataset, predicts):
    raw_dataset = copy.deepcopy(raw_dataset)
    for pred_info in predicts:
        example_info = raw_dataset[pred_info["id"]]
        score = pred_info["score"]

        if len(score) > len(example_info["trunc_input_phrases"]):
            raise RuntimeError("Model prediction does not match number of phrases.")

        # Ensure tensor outputs are safe for serialization if needed
        score = _safe_tensor(score)
        example_info["input_kw_model"] = score

    return raw_dataset


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser("Run longformer inference")
    parser.add_argument("--dataset_dir", required=True, type=str, help="Directory containing validation.jsonl/test.jsonl")
    parser.add_argument("--checkpoint_dir", required=True, type=str, help="Directory containing lightning_logs/*")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save predictions.")
    parser.add_argument(
        "--base_model",
        default="allenai/longformer-base-4096",
        type=str,
        help="Backbone pre-trained model (MUST match the one used for training).",
    )
    parser.add_argument(
        "--precision",
        default="16-mixed",
        type=str,
        help='PyTorch Lightning precision. Recommended: "16-mixed" for GPU inference.',
    )
    args = parser.parse_args()

    ckpt_dir = str(pathlib.Path(args.checkpoint_dir).expanduser())
    best_checkpoint = find_best_checkpoint(ckpt_dir)
    logging.info("Using %s", best_checkpoint)

    # ✅ IMPORTANT: load checkpoint with the SAME backbone used during training
    model = KeywordExtractorClf.load_from_checkpoint(best_checkpoint, base_model=args.base_model)

    trainer = pl.Trainer(devices=1, accelerator="gpu", precision=args.precision)

    dataset_dir = pathlib.Path(args.dataset_dir).expanduser()
    output_dir = pathlib.Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_class = KWDatasetContext

    for split in ["validation", "test"]:
        dataset_path = dataset_dir.joinpath(f"{split}.jsonl")
        if not dataset_path.exists():
            logging.warning("Split file not found, skipping: %s", str(dataset_path))
            continue

        dataset = dataset_class(
            dataset_filename=str(dataset_path),
            base_model=args.base_model,
            example_kw_hit_threshold=0,
            hide_gt=True,
        )
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        predicts = trainer.predict(model, dataloaders=dataloader)
        dataset_with_prediction = parse_result(dataset.raw_dataset, predicts)

        out_path = output_dir.joinpath(f"{split}.jsonl")
        with jsonlines.open(str(out_path), "w") as f:
            f.write_all(dataset_with_prediction)

        logging.info("Wrote predictions to %s", str(out_path))


if __name__ == "__main__":
    main()
