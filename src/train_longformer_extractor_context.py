import argparse
import logging
import pathlib

import jsonlines
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import tqdm
import transformers
from rouge_score import rouge_scorer
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer


class KWDatasetContext(Dataset):
    def __init__(
        self,
        dataset_filename,
        base_model,
        example_kw_hit_threshold=3,
        base_model_max_length=4096,
        hide_gt=False,
    ):
        super().__init__()
        self.data = []
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

        if hide_gt:
            assert example_kw_hit_threshold == 0

        with jsonlines.open(dataset_filename) as f:
            self.raw_dataset = list(f)

        pos_cc = 0
        neg_cc = 0

        for idx, item in tqdm.tqdm(
            enumerate(self.raw_dataset),
            total=len(self.raw_dataset),
            desc="process data",
        ):
            # Start with <s> and ignore label
            x = [0]
            y = [-100]
            example_kw_hit_cc = 0

            selected_keyword_strs = set()
            if not hide_gt:
                for kw in item["trunc_input_phrases"]:
                    best_rouge_f = 0
                    best_rouge_p = 0
                    best_rouge_r = 0
                    for ent in item["trunc_output_phrases"]:
                        score = scorer.score(target=ent["phrase"], prediction=kw["phrase"])
                        best_rouge_f = max(best_rouge_f, score["rouge1"].fmeasure)
                        best_rouge_p = max(best_rouge_p, score["rouge1"].precision)
                        best_rouge_r = max(best_rouge_r, score["rouge1"].recall)
                    if best_rouge_f >= 0.6 or best_rouge_p >= 0.8 or best_rouge_r >= 0.8:
                        selected_keyword_strs.add(kw["phrase"])

            current_text_index = 0
            for kw_info in item["trunc_input_phrases"]:
                if current_text_index < kw_info["index"]:
                    content = item["trunc_input"][current_text_index: kw_info["index"]]
                    content_tokens = self.tokenizer(content, add_special_tokens=False)["input_ids"]
                    if len(content_tokens) > 0:
                        x.extend(content_tokens)
                        y.extend([-100] * len(content_tokens))
                else:
                    # handle overlaps by adding a space
                    if current_text_index != 0:
                        space_tokens = self.tokenizer(" ", add_special_tokens=False)["input_ids"]
                        if len(space_tokens) > 0:
                            x.extend(space_tokens)
                            y.extend([-100] * len(space_tokens))

                format_kw = f"{kw_info['phrase']}"
                input_ids = self.tokenizer(format_kw, add_special_tokens=False)["input_ids"]

                if not hide_gt and kw_info["phrase"] in selected_keyword_strs:
                    labels = [1] * len(input_ids)
                else:
                    labels = [0] * len(input_ids)

                if labels and labels[-1] == 1:
                    example_kw_hit_cc += 1

                x.extend(input_ids)
                y.extend(labels)

                current_text_index = kw_info["index"] + len(kw_info["phrase"])

            if current_text_index < len(item["trunc_input"]):
                content = item["trunc_input"][current_text_index:]
                content_tokens = self.tokenizer(content, add_special_tokens=False)["input_ids"]
                if len(content_tokens) > 0:
                    x.extend(content_tokens)
                    y.extend([-100] * len(content_tokens))

            # Truncate and add </s> (Longformer RoBERTa-style uses 2 for </s>)
            x = x[: base_model_max_length - 1]
            y = y[: base_model_max_length - 1]
            x.append(2)
            y.append(-100)

            x = torch.tensor(x, dtype=torch.long)
            y = torch.tensor(y, dtype=torch.long)

            if example_kw_hit_cc >= example_kw_hit_threshold:
                self.data.append((x, y, idx))
                pos_cc += int((y == 1).sum().item())
                neg_cc += int((y == 0).sum().item())

        denom = (pos_cc + neg_cc) if (pos_cc + neg_cc) > 0 else 1
        logging.info(f"keyword ratio {pos_cc / denom:.6f}")
        logging.info(f"Dataset size: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    # batch size is 1 in your code, but keep it robust
    xs, ys, ids = zip(*batch)
    # pad to max length in batch (still 1)
    x = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=1)   # RoBERTa pad id = 1
    y = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-100)
    idx = torch.tensor(ids, dtype=torch.long)
    return x, y, idx


class KeywordExtractorClf(pl.LightningModule):
    def __init__(self, base_model):
        super().__init__()
        self.save_hyperparameters()
        self.clf = AutoModelForTokenClassification.from_pretrained(base_model, num_labels=2)
        self.validation_step_outputs = []
        # class weights: [neg, pos]
        self.register_buffer("ce_weight", torch.tensor([0.1, 1.0], dtype=torch.float32))

    def predict_step(self, batch, batch_idx):
        x, y, idx = batch
        logits = self.clf(x).logits  # (B, T, 2)

        # Guard: if anything becomes non-finite, replace with zeros to avoid crashing
        if not torch.isfinite(logits).all():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

        logp = F.log_softmax(logits, dim=-1)

        # phrase-level scoring
        kw_count = 0
        score_and_label = []
        current_logits = 0.0
        current_len = 0
        current_in_phrase = False

        for i in range(x.size(1)):
            if y[0][i] != -100:
                current_in_phrase = True
                current_logits += float(logp[0][i][1].item())
                current_len += 1
            else:
                if current_in_phrase and current_len > 0:
                    score_and_label.append({"kw_index": kw_count, "score": float(current_logits / current_len)})
                    kw_count += 1
                current_logits = 0.0
                current_len = 0
                current_in_phrase = False

        return {"id": int(idx[0].item()), "score": score_and_label}

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self.clf(x).logits  # (B, T, 2)

        # Compute CE only on active tokens (y != -100), and mean it (stable!)
        mask = (y != -100)
        if mask.sum() == 0:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            loss = F.cross_entropy(
                logits[mask],
                y[mask],
                reduction="mean",
                weight=self.ce_weight.to(self.device),
            )

        # Guard against NaN loss
        if not torch.isfinite(loss):
            self.log("train/nonfinite_loss", 1.0)
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/logits_std", logits[..., 1].std(), prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self.clf(x).logits

        mask = (y != -100)
        if mask.sum() == 0:
            loss = 0.0
            logp = F.log_softmax(logits, dim=-1)
        else:
            loss_t = F.cross_entropy(
                logits[mask],
                y[mask],
                reduction="mean",
                weight=self.ce_weight.to(self.device),
            )
            loss = float(loss_t.detach().cpu().item())
            logp = F.log_softmax(logits, dim=-1)

        if not torch.isfinite(logp).all():
            logp = torch.nan_to_num(logp, nan=0.0, posinf=0.0, neginf=0.0)

        logp_np = logp.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        score_and_label = []
        current_logits = 0.0
        current_len = 0
        current_in_phrase = False

        for i in range(x.size(1)):
            if y_np[0][i] != -100:
                current_in_phrase = True
                current_logits += float(logp_np[0][i][1])
                current_len += 1
            else:
                if current_in_phrase and current_len > 0:
                    score_and_label.append(
                        {"index": i, "logits": float(current_logits / current_len), "label": int(y_np[0][i - 1])}
                    )
                current_logits = 0.0
                current_len = 0
                current_in_phrase = False

        if len(score_and_label) == 0:
            # avoid NaNs in std/metrics
            step_output = {"loss": loss, "logits_std": 0.0, "logits_std_all": float(np.std(logp_np[0, :, 1]))}
            for top_k in [5, 10, 20]:
                step_output[f"precision_{top_k}"] = 0.0
                step_output[f"recall_{top_k}"] = 0.0
            self.validation_step_outputs.append(step_output)
            return

        score_and_label = sorted(score_and_label, key=lambda item: (item["logits"], -item["index"]), reverse=True)

        all_labels = np.array([item["label"] for item in score_and_label], dtype=np.float32)
        denom = float(all_labels.sum()) if float(all_labels.sum()) > 0 else 1.0

        step_output = {
            "loss": loss,
            "logits_std": float(np.std([item["logits"] for item in score_and_label])),
            "logits_std_all": float(np.std(logp_np[0, :, 1])),
        }

        for top_k in [5, 10, 20]:
            top = score_and_label[:top_k]
            step_output[f"precision_{top_k}"] = float(np.mean([item["label"] for item in top]))
            step_output[f"recall_{top_k}"] = float(np.sum([item["label"] for item in top]) / denom)

        self.validation_step_outputs.append(step_output)

    def on_validation_epoch_end(self):
        if len(self.validation_step_outputs) == 0:
            return

        for top_k in [5, 10, 20]:
            self.log(
                f"val/precision_{top_k}",
                float(np.mean([item[f"precision_{top_k}"] for item in self.validation_step_outputs])),
                sync_dist=True,
                prog_bar=True if top_k == 20 else False,
            )
            self.log(
                f"val/recall_{top_k}",
                float(np.mean([item[f"recall_{top_k}"] for item in self.validation_step_outputs])),
                sync_dist=True,
                prog_bar=True if top_k == 20 else False,
            )

        self.log("val/loss", float(np.mean([item["loss"] for item in self.validation_step_outputs])), sync_dist=True)
        self.log("val/logits_std", float(np.mean([item["logits_std"] for item in self.validation_step_outputs])), sync_dist=True)
        self.log("val/logits_std_all", float(np.mean([item["logits_std_all"] for item in self.validation_step_outputs])), sync_dist=True)

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_training_steps=self.trainer.estimated_stepping_batches,
            num_warmup_steps=100,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser("Train longformer keyword extractor.")
    parser.add_argument("--dataset_dir", required=True, type=str, help="directory of train and validation data.")
    parser.add_argument("--checkpoint_dir", required=True, type=str, help="directory to save checkpoints.")
    parser.add_argument("--seed", default=42, type=int, help="random seed for trainer.")
    parser.add_argument("--base_model", default="allenai/longformer-base-4096", type=str, help="Backbone pre-trained model.")
    parser.add_argument("--load_ckpt", default=None, type=str, help="Pretrained ckpt.")
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    dataset_dir = pathlib.Path(args.dataset_dir).expanduser()
    train_set = KWDatasetContext(
        dataset_filename=str(dataset_dir.joinpath("train.jsonl")),
        base_model=args.base_model,
        example_kw_hit_threshold=1,
    )
    val_set = KWDatasetContext(
        dataset_filename=str(dataset_dir.joinpath("validation.jsonl")),
        base_model=args.base_model,
        example_kw_hit_threshold=3,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    if args.load_ckpt:
        model = KeywordExtractorClf.load_from_checkpoint(args.load_ckpt, base_model=args.base_model)
    else:
        model = KeywordExtractorClf(base_model=args.base_model)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=2,
        monitor="val/recall_20",
        mode="max",
        every_n_epochs=1,
        filename="epoch_{epoch:02d}-step_{step:06d}-recall20_{val/recall_20:.3f}",
        auto_insert_metric_name=False,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=32,  # ✅ FP32 to avoid NaNs on longformer-large
        max_epochs=10,
        callbacks=[checkpoint_callback],
        default_root_dir=args.checkpoint_dir,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        enable_checkpointing=True,
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
