# SigExt Reproduction + Research Extensions  
## Controllable & Faithful Prompt-based Summarization

This repository reproduces and extends the EMNLP 2024 Industry paper:

> **Salient Information Prompting to Steer Content in Prompt-based Abstractive Summarization**  
> Lei Xu, Asad Karim, Saket Dingliwal, Aparna Elangovan  
> EMNLP Industry Track, 2024

---

## 🔎 Project Scope

Large Language Models (LLMs) demonstrate strong zero-shot summarization performance. However, they still lack:

- Explicit content controllability  
- Robust faithfulness guarantees  
- Transparent content selection mechanisms  

SigExt addresses controllability by extracting salient keyphrases from the source document and injecting them into the prompt.

This repository:

- Reproduces the original SigExt pipeline  
- Evaluates across four benchmark datasets  
- Introduces two research extensions for robustness and faithfulness analysis  

---

## 📚 Datasets

Experiments are conducted on:

- SAMSum  
- CNN/DailyMail  
- ArXiv  
- MeetingBank  

---

# 🧠 Research Extensions

## 1️⃣ Negation-aware Faithfulness Analysis

### Motivation

We observed that when a selected keyphrase is negated in the source document, the LLM may generate a positive or neutral statement — introducing a polarity flip.

This issue is not captured by ROUGE or standard lexical overlap metrics.

### 🔁 Negation Flip

A flip occurs when:

- A selected keyphrase is negated in the source document  
- The generated summary mentions it without negation  

### ▶ Run Negation Analysis
```bash
python3 src/analyze_negation_extension.py \
  --run_dir experiments/cnn_sigext_mistral_k15 \
  --scope strict
```


### 📊 Metrics Reported

- Exposure rate  
- Flip@Example  
- Flips per 100 examples  
- Flips per negated keyword  

This extension provides a faithfulness diagnostic beyond lexical overlap and exposes subtle polarity errors.

---

## 2️⃣ Adaptive Keyphrase Budget (Adaptive-K)

### Motivation

The original SigExt uses a fixed **K = 15** for all documents.

However, document lengths vary significantly across datasets. A fixed budget may:

- Under-select content in long transcripts  
- Over-inject phrases in short news articles  

We introduce dynamic phrase selection strategies.

---

### 🔹 Length-Based Adaptive-K

```bash
python3 src/zs_summarization_adaptive.py \
  --model_name mistral \
  --kw_strategy sigext_adaptive \
  --adaptive_mode len \
  --dataset meetingbank \
  --dataset_dir experiments/meetingbank_dataset_with_keyphrase/ \
  --output_dir experiments/meetingbank_sigext_mistral_adaptive_len/
```
### 🔹 Score-Mass Adaptive-K
```bash
python3 src/zs_summarization_adaptive.py \
  --model_name mistral \
  --kw_strategy sigext_adaptive \
  --adaptive_mode mass \
  --k_tau 0.85 \
  --k_min 10 --k_max 20 \
  --dataset meetingbank \
  --dataset_dir experiments/meetingbank_dataset_with_keyphrase/ \
  --output_dir experiments/meetingbank_sigext_mistral_adaptive_mass_tau085/
```

### ▶ Analyze Adaptive Behavior
```bash
python3 src/analyze_adaptive_k.py \
  --run_dir experiments/meetingbank_sigext_mistral_adaptive_len \
  --split test
```
### 📈 Outputs

- Average K  
- Min–Max statistics  
- Distribution analysis

# 🔁 Reproducing the Baseline (Example: CNN)

## 1️⃣ Prepare Dataset

```bash
python3 src/prepare_data.py \
  --dataset cnn \
  --output_dir experiments/cnn_dataset/
```
## 2️⃣ Train Longformer Keyphrase Extractor

```bash
python3 src/train_longformer_extractor_context.py \
  --dataset_dir experiments/cnn_dataset/ \
  --checkpoint_dir experiments/cnn_extractor_model/
```
## 3️⃣ Inference Keyphrase Extractor

```bash
python3 src/inference_longformer_extractor.py \
  --dataset_dir experiments/cnn_dataset/ \
  --checkpoint_dir experiments/cnn_extractor_model/ \
  --output_dir experiments/cnn_dataset_with_keyphrase/
```
## 4️⃣ Run SigExt Summarization (Fixed-K)

### Claude

```bash
python3 src/zs_summarization.py \
  --model_name claude \
  --kw_strategy sigext_topk \
  --kw_model_top_k 15 \
  --dataset cnn \
  --dataset_dir experiments/cnn_dataset_with_keyphrase/ \
  --output_dir experiments/cnn_sigext_claude_k15/
```

### Mistral

```bash
python3 src/zs_summarization.py \
  --model_name mistral \
  --kw_strategy sigext_topk \
  --kw_model_top_k 15 \
  --dataset cnn \
  --dataset_dir experiments/cnn_dataset_with_keyphrase/ \
  --output_dir experiments/cnn_sigext_mistral_k15/
```

# 🔐 Environment Setup

Create a `.env` file in the project root directory based on `.env.example`.

### Example
```ini
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-west-2
```
The `.env` file is excluded from version control for security reasons.

# ⚙ Hardware & Runtime

- AWS GPU instances for extractor training  
- API-based LLM inference  
- Each test run (~500 examples): ~1.5 hours  

---

# 📌 Reproducibility Notes

Results may slightly differ from the original paper due to:

- Retired LLM API versions  
- Unavailable random seeds  
- Hardware and API variance    

However, qualitative trends remain consistent:

- SigExt outperforms naive prompting  
- Adaptive-K improves robustness  
- Negation-aware analysis reveals measurable polarity errors  
