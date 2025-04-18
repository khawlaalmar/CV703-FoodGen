# Food Image Captioning Pipeline

This project implements a pipeline for generating, refining, and evaluating captions for food images. It leverages vision-language models including InstructBLIP, CLIP, and LLaMA to create high-quality, label-aligned image descriptions. The primary goal is to improve caption quality through refinement and fine-tuning, and assess the impact using vision-language alignment metrics.

---

## Pipeline Overview

### Step 1: Caption Generation
- **Model**: `Salesforce/instructblip-flan-t5-xl`
- **Objective**: Generate initial image captions with minimal context and no assumptions.
- **Output**: `synced_captions.json`
- **Script**: `scripts/baseline.py`

### Step 2: CLIP Scoring + SLA Evaluation
- **Model**: CLIP (`ViT-L/14`)
- **Metrics**:
  - **CLIP Score**: Cosine similarity between image and caption embeddings.
  - **SLA Rank**: Rank of the matching caption among all captions in the batch (lower is better).
- **Output**: JSON files with per-caption scores (e.g., `*_scores_with_sla.json`).
- **Script**: `scripts/clip_score_sla.py`

### Step 3: Caption Refinement
- **Refinement Strategy**: Select low-performing captions based on CLIP/SLA, and refine them using a large language model.
- **Model**: `meta-llama/Llama-2-7b-chat-hf` or `gpt-4` via API (optional).
- **Output**: `refined_captions.json`, `llama_refined_captions.json`
- **Script**: `scripts/refining_captions.py`

### Step 4: Evaluation (Post-Refinement)
- Re-evaluate refined captions using CLIP and SLA metrics.
- Compare scores with original captions to assess improvements.
- **Optional**: Evaluate using BLIP2 ITM score (image-text matching probability).
- **Script**: Reuse `clip_score_sla.py`, optionally `blip_score.py`

### Step 5: Fine-tuning InstructBLIP
- **Approach**: LoRA (Low-Rank Adaptation) for efficient parameter-efficient fine-tuning.
- **Inputs**: Refined captions + corresponding images.
- **Frozen Parameters**: All layers except `q_proj`, `v_proj` projections.
- **Output**: Fine-tuned model saved to `instructblip_finetuned/`
- **Script**: `scripts/finetune.py`

### Step 6: Final Evaluation
- Re-generate captions with the fine-tuned model.
- Score them using CLIP and compare with original and refined results.
- **Goal**: Determine if fine-tuning improves alignment and descriptive quality.

---

## 🧪 Evaluation Metrics

| Metric        | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| CLIP Score    | Cosine similarity between image and caption embeddings (higher is better). |
| SLA Rank      | Rank of true image-caption pair in similarity matrix (lower is better).    |
| BLIP2 ITM     | Image-text match score using BLIP2’s classification head (0.0 to 1.0).     |

---

## 📁 Key Files

| File/Directory                         | Purpose                                      |
|----------------------------------------|----------------------------------------------|
| `scripts/baseline.py`                 | Generate baseline captions                   |
| `scripts/refining_captions.py`       | Refine low-score captions with LLM           |
| `scripts/clip_score_sla.py`          | CLIP scoring and SLA evaluation              |
| `scripts/finetune.py`                | Fine-tune InstructBLIP using LoRA            |
| `scripts/blip_score.py`              | (Optional) BLIP2 ITM score computation       |
| `refined_captions.json`              | Refined caption dataset                      |
| `caption_scores/`                    | Folder containing caption evaluation results |
| `synced_fixed_image_list.txt`        | Filenames of selected subset of images       |

---

## 🧠 Model Summary

| Component         | Model Name                                 |
|------------------|---------------------------------------------|
| Caption Generator| `Salesforce/instructblip-flan-t5-xl`        |
| Caption Refiner  | `meta-llama/Llama-2-7b-chat-hf` or `GPT-4`  |
| Scoring Model    | `openai/clip-vit-large-patch14`             |
| ITM Scoring      | `BLIP2` from LAVIS                          |

---

## 🔄 Future Work

- Automate evaluation report generation with visualizations
- Expand to multilingual food captioning
- Integrate retrieval metrics (Recall@K) for stronger alignment checks

---

## 📌 Notes

- All scripts are modular and can be reused across variants (e.g., original, refined, and fine-tuned).
- For large-scale scoring (20k images), precompute embeddings and avoid real-time scoring in loops.
- Store checkpoints regularly during scoring or training for robustness.

---

## 🧪 Environment Requirements

- Python 3.9+
- PyTorch with CUDA support
- `transformers`, `peft`, `clip-by-openai`, `tqdm`, `Pillow`, `LAVIS`
- Optional: Hugging Face login for gated models (e.g., LLaMA)

---

## ✍️ Author

This project is developed as part of a vision-language captioning evaluation task. For assistance, suggestions, or collaboration, contact the maintainer.

