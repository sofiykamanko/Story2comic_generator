# Story-to-Comic Strip Generator

Turn any short text story into a 4-panel comic strip using fine-tuned Stable Diffusion + Adaptive Consistent Self-Attention + BubbleCleaner.


---

## Overview

The pipeline takes a 3–6 sentence story and produces a full comic strip with readable dialogue in speech bubbles. It combines four components:

1. **LLM (Claude)** — parses the story into 4 scenes and generates dialogues by looking at the actual generated images
2. **SD 1.5 + LoRA** — generates comic-style panel images
3. **Adaptive Consistent Self-Attention** — keeps character appearance consistent across panels
4. **BubbleCleaner** — replaces illegible SD-generated scribbles with readable text

---

## Installation

```bash
# Google Colab (T4 GPU recommended)
!pip install git+https://github.com/facebookresearch/sam2.git -q
!pip install simple-lama-inpainting easyocr ultralytics diffusers transformers -q
!wget -q "https://github.com/google/fonts/raw/main/ofl/bangers/Bangers-Regular.ttf" -O /content/Bangers.ttf
```

Set your Anthropic API key:
```python
import os
os.environ["ANTHROPIC_API_KEY"] = "your-key-here"
```

---

## File Structure

```
├── pipeline.py           # Main entry point — text → comic strip
├── llm.py                # Claude: story parsing + dialogue generation
├── image_gen.py          # SD 1.5 pipeline + LoRA loading
├── consistent_attention.py  # Adaptive Consistent Self-Attention
├── bubble_cleaner.py     # YOLO + LaMa + PIL bubble replacement
├── bubbles.py            # Caption boxes + comic strip assembly
└── app.py                # Gradio UI
```

---

## Usage

### Basic

```python
from image_gen import load_pipeline
from pipeline import run_pipeline

pipe = load_pipeline(
    model_id="runwayml/stable-diffusion-v1-5",
    lora_path="/content/pytorch_lora_weights.safetensors",
    device="cuda"
)

comic, result, prompts, images = run_pipeline(
    story="A brave knight discovers a dragon guarding a mysterious cave. "
          "The knight challenges the dragon to a battle. "
          "The dragon reveals it is protecting ancient books. "
          "The knight and dragon become friends and open a library together.",
    pipe=pipe,
    mode="adaptive",   # "adaptive" | "standard" | "baseline"
    decay=0.5,
    seed=42,
    save_path="/content/comic.png",
)
comic
```

### Compare All Modes

```python
from pipeline import compare_modes

comparison = compare_modes(
    story="your story here",
    pipe=pipe,
    save_path="/content/comparison.png",
)
comparison
```

### Character Overrides

```python
comic, *_ = run_pipeline(
    story="...",
    pipe=pipe,
    character_overrides={
        "sarah": "woman in red coat, dark hair",
        "ghost": "translucent glowing figure",
    },
)
```

---

## Attention Modes

| Mode | Description | Best for |
|------|-------------|----------|
| `adaptive` | Our contribution — Adaptive Consistent SA on the middle UNet block only, with decay-weighted neighbor influence | Best quality + consistency balance |
| `standard` | StoryDiffusion SA on all 16 UNet blocks | Higher consistency, lower image quality |
| `baseline` | Standard SD, no modification | Best image quality, no character consistency |

### Adaptive SA — How It Works

Instead of treating all panels equally, neighboring panels influence each other more than distant ones:

```
out_i = Σ_j  w[i,j] × Attn(Q_i, K_j, V_j)
where  w[i,j] = decay^|i-j|,  normalized so Σ_j w[i,j] = 1
```

Weight matrix at `decay=0.5`:
```
Panel 0: [0.53, 0.27, 0.13, 0.07]
Panel 1: [0.27, 0.53, 0.27, 0.13]
Panel 2: [0.13, 0.27, 0.53, 0.27]
Panel 3: [0.07, 0.13, 0.27, 0.53]
```

Applied only to the middle UNet block — best quality/consistency tradeoff.

---

## BubbleCleaner Pipeline

SD 1.5 cannot generate readable text — it treats text as texture. BubbleCleaner solves this in 5 steps:

1. **Fine-tuned YOLO** (`yolo_model.pt`) — detects speech bubble locations (mAP50=0.97)
2. **kitsumed segmentation YOLO** — refines mask shape inside each detected bbox
3. **SAM2 fallback** — used when YOLO finds nothing; selects best candidate by brightness + OCR signal
4. **LaMa inpainting** — reconstructs background under the scribbles
5. **PIL rendering** — fills bubble with original color, renders Bangers font with adaptive size + word wrap

When multiple bubbles are detected, Claude Vision assigns the correct dialogue to each bubble based on character positions.

### YOLO Fine-tuning

The detection model was fine-tuned on SD-generated panels to close the domain gap with real comics:

- Generated 200 panels from our model (50 different stories)
- Labeled on Roboflow using Auto Label with SAM3 (94% confidence)
- Fine-tuned `ogkalu/yolov8m_seg-speech-bubble`: 50 epochs, imgsz=640
- Result: **mAP50=0.97**, Precision=0.981, Recall=0.947

---

## LoRA Training

Fine-tuned SD 1.5 on [VLR-CVC/ComicsPAP](https://huggingface.co/datasets/VLR-CVC/ComicsPAP) (2502 American comic images).

| Parameter | Value |
|-----------|-------|
| Steps | 2000 |
| Batch size | 8 |
| Rank | 4 |
| Precision | fp16 |
| Trigger word | `comicstyle` |
| Loss | 0.2012 → 0.1859 (−7.6%) |
| Trained params | ~0.1% of total |

---

## Gradio UI

```bash
python app.py
```

Or in Colab:
```python
!python app.py
```

---

## Requirements

- Python 3.10+
- CUDA GPU (T4 or better recommended)
- Anthropic API key
- `pytorch_lora_weights.safetensors` — LoRA weights
- `yolo_model.pt` — fine-tuned bubble detection model
- `Bangers.ttf` — comic font
