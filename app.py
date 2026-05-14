"""
Gradio UI for Comic Strip Generator.

Run in Colab:
    !python app.py
"""

import os
import gradio as gr
import torch
from image_gen import load_pipeline
from pipeline import run_pipeline, compare_modes

MODEL_ID  = "runwayml/stable-diffusion-v1-5"
LORA_PATH = os.environ.get("LORA_PATH", "pytorch_lora_weights.safetensors")
FONT_PATH = os.environ.get("FONT_PATH", "/content/Bangers.ttf")
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading pipeline...")
pipe = load_pipeline(MODEL_ID, lora_path=LORA_PATH, device=DEVICE)
print("Ready!")


def generate(story, mode, decay, seed, steps, cfg, char1_name, char1_desc, char2_name, char2_desc):
    if not story.strip():
        return None, "Please enter a story."
    try:
        overrides = {}
        if char1_name.strip() and char1_desc.strip():
            overrides[char1_name.strip()] = char1_desc.strip()
        if char2_name.strip() and char2_desc.strip():
            overrides[char2_name.strip()] = char2_desc.strip()

        comic, result, prompts, _ = run_pipeline(
            story=story,
            pipe=pipe,
            mode=mode.lower(),
            decay=float(decay),
            seed=int(seed),
            num_inference_steps=int(steps),
            guidance_scale=float(cfg),
            save_path="/tmp/comic_output.png",
            font_path=FONT_PATH,
            character_overrides=overrides or None,
        )

        info = f"**Characters:** {', '.join(c['name'] for c in result['characters'])}\n\n"
        for i, (panel, pn) in enumerate(zip(result["panels"], prompts)):
            p = pn[0] if isinstance(pn, tuple) else pn
            info += f"**Panel {i+1}:** {panel['scene'][:70]}\n"
            info += f"*Dialogue:* {panel.get('dialogue') or '—'} | *Caption:* {panel.get('caption') or '—'}\n\n"
        return comic, info

    except Exception as e:
        import traceback
        return None, f"Error: {e}\n{traceback.format_exc()}"


def ablation(story, seed, char1_name, char1_desc, char2_name, char2_desc):
    if not story.strip():
        return None, "Please enter a story."
    try:
        overrides = {}
        if char1_name.strip() and char1_desc.strip():
            overrides[char1_name.strip()] = char1_desc.strip()
        if char2_name.strip() and char2_desc.strip():
            overrides[char2_name.strip()] = char2_desc.strip()

        combined = compare_modes(
            story=story,
            pipe=pipe,
            save_path="/tmp/comparison.png",
            font_path=FONT_PATH,
            character_overrides=overrides or None,
        )
        return combined, "Top: Baseline | Middle: Standard SA | Bottom: Adaptive SA (ours)"
    except Exception as e:
        return None, f"Error: {e}"


EXAMPLE = (
    "Detective Sarah Kane receives a mysterious envelope with a map to a haunted mansion.\n"
    "She enters the dark mansion alone, her flashlight revealing strange shadows.\n"
    "A ghost appears, pointing at a hidden door behind the bookshelf.\n"
    "Sarah finds a room full of stolen treasure and laughs triumphantly."
)

with gr.Blocks(title="Comic Strip Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Comic Strip Generator")
    gr.Markdown(
        "Turn any short story into a 4-panel comic using "
        "**LoRA fine-tuning** + **Adaptive Consistent Self-Attention** + **BubbleCleaner**."
    )

    with gr.Row():
        with gr.Column(scale=1):
            story_input = gr.Textbox(label="Story", lines=6, value=EXAMPLE,
                                     placeholder="Write a short story (3–6 sentences)...")

            gr.Markdown("### Settings")
            with gr.Row():
                mode_select = gr.Radio(["Adaptive", "Standard", "Baseline"],
                                       value="Adaptive", label="Attention Mode")
                decay_slider = gr.Slider(0.1, 0.9, value=0.5, step=0.1, label="Decay")
            with gr.Row():
                seed_input  = gr.Number(value=42,  label="Seed",      precision=0)
                steps_input = gr.Number(value=50,  label="Steps",     precision=0)
                cfg_input   = gr.Number(value=7.5, label="CFG Scale")

            gr.Markdown("### Character overrides (optional)")
            with gr.Row():
                char1_name = gr.Textbox(label="Char 1 name", placeholder="e.g. detective")
                char1_desc = gr.Textbox(label="Description",
                                        placeholder="e.g. woman in red coat, dark hair",
                                        layout=gr.Layout(width="400px") if hasattr(gr, "Layout") else {})
            with gr.Row():
                char2_name = gr.Textbox(label="Char 2 name", placeholder="e.g. ghost")
                char2_desc = gr.Textbox(label="Description",
                                        placeholder="e.g. translucent glowing figure")

            with gr.Row():
                gen_btn  = gr.Button("Generate Comic", variant="primary")
                comp_btn = gr.Button("Compare All Modes", variant="secondary")

        with gr.Column(scale=2):
            output_image = gr.Image(label="Comic Strip", type="pil")
            output_info  = gr.Markdown()

    gen_btn.click(
        fn=generate,
        inputs=[story_input, mode_select, decay_slider, seed_input,
                steps_input, cfg_input, char1_name, char1_desc, char2_name, char2_desc],
        outputs=[output_image, output_info],
    )
    comp_btn.click(
        fn=ablation,
        inputs=[story_input, seed_input, char1_name, char1_desc, char2_name, char2_desc],
        outputs=[output_image, output_info],
    )

    gr.Markdown("""
---
**Attention modes:**
- **Adaptive** *(our contribution)* — Adaptive Consistent SA on the middle UNet block only.
  Decay-weighted: neighboring panels influence each other more than distant ones.
  Best balance of quality and consistency.
- **Standard** — StoryDiffusion Consistent SA on all 16 blocks.
  Higher consistency but lower image quality.
- **Baseline** — Standard SD without any SA modification.
  Best image quality, no character consistency across panels.
""")

if __name__ == "__main__":
    demo.launch(share=True)
