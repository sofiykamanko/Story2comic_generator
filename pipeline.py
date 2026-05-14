"""
Main pipeline: text story → 4-panel comic strip with readable dialogue.

Steps:
  1. Claude parses the story into 4 scenes (no dialogue yet)
  2. SD 1.5 + LoRA + Adaptive Consistent SA generates 4 panels
  3. Claude looks at the generated images and writes dialogues/captions
  4. BubbleCleaner replaces scribbles with readable text
  5. Caption boxes + final assembly → comic strip PNG
"""

import os
from PIL import Image

from llm import parse_story, build_prompt, generate_dialogues
from image_gen import load_pipeline, generate_panels, apply_middle_block_attention, reset_attention
from bubble_cleaner import process_comic_panels
from bubbles import create_comic


def run_pipeline(
    story: str,
    pipe,
    mode: str = "adaptive",
    decay: float = 0.5,
    seed: int = 42,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    save_path: str = "comic_output.png",
    panels_dir: str = "/tmp/panels",
    font_path: str = "/content/Bangers.ttf",
    character_overrides: dict = None,
    custom_model_path: str = None,
):
    """
    End-to-end pipeline from a text story to a saved comic strip.

    Args:
        story:               the input story (3–6 sentences)
        pipe:                loaded SD pipeline
        mode:                attention mode — 'adaptive' | 'standard' | 'baseline'
                             'adaptive'  our contribution: middle-block SA with decay weights
                             'standard'  StoryDiffusion SA on all blocks
                             'baseline'  standard SD, no consistency modification
        decay:               neighbor influence for adaptive mode (0.1–0.9, default 0.5)
        seed:                random seed for reproducibility
        num_inference_steps: DDIM steps (50 recommended)
        guidance_scale:      CFG scale (7.5 recommended)
        save_path:           where to save the final comic strip PNG
        panels_dir:          temp directory for raw panel images
        font_path:           path to the Bangers .ttf font
        character_overrides: override character descriptions, e.g.
                             {"sarah": "woman in red coat, dark hair"}
        custom_model_path:   ignored (YOLO path is set via YOLO_MODEL_PATH in bubble_cleaner.py)

    Returns:
        (comic_image, result, prompts, clean_images)
    """
    os.makedirs(panels_dir, exist_ok=True)

    # Step 1: parse story into scenes (no dialogue yet)
    print("Step 1: Parsing story...")
    result = parse_story(story)
    print(f"  Characters: {[c['name'] for c in result['characters']]}")
    for p in result["panels"]:
        print(f"  Panel {p['panel_number']}: {p['scene'][:60]}")

    if character_overrides:
        for c in result["characters"]:
            for key, val in character_overrides.items():
                if key.lower() in c["name"].lower():
                    c["description"] = val
                    print(f"  Override: {c['name']} → {val}")

    # Step 2: build SD prompts
    print("\nStep 2: Building prompts...")
    prompts = [build_prompt(p, result["characters"]) for p in result["panels"]]
    for i, pn in enumerate(prompts):
        p = pn[0] if isinstance(pn, tuple) else pn
        print(f"  Panel {i+1}: {p[:80]}")

    # Step 3: set attention mode
    print(f"\nStep 3: Attention mode='{mode}'...")
    if mode == "baseline":
        reset_attention(pipe)
    elif mode == "adaptive":
        apply_middle_block_attention(pipe, decay=decay)
    elif mode == "standard":
        from consistent_attention import apply_consistent_attention
        apply_consistent_attention(pipe, mode="standard")

    # Step 4: generate panel images
    print(f"\nStep 4: Generating panels (steps={num_inference_steps}, cfg={guidance_scale}, seed={seed})...")
    images = generate_panels(
        prompts=prompts,
        pipe=pipe,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )

    panel_paths = []
    for i, img in enumerate(images):
        path = f"{panels_dir}/panel_{i+1:02d}.png"
        img.save(path)
        panel_paths.append(path)
    print(f"  Saved {len(images)} raw panels to {panels_dir}")

    # Step 5: generate dialogues by looking at the actual images
    print("\nStep 5: Generating dialogues (Claude Vision)...")
    panel_dialogues = generate_dialogues(images, result, story)

    for panel, d in zip(result["panels"], panel_dialogues):
        panel["dialogue"] = d.get("dialogue", "")
        panel["caption"] = d.get("caption", "")

    # Step 6: replace scribbles with readable text
    print("\nStep 6: BubbleCleaner (YOLO + LaMa + PIL)...")
    characters = [c["name"] for c in result["characters"]]
    dialogues = [p.get("dialogue", "") for p in result["panels"]]
    captions = [p.get("caption", "") for p in result["panels"]]
    scene_contexts = [p.get("scene", "") for p in result["panels"]]

    clean_images = process_comic_panels(
        panel_paths,
        dialogues,
        captions=captions,
        font_path=font_path,
        characters=characters,
        scene_contexts=scene_contexts,
    )

    # Step 7: assemble final comic strip
    print("\nStep 7: Assembling comic strip...")
    comic = create_comic(clean_images, result["panels"])
    comic.save(save_path, dpi=(150, 150))
    print(f"  Saved to {save_path}")

    return comic, result, prompts, clean_images


def compare_modes(
    story: str,
    pipe,
    save_path: str = "comparison.png",
    font_path: str = "/content/Bangers.ttf",
    character_overrides: dict = None,
):
    """
    Generate the same story in all 3 attention modes and stack them vertically.
    Used for ablation study figures in the thesis.
    """
    all_strips = []
    for mode in ["baseline", "standard", "adaptive"]:
        print(f"\n{'='*40}\nMode: {mode}\n{'='*40}")
        comic, _, _, _ = run_pipeline(
            story, pipe,
            mode=mode,
            save_path=f"/tmp/comic_{mode}.png",
            panels_dir=f"/tmp/panels_{mode}",
            font_path=font_path,
            character_overrides=character_overrides,
        )
        all_strips.append((mode, comic))

    w = all_strips[0][1].width
    h = all_strips[0][1].height
    label_h = 30
    combined = Image.new("RGB", (w, (h + label_h) * 3 + 20), color="white")

    from PIL import ImageDraw
    draw = ImageDraw.Draw(combined)
    labels = {
        "baseline": "Baseline (no SA) — best quality, no consistency",
        "standard": "Standard Consistent SA (StoryDiffusion) — all blocks",
        "adaptive": "Adaptive SA — Ours (middle block only, decay=0.5)",
    }

    for i, (mode, strip) in enumerate(all_strips):
        y = i * (h + label_h + 10)
        combined.paste(strip, (0, y))
        draw.text((10, y + h + 5), labels[mode], fill="black")

    combined.save(save_path)
    print(f"\nComparison saved to {save_path}")
    return combined
