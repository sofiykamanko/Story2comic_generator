"""
BubbleCleaner: replaces illegible SD-generated scribbles in speech bubbles
with readable text using YOLO detection, LaMa inpainting, and PIL rendering.

Pipeline per panel:
  1. Fine-tuned YOLO detects all speech bubbles and returns masks + bboxes
  2. SAM2 runs as a fallback if YOLO finds nothing
  3. Single bubble  → use the pre-written dialogue from the JSON
     Multiple bubbles → Claude Vision looks at the image and assigns
                        the right dialogue to each bubble based on
                        where each character is standing
  4. LaMa inpaints each bubble (reconstructs background under the scribbles)
  5. PIL fills the bubble with its original color and draws readable text

Install (Google Colab, T4 GPU):
    !pip install git+https://github.com/facebookresearch/sam2.git -q
    !pip install simple-lama-inpainting easyocr ultralytics -q
    !wget -q "https://github.com/google/fonts/raw/main/ofl/bangers/Bangers-Regular.ttf" -O /content/Bangers.ttf
"""

import numpy as np
import json
import base64
import io
import cv2
import anthropic
from PIL import Image as PILImage, ImageDraw, ImageFont
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import easyocr

# lazy-loaded singletons — initialized once on first use
_mask_generator = None
_lama = None
_ocr_reader = None
_yolo_bubble = None   # fine-tuned detection model (finds bubbles accurately)
_yolo_seg = None      # kitsumed segmentation model (refines mask shape)

YOLO_MODEL_PATH = "/content/yolo_model.pt"  # fine-tuned on SD-generated panels
YOLO_CONF = 0.25


def _init_models():
    global _mask_generator, _lama, _ocr_reader, _yolo_bubble, _yolo_seg

    if _mask_generator is None:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        checkpoint_path = hf_hub_download(
            repo_id="facebook/sam2-hiera-base-plus",
            filename="sam2_hiera_base_plus.pt"
        )
        model = build_sam2("sam2_hiera_b+.yaml", checkpoint_path, device="cuda")
        _mask_generator = SAM2AutomaticMaskGenerator(model)
        print("SAM2 ready")

    if _lama is None:
        from simple_lama_inpainting import SimpleLama
        _lama = SimpleLama()
        print("LaMa ready")

    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(["en"], gpu=True)
        print("EasyOCR ready")

    if _yolo_bubble is None:
        _yolo_bubble = YOLO(YOLO_MODEL_PATH)
        print("YOLO ready")

    if _yolo_seg is None:
        seg_path = hf_hub_download(
            repo_id="kitsumed/yolov8m_seg-speech-bubble",
            filename="model.pt"
        )
        _yolo_seg = YOLO(seg_path)
        print("Segmentation YOLO ready")


# ── Bubble detection ──────────────────────────────────────────────────────────

def detect_bubbles_yolo(image_np, conf=YOLO_CONF):
    """
    Combined detector:
      1. Fine-tuned YOLO finds bubble locations (high accuracy for SD panels)
      2. kitsumed segmentation model refines the mask shape inside each bbox
      3. Falls back to rectangular bbox mask if kitsumed finds nothing
    Returns a list of dicts: {mask (bool array), bbox (x/y/w/h), conf (float)}
    """
    results = _yolo_bubble.predict(source=image_np, conf=conf, verbose=False)
    if not results or results[0].boxes is None:
        return []

    h, w = image_np.shape[:2]
    total_area = h * w
    bubbles = []

    for i in range(len(results[0].boxes)):
        conf_score = float(results[0].boxes.conf[i].cpu().numpy())
        x1, y1, x2, y2 = map(int, results[0].boxes.xyxy[i].cpu().numpy())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # try to get a precise mask shape from kitsumed segmentation model
        mask_bool = None
        if _yolo_seg is not None:
            crop = image_np[y1:y2, x1:x2]
            seg_res = _yolo_seg.predict(source=crop, conf=0.1, verbose=False)
            if seg_res and seg_res[0].masks is not None:
                seg_masks = seg_res[0].masks.data.cpu().numpy()
                best = max(seg_masks, key=lambda m: m.sum())
                ch, cw = y2 - y1, x2 - x1
                seg_img = PILImage.fromarray((best * 255).astype(np.uint8)).resize(
                    (cw, ch), PILImage.NEAREST
                )
                seg_crop = np.array(seg_img) > 127
                mask_bool = np.zeros((h, w), dtype=bool)
                mask_bool[y1:y2, x1:x2] = seg_crop

        # fallback: rectangular bbox mask
        if mask_bool is None or mask_bool.sum() / total_area < 0.003:
            mask_bool = np.zeros((h, w), dtype=bool)
            mask_bool[y1:y2, x1:x2] = True

        area_ratio = mask_bool.sum() / total_area
        if area_ratio < 0.003 or area_ratio > 0.7:
            continue

        rows, cols = np.where(mask_bool)
        bbox = {
            "x": int(cols.min()),
            "y": int(rows.min()),
            "w": int(cols.max() - cols.min()),
            "h": int(rows.max() - rows.min()),
        }
        bubbles.append({
            "mask": mask_bool,
            "bbox": bbox,
            "conf": conf_score,
        })

    return bubbles


def find_bubble_mask_sam2(image_np, masks):
    """
    SAM2 fallback: find the single best speech bubble candidate
    using brightness, uniformity, aspect ratio, and OCR signal.
    """
    total_area = image_np.shape[0] * image_np.shape[1]
    candidates = []

    for mask in masks:
        seg = mask["segmentation"]
        area = seg.sum()
        if area < total_area * 0.02 or area > total_area * 0.6:
            continue

        pixels = image_np[seg].astype(float)
        brightness = pixels.mean()
        std = pixels.std()

        if brightness <= 100 or std >= 100:
            continue

        rows, cols = np.where(seg)
        h = rows.max() - rows.min()
        w = cols.max() - cols.min()
        aspect = min(h, w) / max(h, w) if max(h, w) > 0 else 0

        if aspect < 0.2:
            continue

        y1, y2 = int(rows.min()), int(rows.max())
        x1, x2 = int(cols.min()), int(cols.max())
        crop = image_np[y1:y2, x1:x2]
        ocr_results = _ocr_reader.readtext(crop)
        text_bonus = 200 if ocr_results else 0

        score = brightness * 0.4 + (255 - std) * 0.3 + aspect * 10 + text_bonus
        candidates.append((seg, score, len(ocr_results) > 0, aspect))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[1], reverse=True)
    best = candidates[0]
    if best[2] and best[1] >= 150:
        return best[0]
    if best[1] >= 180 and best[3] >= 0.3:
        return best[0]
    return None


# ── Claude Vision: assign dialogues to multiple bubbles ───────────────────────

def _image_to_base64(image: PILImage.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def assign_dialogues_with_claude(
    image: PILImage.Image,
    bubbles: list,
    characters: list,
    scene_context: str = "",
    fallback_dialogue: str = "",
) -> list:
    """
    When there are multiple bubbles, Claude looks at the actual image
    and figures out which character is speaking in each bubble based
    on their position (left-to-right ordering).

    Returns dialogues in the same order as the input bubbles list.
    """
    client = anthropic.Anthropic()

    # sort left-to-right so Claude gets a consistent ordering
    indexed = sorted(enumerate(bubbles), key=lambda x: x[1]["bbox"]["x"])
    descriptions = [
        f"Bubble {rank+1}: x={b['bbox']['x']}, y={b['bbox']['y']}"
        for rank, (_, b) in enumerate(indexed)
    ]

    prompt = f"""This is a comic panel with {len(bubbles)} speech bubbles containing illegible scribbles.

Characters in the scene: {', '.join(characters)}
Scene: {scene_context}

Bubble positions (left to right):
{chr(10).join(descriptions)}

Look at the image and figure out which character is near each bubble.
Write a short line of dialogue (max 6 words) for each bubble.

Return ONLY a JSON array with exactly {len(bubbles)} strings, left to right:
["dialogue 1", "dialogue 2", ...]"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": _image_to_base64(image),
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }]
        )
        raw = response.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        dialogues_sorted = json.loads(raw)

        result = [fallback_dialogue] * len(bubbles)
        for rank, (orig_idx, _) in enumerate(indexed):
            if rank < len(dialogues_sorted):
                result[orig_idx] = dialogues_sorted[rank]
        return result

    except Exception as e:
        print(f"  Claude Vision error: {e}")
        return [fallback_dialogue] * len(bubbles)


# ── Inpainting and text rendering ─────────────────────────────────────────────

def postprocess_mask(mask_bool):
    """Close small gaps and slightly expand the mask for cleaner inpainting."""
    mask = mask_bool.astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask > 127


def draw_centered_text(draw, text, cx, cy, font, fill="black", max_width=None):
    """Render text centered in the bubble with automatic word wrapping."""
    if max_width:
        words = text.replace("\n", " ").split()
        lines = []
        current = []
        for word in words:
            test = " ".join(current + [word])
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2] - bbox[0] > max_width * 0.85 and current:
                lines.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            lines.append(" ".join(current))
    else:
        lines = text.split("\n")

    line_height = font.getbbox("A")[3] + 8
    total_h = line_height * len(lines)
    y = cy - total_h // 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        lw = bbox[2] - bbox[0]
        draw.text((cx - lw // 2, y), line, fill=fill, font=font)
        y += line_height


def _get_font(font_path, size):
    for path in [font_path, "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _process_single_mask(image_obj, mask, fill_text, font_path):
    """
    For one bubble mask:
      1. Sample the bubble's original color (90th percentile brightness)
      2. LaMa inpaints the scribbles away
      3. Fill the bubble region with its original color
      4. Render the dialogue text centered inside, scaled to fit
    """
    if mask is None or not fill_text:
        return image_obj

    image_np = np.array(image_obj)
    img_h, img_w = image_np.shape[:2]

    def resize_mask(m, w, h):
        return np.array(
            PILImage.fromarray(m.astype(np.uint8) * 255).resize((w, h), PILImage.NEAREST)
        ).astype(bool)

    # sample bubble color before inpainting
    mask_orig = resize_mask(mask, img_w, img_h)
    bubble_pixels = image_np[mask_orig]
    bubble_color = tuple(np.percentile(bubble_pixels, 90, axis=0).astype(int))

    # inpaint scribbles
    mask_pil = PILImage.fromarray(mask.astype(np.uint8) * 255).resize(
        (img_w, img_h), PILImage.NEAREST
    )
    result = _lama(image_obj, mask_pil)
    result_np = np.array(result).copy()

    # restore bubble color
    mask_bool = resize_mask(mask, img_w, img_h)
    result_np[mask_bool] = bubble_color
    result_img = PILImage.fromarray(result_np)
    draw = ImageDraw.Draw(result_img)

    rows, cols = np.where(mask_bool)
    b_h = int(rows.max() - rows.min())
    b_w = int(cols.max() - cols.min())

    # find the best font size that fits the text inside the bubble
    words = fill_text.split()
    max_chars = max(len(w) for w in words) if words else 10
    font_size = max(12, min(40, b_w // max(max_chars, 1)))
    font_size = min(font_size, b_h // 3)
    font = _get_font(font_path, font_size)

    while font_size > 10:
        test_lines = []
        cur = []
        for word in words:
            test = " ".join(cur + [word])
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2] - bbox[0] > b_w * 0.85 and cur:
                test_lines.append(" ".join(cur))
                cur = [word]
            else:
                cur.append(word)
        if cur:
            test_lines.append(" ".join(cur))
        total_h = (font.getbbox("A")[3] + 8) * len(test_lines)
        if total_h < b_h * 0.85:
            break
        font_size -= 2
        font = _get_font(font_path, font_size)

    # caption box (wide & flat) → center geometrically
    # speech bubble → center in upper 55% to avoid the tail
    if b_w > b_h * 2:
        cx = int((cols.min() + cols.max()) / 2)
        cy = int((rows.min() + rows.max()) / 2)
    else:
        row_cutoff = int(rows.min() + (rows.max() - rows.min()) * 0.55)
        upper_mask = mask_bool.copy()
        upper_mask[row_cutoff:, :] = False
        upper_rows, upper_cols = np.where(upper_mask)
        if len(upper_rows) == 0:
            cx = int((cols.min() + cols.max()) / 2)
            cy = int((rows.min() + rows.max()) / 2)
        else:
            cx = int(upper_cols.mean())
            cy = int(upper_rows.mean())

    # clamp so text never goes out of frame
    margin = b_w // 2 + 10
    cx = max(margin, min(img_w - margin, cx))

    draw_centered_text(draw, fill_text, cx, cy, font, max_width=b_w)
    return result_img


# ── Public API ────────────────────────────────────────────────────────────────

def process_panel(
    image_path,
    dialogue,
    caption_text=None,
    output_path=None,
    font_path="/content/Bangers.ttf",
    characters=None,
    scene_context="",
):
    """
    Full pipeline for one comic panel.

    - YOLO finds all bubbles; SAM2 is used as fallback if YOLO finds nothing
    - Single bubble → use the provided dialogue string
    - Multiple bubbles → Claude Vision assigns the right line to each bubble
    - Each bubble is inpainted and re-filled with readable text
    """
    _init_models()

    image = PILImage.open(image_path).convert("RGB")
    image_np = np.array(image)

    yolo_bubbles = detect_bubbles_yolo(image_np)
    print(f"  YOLO: {len(yolo_bubbles)} bubble(s) found")

    if not yolo_bubbles:
        print("  Falling back to SAM2...")
        masks = _mask_generator.generate(image_np)
        sam_mask = find_bubble_mask_sam2(image_np, masks)
        if sam_mask is None:
            print("  No bubble found — keeping original")
            return image
        sam_mask = postprocess_mask(sam_mask)
        image = _process_single_mask(image, sam_mask, dialogue, font_path)
        if output_path:
            image.save(output_path)
        return image

    for b in yolo_bubbles:
        b["mask"] = postprocess_mask(b["mask"])

    if len(yolo_bubbles) == 1:
        dialogues = [dialogue]
    else:
        print(f"  Multiple bubbles — asking Claude Vision...")
        dialogues = assign_dialogues_with_claude(
            image=image,
            bubbles=yolo_bubbles,
            characters=characters or [],
            scene_context=scene_context,
            fallback_dialogue=dialogue,
        )
        print(f"  Assigned: {dialogues}")

    for bubble, text in zip(yolo_bubbles, dialogues):
        if text:
            image = _process_single_mask(image, bubble["mask"], text, font_path)

    if output_path:
        image.save(output_path)
        print(f"  Saved: {output_path}")

    return image


def process_comic_panels(
    image_paths,
    dialogues,
    captions=None,
    output_dir="/content",
    font_path="/content/Bangers.ttf",
    characters=None,
    scene_contexts=None,
):
    """
    Process all 4 panels of a comic strip.

    Args:
        image_paths:    paths to the raw generated panel images
        dialogues:      fallback dialogue strings (used when there is one bubble)
        captions:       narrator captions — rendered by bubbles.py, not here
        output_dir:     where to save cleaned panels
        font_path:      path to the Bangers .ttf font file
        characters:     character names passed to Claude Vision
        scene_contexts: scene descriptions passed to Claude Vision
    """
    if captions is None:
        captions = [""] * len(image_paths)
    if scene_contexts is None:
        scene_contexts = [""] * len(image_paths)

    results = []
    for i, (path, text, cap, ctx) in enumerate(
        zip(image_paths, dialogues, captions, scene_contexts)
    ):
        print(f"\nPanel {i+1}: {path}")
        result = process_panel(
            path,
            dialogue=text,
            caption_text=cap,
            output_path=f"{output_dir}/clean_{i+1:02d}.png",
            font_path=font_path,
            characters=characters,
            scene_context=ctx,
        )
        results.append(result)
    return results
