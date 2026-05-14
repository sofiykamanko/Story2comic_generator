from PIL import Image, ImageDraw, ImageFont


def _get_font(size: int = 16) -> ImageFont.ImageFont:
    for path in [
        "/content/Bangers.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def add_caption_box(image: Image.Image, caption: str) -> Image.Image:
    """Add a black caption bar at the bottom of a panel with white narrator text."""
    if not caption:
        return image

    img = image.copy()
    w, h = img.size
    font = _get_font(14)
    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), caption, font=font)
    text_h = bbox[3] - bbox[1]
    box_h = text_h + 16

    draw.rectangle([0, h - box_h, w, h], fill="black")

    bbox = draw.textbbox((0, 0), caption, font=font)
    text_w = bbox[2] - bbox[0]
    draw.text(((w - text_w) // 2, h - box_h + 8), caption, fill="white", font=font)

    return img


def create_comic(images: list, panels: list, border: int = 6, gap: int = 4) -> Image.Image:
    """
    Assemble 4 panels into a horizontal comic strip.
    BubbleCleaner has already handled the speech bubbles.
    Here we just add caption boxes and stitch everything together.
    """
    panel_w, panel_h = images[0].size
    n = len(images)
    total_w = n * panel_w + (n - 1) * gap + 2 * border
    total_h = panel_h + 2 * border

    comic = Image.new("RGB", (total_w, total_h), color="black")

    for i, (img, panel) in enumerate(zip(images, panels)):
        img = add_caption_box(img, panel.get("caption", ""))
        x = border + i * (panel_w + gap)
        comic.paste(img, (x, border))

    return comic
