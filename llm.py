import anthropic
import json
import re
import base64
import io
from PIL import Image as PILImage
from transformers import CLIPTokenizer

client = anthropic.Anthropic()
tokenizer_clip = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")


# Step 1: parse the story into 4 scenes (no dialogues yet — those come after image generation)

SCENE_PROMPT = """You are a comic strip writer. Given a short story, split it into exactly 4 panels.

Return ONLY a valid JSON object:
{
  "characters": [
    {
      "name": "character_name",
      "description": "ONE main feature, clothing, ONE distinctive feature. Max 10 words."
    }
  ],
  "panels": [
    {
      "panel_number": 1,
      "scene": "background setting only, no characters",
      "action": "what the character is physically doing",
      "emotion": "one word emotion",
      "characters_present": ["ONLY ONE character name"]
    }
  ]
}

Rules:
- Exactly 4 panels: setup -> conflict -> climax -> resolution
- Each panel has EXACTLY ONE character
- Main character appears in panels 1, 2, 4 — secondary character in panel 3
- scene = background only, action = what the character does
- Return ONLY JSON, no extra text"""


def parse_story(story: str) -> dict:
    """
    Break the story into 4 scenes with character descriptions.
    Dialogues are NOT generated here — they come after we see the actual images.
    """
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        system=SCENE_PROMPT,
        messages=[{"role": "user", "content": f"Story:\n{story}"}],
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


# Step 2: generate dialogues by looking at the actual generated images

DIALOGUE_PROMPT = """You are a comic strip writer. Look at this comic panel image.

Characters in scene: {characters}
Scene description: {scene}
Story context: {story}

Write a short dialogue (max 6 words) spoken by the character in this panel,
and a short narrator caption (max 8 words) describing what is happening.

Return ONLY valid JSON:
{{"dialogue": "spoken text or empty string", "caption": "narrator text or empty string"}}"""


def _image_to_base64(image: PILImage.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def generate_dialogues(images: list, result: dict, story: str) -> list:
    """
    Claude looks at each generated panel and writes dialogue + caption
    based on what is actually drawn — not guessed in advance.

    Returns a list of dicts: [{"dialogue": "...", "caption": "..."}, ...]
    """
    characters = [c["name"] for c in result["characters"]]
    dialogues = []

    for i, (image, panel) in enumerate(zip(images, result["panels"])):
        print(f"  Panel {i+1}: generating dialogue...")
        prompt = DIALOGUE_PROMPT.format(
            characters=", ".join(characters),
            scene=panel.get("scene", ""),
            story=story[:200],
        )
        try:
            msg = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=200,
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
            raw = msg.content[0].text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            panel_dialogue = json.loads(raw)
        except Exception as e:
            print(f"    Error: {e}")
            panel_dialogue = {"dialogue": "", "caption": ""}

        dialogues.append(panel_dialogue)
        print(f"    → '{panel_dialogue.get('dialogue', '')}' | '{panel_dialogue.get('caption', '')}'")

    return dialogues


# SD prompt builder

def build_prompt(panel: dict, characters: list) -> tuple:
    """
    Build (prompt, negative_prompt) for SD image generation.
    Keeps it under 75 CLIP tokens to avoid truncation.
    """
    char_map = {c["name"]: c["description"] for c in characters}

    chars = []
    for name in panel.get("characters_present", []):
        if name in char_map:
            chars.append(char_map[name].split(",")[0])

    scene = panel.get("scene", "")
    action = panel.get("action", "")
    emotion = panel.get("emotion", "")

    style = "comicstyle, comic book, bold outlines, speech bubble"

    parts = [style, scene]
    if chars:
        parts.append(", ".join(chars))
    if action:
        parts.append(action)
    if emotion:
        parts.append(f"{emotion} expression")

    prompt = ", ".join(p for p in parts if p)
    negative = "text, watermark, blurry, extra characters, low quality, deformed"

    tokens = tokenizer_clip.encode(prompt)
    if len(tokens) > 75:
        tokens = tokens[:75]
        prompt = tokenizer_clip.decode(tokens, skip_special_tokens=True)

    return prompt, negative


def find_character_token_indices(prompt: str, character_name: str) -> list:
    tokens = tokenizer_clip.tokenize(prompt)
    name_tokens = tokenizer_clip.tokenize(character_name)
    indices = []
    for i in range(len(tokens) - len(name_tokens) + 1):
        if tokens[i:i + len(name_tokens)] == name_tokens:
            indices.extend(range(i + 1, i + len(name_tokens) + 1))
    return indices


def build_character_indices(result: dict, prompts: list) -> dict:
    character_indices = {}
    for char in result["characters"]:
        panel_map = {}
        for i, panel in enumerate(result["panels"]):
            if char["name"] in panel.get("characters_present", []):
                p = prompts[i][0] if isinstance(prompts[i], tuple) else prompts[i]
                indices = find_character_token_indices(p, char["name"])
                if indices:
                    panel_map[i] = indices
        if panel_map:
            character_indices[char["name"]] = panel_map
    return character_indices
