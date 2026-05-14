import torch
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from consistent_attention import AdaptiveConsistentSelfAttentionProcessor


def load_pipeline(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    lora_path: str = None,
    device: str = "cuda",
) -> StableDiffusionPipeline:
    """Load SD 1.5 with optional LoRA weights."""
    print(f"Loading {model_id}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    if lora_path:
        print(f"Loading LoRA from {lora_path}...")
        pipe.load_lora_weights(lora_path)
        print("LoRA loaded")

    print("Pipeline ready")
    return pipe


def reset_attention(pipeline) -> None:
    """Restore standard self-attention (no consistency, best image quality)."""
    for _, module in pipeline.unet.named_modules():
        if isinstance(module, Attention):
            module.set_processor(AttnProcessor2_0())
    print("Attention reset to standard")


def apply_middle_block_attention(pipeline, decay: float = 0.5):
    """
    Apply Adaptive Consistent SA only to the UNet middle block.

    This is the key finding of our work: applying SA to all 16 blocks
    degrades image quality significantly, but limiting it to just the
    middle block preserves visual quality while still achieving
    consistent character appearance across panels.

    decay controls how much neighboring panels influence each other.
    Lower decay = more local, higher decay = more global consistency.
    """
    for _, module in pipeline.unet.named_modules():
        if isinstance(module, Attention):
            module.set_processor(AttnProcessor2_0())

    processor = AdaptiveConsistentSelfAttentionProcessor(num_panels=4, decay=decay)
    count = 0
    for name, module in pipeline.unet.named_modules():
        if isinstance(module, Attention):
            if "mid_block" in name and module.to_k.in_features == module.to_q.in_features:
                module.set_processor(processor)
                count += 1
    print(f"Middle block SA: {count} layer(s) replaced (decay={decay})")
    return pipeline


def generate_panels(
    prompts: list,
    pipe: StableDiffusionPipeline,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
    seed: int = None,
) -> list:
    """
    Generate 4 comic panels in a single forward pass.
    Accepts prompts as plain strings or (prompt, negative_prompt) tuples.
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

    if prompts and isinstance(prompts[0], tuple):
        prompts_only = [p[0] for p in prompts]
        negatives = [p[1] for p in prompts]
    else:
        prompts_only = list(prompts)
        negatives = ["text, watermark, blurry, low quality"] * len(prompts)

    with torch.autocast("cuda"):
        result = pipe(
            prompt=prompts_only,
            negative_prompt=negatives,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        )
    return result.images
