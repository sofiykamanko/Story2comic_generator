import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention


class ConsistentSelfAttentionProcessor:
    """
    Standard Consistent Self-Attention from StoryDiffusion (NeurIPS 2024).

    Instead of each panel attending only to itself:
        out_i = Attn(Q_i, K_i, V_i)

    All panels share keys and values:
        K_shared = concat([K_1, K_2, K_3, K_4])
        out_i    = Attn(Q_i, K_shared, V_shared)

    This forces the model to produce consistent visual features
    (same character appearance) across all 4 panels.

    Note: batch=8 because CFG runs unconditioned + conditioned passes,
    so 4 panels × 2 = 8 items in the batch.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len, dim = hidden_states.shape
        heads = attn.heads
        half = batch_size // 2
        head_dim = dim // heads

        query = attn.head_to_batch_dim(attn.to_q(hidden_states))
        key   = attn.head_to_batch_dim(attn.to_k(hidden_states))
        value = attn.head_to_batch_dim(attn.to_v(hidden_states))

        if encoder_hidden_states is None:
            q = query.reshape(batch_size, heads, seq_len, head_dim)
            k = key.reshape(batch_size, heads, seq_len, head_dim)
            v = value.reshape(batch_size, heads, seq_len, head_dim)

            outputs = []
            for b in range(batch_size):
                same_half = slice(0, half) if b < half else slice(half, batch_size)
                k_shared = k[same_half].permute(1, 0, 2, 3).reshape(heads, half * seq_len, head_dim)
                v_shared = v[same_half].permute(1, 0, 2, 3).reshape(heads, half * seq_len, head_dim)
                scores = torch.bmm(q[b], k_shared.transpose(1, 2)) * attn.scale
                scores = F.softmax(scores, dim=-1)
                outputs.append(torch.bmm(scores, v_shared))

            hidden_states = torch.stack(outputs, dim=0).reshape(batch_size * heads, seq_len, head_dim)
            hidden_states = attn.batch_to_head_dim(hidden_states)

        else:
            key   = attn.head_to_batch_dim(attn.to_k(encoder_hidden_states))
            value = attn.head_to_batch_dim(attn.to_v(encoder_hidden_states))
            scores = torch.bmm(query, key.transpose(1, 2)) * attn.scale
            scores = F.softmax(scores, dim=-1)
            hidden_states = torch.bmm(scores, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class AdaptiveConsistentSelfAttentionProcessor:
    """
    Our modification of StoryDiffusion — Adaptive Consistent Self-Attention.

    Instead of treating all panels equally, we use decay-weighted attention:

        out_i = sum_j( w[i,j] * Attn(Q_i, K_j, V_j) )

    where w[i,j] = decay^|i-j|, normalized so weights sum to 1.

    This means neighboring panels have more influence on each other
    than distant ones, which better preserves local narrative coherence.

    Weight matrix for 4 panels with decay=0.5:
        panel 0: [0.53, 0.27, 0.13, 0.07]
        panel 1: [0.27, 0.53, 0.27, 0.13]
        panel 2: [0.13, 0.27, 0.53, 0.27]
        panel 3: [0.07, 0.13, 0.27, 0.53]

    We apply this only to the UNet middle block (see apply_middle_block_attention),
    which gives the best quality/consistency tradeoff.
    """

    def __init__(self, num_panels: int = 4, decay: float = 0.5):
        self.num_panels = num_panels
        self.decay = decay

        weights = torch.zeros(num_panels, num_panels)
        for i in range(num_panels):
            for j in range(num_panels):
                weights[i, j] = decay ** abs(i - j)
        self.weights = weights / weights.sum(dim=1, keepdim=True)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len, dim = hidden_states.shape
        heads = attn.heads
        half = batch_size // 2
        head_dim = dim // heads

        query = attn.head_to_batch_dim(attn.to_q(hidden_states))
        key   = attn.head_to_batch_dim(attn.to_k(hidden_states))
        value = attn.head_to_batch_dim(attn.to_v(hidden_states))

        if encoder_hidden_states is None:
            w = self.weights.to(hidden_states.device)

            q = query.reshape(batch_size, heads, seq_len, head_dim)
            k = key.reshape(batch_size, heads, seq_len, head_dim)
            v = value.reshape(batch_size, heads, seq_len, head_dim)

            outputs = []
            for b in range(batch_size):
                panel_idx = b if b < half else b - half
                same_half = slice(0, half) if b < half else slice(half, batch_size)

                q_b    = q[b]
                k_half = k[same_half]
                v_half = v[same_half]

                out_b = torch.zeros_like(q_b)
                for j in range(half):
                    scores_j = torch.bmm(q_b, k_half[j].transpose(1, 2)) * attn.scale
                    scores_j = F.softmax(scores_j, dim=-1)
                    out_b = out_b + w[panel_idx, j] * torch.bmm(scores_j, v_half[j])

                outputs.append(out_b)

            hidden_states = torch.stack(outputs, dim=0).reshape(batch_size * heads, seq_len, head_dim)
            hidden_states = attn.batch_to_head_dim(hidden_states)

        else:
            key   = attn.head_to_batch_dim(attn.to_k(encoder_hidden_states))
            value = attn.head_to_batch_dim(attn.to_v(encoder_hidden_states))
            scores = torch.bmm(query, key.transpose(1, 2)) * attn.scale
            scores = F.softmax(scores, dim=-1)
            hidden_states = torch.bmm(scores, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class SharedCharacterCrossAttentionProcessor:
    """
    Our original contribution (not in StoryDiffusion).

    We share character token embeddings across panels via cross-attention.
    When the same character appears in multiple panels, we average their
    text embeddings so the model gets a consistent description of the
    character regardless of which panel is being generated.

    StoryDiffusion modifies self-attention only.
    We additionally modify cross-attention.
    """

    def __init__(self, character_indices: dict, num_panels: int = 4):
        self.character_indices = character_indices
        self.num_panels = num_panels

        self.panel_char_map = {}
        for char_name, panel_map in character_indices.items():
            panels_with_char = list(panel_map.keys())
            for panel_idx, token_indices in panel_map.items():
                if panel_idx not in self.panel_char_map:
                    self.panel_char_map[panel_idx] = {}
                other_panels = [p for p in panels_with_char if p != panel_idx]
                self.panel_char_map[panel_idx][char_name] = {
                    "own_indices": token_indices,
                    "other_panels": other_panels,
                    "other_indices": {p: panel_map[p] for p in other_panels},
                }

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states=None,
        attention_mask=None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len, dim = hidden_states.shape
        heads = attn.heads
        half = batch_size // 2

        query = attn.head_to_batch_dim(attn.to_q(hidden_states))

        if encoder_hidden_states is None:
            key   = attn.head_to_batch_dim(attn.to_k(hidden_states))
            value = attn.head_to_batch_dim(attn.to_v(hidden_states))
        else:
            enc = encoder_hidden_states.clone()
            for b in range(batch_size):
                panel_idx = b if b < half else b - half
                if panel_idx not in self.panel_char_map:
                    continue
                for char_name, char_info in self.panel_char_map[panel_idx].items():
                    own_indices = char_info["own_indices"]
                    shared_emb = enc[b, own_indices].clone()
                    for other_panel, other_indices in char_info["other_indices"].items():
                        other_b = other_panel if b < half else other_panel + half
                        if other_b < batch_size and len(other_indices) > 0:
                            n = min(len(own_indices), len(other_indices))
                            shared_emb[:n] = (shared_emb[:n] + enc[other_b, other_indices[:n]]) / 2
                    enc[b, own_indices] = shared_emb
            key   = attn.head_to_batch_dim(attn.to_k(enc))
            value = attn.head_to_batch_dim(attn.to_v(enc))

        scores = torch.bmm(query, key.transpose(1, 2)) * attn.scale
        scores = F.softmax(scores, dim=-1)
        hidden_states = torch.bmm(scores, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


def apply_consistent_attention(pipeline, mode: str = "adaptive", decay: float = 0.5):
    """
    Replace self-attention processors in all UNet blocks.

    mode:
        "standard"  — original StoryDiffusion Consistent SA
        "adaptive"  — our Adaptive SA with decay-weighted neighbor influence
    """
    processor = (
        AdaptiveConsistentSelfAttentionProcessor(num_panels=4, decay=decay)
        if mode == "adaptive"
        else ConsistentSelfAttentionProcessor()
    )

    count = 0
    for _, module in pipeline.unet.named_modules():
        if isinstance(module, Attention):
            if module.to_k.in_features == module.to_q.in_features:
                module.set_processor(processor)
                count += 1
    print(f"Replaced {count} self-attention blocks (mode={mode})")
    return pipeline


def apply_full_attention(pipeline, character_indices: dict, mode: str = "adaptive", decay: float = 0.5):
    """
    Replace both self-attention (consistent) and cross-attention (shared character).
    This is our full pipeline combining both contributions.
    """
    self_proc = (
        AdaptiveConsistentSelfAttentionProcessor(num_panels=4, decay=decay)
        if mode == "adaptive"
        else ConsistentSelfAttentionProcessor()
    )
    cross_proc = SharedCharacterCrossAttentionProcessor(character_indices)

    self_count = cross_count = 0
    for _, module in pipeline.unet.named_modules():
        if isinstance(module, Attention):
            if module.to_k.in_features == module.to_q.in_features:
                module.set_processor(self_proc)
                self_count += 1
            else:
                module.set_processor(cross_proc)
                cross_count += 1

    print(f"Self-attention: {self_count} blocks replaced (mode={mode})")
    print(f"Cross-attention: {cross_count} blocks replaced (SharedCharacter)")
    return pipeline
