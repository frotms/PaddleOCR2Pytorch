"""
SLANeXt: Table Structure Recognition Model (PyTorch Implementation)

Architecture: ViT-based vision encoder + GRU-attention autoregressive decoder.
Based on PaddleOCR's SLANeXt model for PP-StructureV3.

Weight mapping:
    Paddle key pattern → PyTorch key pattern:
    xx._mean → xx.running_mean
    xx._variance → xx.running_var
    backbone.vision_tower.* → backbone.vision_tower.*
    backbone.post_conv.* → backbone.post_conv.*
    head.structure_attention_cell.* → head.structure_attention_cell.*
    head.structure_generator.* → head.structure_generator.*
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class SLANeXtVisionConfig:
    """Vision encoder configuration for SLANeXt."""
    def __init__(self, **kwargs):
        self.hidden_size = kwargs.get("hidden_size", 768)
        self.output_channels = kwargs.get("output_channels", 256)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 12)
        self.num_attention_heads = kwargs.get("num_attention_heads", 12)
        self.num_channels = kwargs.get("num_channels", 3)
        self.image_size = kwargs.get("image_size", 512)
        self.patch_size = kwargs.get("patch_size", 16)
        self.hidden_act = kwargs.get("hidden_act", "gelu")
        self.layer_norm_eps = kwargs.get("layer_norm_eps", 1e-6)
        self.qkv_bias = kwargs.get("qkv_bias", True)
        self.use_abs_pos = kwargs.get("use_abs_pos", True)
        self.use_rel_pos = kwargs.get("use_rel_pos", True)
        self.window_size = kwargs.get("window_size", 14)
        self.global_attn_indexes = kwargs.get("global_attn_indexes", [2, 5, 8, 11])
        self.mlp_dim = kwargs.get("mlp_dim", 3072)
        self.attention_dropout = kwargs.get("attention_dropout", 0.0)


class SLANeXtConfig:
    """Configuration for SLANeXt table structure recognition."""
    def __init__(self, **kwargs):
        vision_config = kwargs.pop("vision_config", None)
        if vision_config is None:
            self.vision_config = SLANeXtVisionConfig()
        elif isinstance(vision_config, dict):
            self.vision_config = SLANeXtVisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        self.post_conv_in_channels = kwargs.get("post_conv_in_channels", 256)
        self.post_conv_out_channels = kwargs.get("post_conv_out_channels", 512)
        self.out_channels = kwargs.get("out_channels", 50)
        self.hidden_size = kwargs.get("hidden_size", 512)
        self.max_text_length = kwargs.get("max_text_length", 500)
        self.loc_reg_num = kwargs.get("loc_reg_num", 8)


# ---------------------------------------------------------------------------
# Vision Encoder Components (GotOcr2VisionEncoder equivalent)
# ---------------------------------------------------------------------------

class PatchEmbeddings(nn.Module):
    """Image to patch embeddings using Conv2d projection."""
    def __init__(self, config: SLANeXtVisionConfig):
        super().__init__()
        self.projection = nn.Conv2d(
            config.num_channels, config.hidden_size,
            kernel_size=config.patch_size, stride=config.patch_size
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [B, C, H, W]
        x = self.projection(pixel_values)
        # [B, C, H, W] -> [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        return x


class MLPBlock(nn.Module):
    """Two-layer MLP for vision transformer."""
    def __init__(self, config: SLANeXtVisionConfig):
        super().__init__()
        self.lin1 = nn.Linear(config.hidden_size, config.mlp_dim)
        self.lin2 = nn.Linear(config.mlp_dim, config.hidden_size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


class VisionAttention(nn.Module):
    """Multi-head attention with relative positional embeddings."""
    def __init__(self, config: SLANeXtVisionConfig, window_size: int):
        super().__init__()
        input_size = (config.image_size // config.patch_size,
                      config.image_size // config.patch_size) if window_size == 0 else (window_size, window_size)

        self.num_heads = config.num_attention_heads
        head_dim = config.hidden_size // config.num_attention_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3,
                             bias=config.qkv_bias)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.use_rel_pos = config.use_rel_pos
        if self.use_rel_pos:
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def get_rel_pos(self, q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
        max_rel_dist = int(2 * max(q_size, k_size) - 1)
        if rel_pos.shape[0] != max_rel_dist:
            rel_pos_resized = F.interpolate(
                rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
                size=max_rel_dist, mode="linear"
            )
            rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
        else:
            rel_pos_resized = rel_pos

        q_coords = torch.arange(q_size, dtype=torch.float32).unsqueeze(-1) * max(k_size / q_size, 1.0)
        k_coords = torch.arange(k_size, dtype=torch.float32).unsqueeze(0) * max(q_size / k_size, 1.0)
        relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
        return rel_pos_resized[relative_coords.long()]

    def get_decomposed_rel_pos(self, q: torch.Tensor, rel_pos_h: torch.Tensor,
                                rel_pos_w: torch.Tensor, q_size: Tuple[int, int],
                                k_size: Tuple[int, int]) -> torch.Tensor:
        q_h, q_w = q_size
        k_h, k_w = k_size
        Rh = self.get_rel_pos(q_h, k_h, rel_pos_h)
        Rw = self.get_rel_pos(q_w, k_w, rel_pos_w)

        B, _, dim = q.shape
        r_q = q.reshape((B, q_h, q_w, dim))
        rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
        rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
        return rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = hidden_states.shape
        qkv = (self.qkv(hidden_states)
               .reshape(B, H * W, 3, self.num_heads, -1)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(dim=0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            decomposed = self.get_decomposed_rel_pos(
                q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
            attn = attn.reshape(B, self.num_heads, H, W, H * W)
            attn = attn + decomposed.reshape(B, self.num_heads, H, W, H * W)
            attn = attn.reshape(B * self.num_heads, H * W, H * W)

        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).reshape(B, self.num_heads, H, W, -1) \
            .permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        return self.proj(x)


class VisionLayer(nn.Module):
    """Transformer block with optional window attention."""
    def __init__(self, config: SLANeXtVisionConfig, window_size: int):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = VisionAttention(config, window_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = MLPBlock(config)
        self.window_size = window_size

    @staticmethod
    def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        B, H, W, C = x.shape
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w
        x = x.reshape(B, Hp // window_size, window_size,
                      Wp // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
        return windows, (Hp, Wp)

    @staticmethod
    def window_unpartition(windows: torch.Tensor, window_size: int,
                           pad_hw: Tuple[int, int], hw: Tuple[int, int]) -> torch.Tensor:
        Hp, Wp = pad_hw
        H, W = hw
        B = windows.shape[0] // (Hp * Wp // window_size // window_size)
        x = windows.reshape(B, Hp // window_size, Wp // window_size,
                            window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
        if Hp > H or Wp > W:
            x = x[:, :H, :W, :]
        return x

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        if self.window_size > 0:
            H, W = hidden_states.shape[1], hidden_states.shape[2]
            hidden_states, pad_hw = self.window_partition(hidden_states, self.window_size)
        hidden_states = self.attn(hidden_states)
        if self.window_size > 0:
            hidden_states = self.window_unpartition(hidden_states, self.window_size, pad_hw, (H, W))
        hidden_states = residual + hidden_states
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


class VisionNeck(nn.Module):
    """Vision neck with 1x1 + 3x3 conv."""
    def __init__(self, config: SLANeXtVisionConfig):
        super().__init__()
        self.conv1 = nn.Conv2d(config.hidden_size, config.output_channels,
                               kernel_size=1, bias=False)
        self.layer_norm1 = nn.LayerNorm(config.output_channels, eps=1e-6)
        self.conv2 = nn.Conv2d(config.output_channels, config.output_channels,
                               kernel_size=3, padding=1, bias=False)
        self.layer_norm2 = nn.LayerNorm(config.output_channels, eps=1e-6)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # B H W C -> B C H W
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.conv1(hidden_states)
        # B C H W -> B H W C for LayerNorm, then back
        B, C, H, W = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1)
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        hidden_states = self.conv2(hidden_states)
        hidden_states = hidden_states.permute(0, 2, 3, 1)
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = hidden_states.permute(0, 3, 1, 2)
        return hidden_states


class VisionEncoder(nn.Module):
    """SAM-ViT based vision encoder (equivalent to GotOcr2VisionEncoder)."""
    def __init__(self, config: SLANeXtVisionConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbeddings(config)

        self.pos_embed = None
        if config.use_abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(
                1, config.image_size // config.patch_size,
                config.image_size // config.patch_size, config.hidden_size
            ))

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            ws = config.window_size if i not in config.global_attn_indexes else 0
            self.layers.append(VisionLayer(config, window_size=ws))

        self.neck = VisionNeck(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(pixel_values)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.neck(x)
        return x


# ---------------------------------------------------------------------------
# SLANeXt Table Structure Recognition Model
# ---------------------------------------------------------------------------

class AttentionGRUCell(nn.Module):
    """Attention-based GRU cell for autoregressive decoding."""
    def __init__(self, input_size: int, hidden_size: int, num_embeddings: int):
        super().__init__()
        self.input_to_hidden = nn.Linear(input_size, hidden_size, bias=False)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.GRUCell(input_size + num_embeddings, hidden_size)

    def forward(self, prev_hidden: torch.Tensor, batch_hidden: torch.Tensor,
                char_onehots: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_hidden_proj = self.input_to_hidden(batch_hidden)
        prev_hidden_proj = self.hidden_to_hidden(prev_hidden).unsqueeze(1)
        attention_scores = torch.tanh(batch_hidden_proj + prev_hidden_proj)
        attention_scores = self.score(attention_scores)
        attn_weights = F.softmax(attention_scores, dim=1)
        attn_weights = attn_weights.permute(0, 2, 1)
        context = torch.bmm(attn_weights, batch_hidden).squeeze(1)
        concat_context = torch.cat([context, char_onehots], dim=1)
        hidden_states = self.rnn(concat_context, prev_hidden)
        return hidden_states, attn_weights


class StructureMLP(nn.Module):
    """Two-layer MLP for structure token prediction."""
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


class SLANeXtBackbone(nn.Module):
    """Vision backbone for SLANeXt."""
    def __init__(self, config: SLANeXtConfig):
        super().__init__()
        self.vision_tower = VisionEncoder(config.vision_config)
        self.post_conv = nn.Conv2d(
            config.post_conv_in_channels, config.post_conv_out_channels,
            kernel_size=3, stride=2, padding=1, bias=False
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.vision_tower(pixel_values)
        x = self.post_conv(x)
        # [B, C, H, W] -> [B, H*W, C]
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)
        return x


class SLAHead(nn.Module):
    """Autoregressive SLA head for table structure prediction."""
    def __init__(self, config: SLANeXtConfig):
        super().__init__()
        self.config = config
        self.structure_attention_cell = AttentionGRUCell(
            config.post_conv_out_channels, config.hidden_size, config.out_channels
        )
        self.structure_generator = StructureMLP(config.hidden_size, config.out_channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        dtype = hidden_states.dtype
        features = torch.zeros(batch_size, self.config.hidden_size,
                               device=hidden_states.device, dtype=dtype)
        predicted_chars = torch.zeros(batch_size, dtype=torch.long,
                                       device=hidden_states.device)

        structure_preds_list = []
        structure_ids_list = []
        for _ in range(self.config.max_text_length + 1):
            embedding_feature = F.one_hot(predicted_chars, self.config.out_channels).to(dtype)
            features, _ = self.structure_attention_cell(
                features, hidden_states, embedding_feature
            )
            structure_step = self.structure_generator(features)
            predicted_chars = structure_step.argmax(dim=1)
            structure_preds_list.append(structure_step)
            structure_ids_list.append(predicted_chars)
            # Check if all sequences have generated EOS token
            ids_stack = torch.stack(structure_ids_list, dim=1)
            if (ids_stack == self.config.out_channels - 1).any(dim=-1).all():
                break

        structure_preds = torch.stack(structure_preds_list, dim=1)
        structure_probs = F.softmax(structure_preds, dim=-1)
        return structure_probs


class SLANeXt(nn.Module):
    """SLANeXt table structure recognition model (PyTorch).

    Pipeline:
        pixel_values [B, 3, H, W] → VisionEncoder → Post-Conv
        → [B, N, C] features → Autoregressive GRU Decoder
        → [B, max_len, num_classes] structure logits

    Usage:
        config = SLANeXtConfig()
        model = SLANeXt(config)
        output = model(torch.randn(1, 3, 488, 488))
    """
    def __init__(self, config: SLANeXtConfig):
        super().__init__()
        self.config = config
        self.backbone = SLANeXtBackbone(config)
        self.head = SLAHead(config)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 3, H, W] input images, normalized.

        Returns:
            structure_probs: [B, max_len, num_classes] token probabilities.
        """
        # Handle 1-channel input
        if pixel_values.shape[1] == 1:
            pixel_values = pixel_values.expand(-1, 3, -1, -1)

        features = self.backbone(pixel_values)
        structure_probs = self.head(features)
        return structure_probs

    def get_structure_tokens(self, pixel_values: torch.Tensor) -> List[List[int]]:
        """Get discrete structure token sequences.

        Returns:
            List of token sequences for each batch item.
        """
        probs = self.forward(pixel_values)
        tokens = probs.argmax(dim=-1)  # [B, max_len]
        return tokens.cpu().tolist()
