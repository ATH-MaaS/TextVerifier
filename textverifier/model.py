"""
CJK Character CLIP v6 — Model Definitions

Image Encoder: ViT-B/16 or ViT-L/16 (timm) — 128×128 input, patch=16
Text Encoder:  DeBERTa-v2 (base or large) — from scratch, vocab=981 IDS tokens
Projectors:    2-layer MLP (hidden = max(in, out)), out_dim = 1024
Shared dim:    1024

--model-size {base, large} switches:
  base:  ViT-B/16 + DeBERTa-base  (~175M total)
  large: ViT-L/16 + DeBERTa-large (~612M total)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import DebertaV2Config, DebertaV2Model


# ─── Image Encoder (ViT) ─────────────────────────────────────────────────────

class ViTImageEncoder(nn.Module):
    """
    ViT-B/16 or ViT-L/16 via timm. Input 128×128, patch 16 -> 64 tokens + CLS.
    Uses timm's num_classes=0 to strip classification head, then patch mean pooling.
    """

    def __init__(self, size: str = 'base', pretrained: bool = False,
                 img_size: int = 128, patch_size: int = 16):
        super().__init__()
        assert size in ('base', 'large'), f'unknown size={size}'
        arch = 'vit_base_patch16_224' if size == 'base' else 'vit_large_patch16_224'

        self.backbone = timm.create_model(
            arch,
            pretrained=pretrained,
            img_size=img_size,
            patch_size=patch_size,
            num_classes=0,       # remove classification head
            global_pool='avg',   # patch mean pooling, excludes CLS token
        )
        self.output_dim = self.backbone.embed_dim  # 768 or 1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # timm ViT with global_pool='avg' returns patch-mean-pooled [B, embed_dim]
        return self.backbone(x)


# ─── Text Encoder (DeBERTa-v2) ───────────────────────────────────────────────

class DeBERTaTextEncoder(nn.Module):
    """
    DeBERTa-v2 encoder for IDS sequences. From scratch (no HF weights).
    Uses transformers.DebertaV2Model, takes [CLS] position as output.
    """

    def __init__(
        self,
        vocab_size: int,
        size: str = 'base',
        max_position_embeddings: int = 30,
        pad_token_id: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert size in ('base', 'large'), f'unknown size={size}'

        if size == 'base':
            hidden = 768
            n_layers = 12
            n_heads = 12
            ffn = 3072
        else:  # large
            hidden = 1024
            n_layers = 24
            n_heads = 16
            ffn = 4096

        config = DebertaV2Config(
            vocab_size=vocab_size,
            hidden_size=hidden,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            intermediate_size=ffn,
            hidden_act='gelu',
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=0,        # no segment IDs
            pad_token_id=pad_token_id,
            layer_norm_eps=1e-7,
            relative_attention=True,   # DeBERTa disentangled attention
            position_biased_input=False,
            pos_att_type=['p2c', 'c2p'],
            norm_rel_ebd='layer_norm',
            share_att_key=True,        # DeBERTa-v3 default
            position_buckets=16,        # relative pos buckets for small seq
        )
        self.deberta = DebertaV2Model(config)
        self.pad_token_id = pad_token_id
        self.output_dim = hidden

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        # attention_mask: 1 for real tokens, 0 for pad
        attention_mask = (text != self.pad_token_id).long()
        out = self.deberta(input_ids=text, attention_mask=attention_mask)
        # last_hidden_state: [B, L, hidden]; take [CLS] at position 0
        cls = out.last_hidden_state[:, 0, :]
        return cls


# ─── 2-Layer MLP Projector (hidden = max(in, out) = in_dim) ─────────────────

class MLPProjector(nn.Module):
    """
    2-layer MLP: Linear -> LayerNorm -> GELU -> Linear (no bias).
    Hidden dim = max(in_dim, out_dim).
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        hidden = max(in_dim, out_dim)
        self.fc1 = nn.Linear(in_dim, hidden, bias=False)
        self.ln = nn.LayerNorm(hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out_dim, bias=False)

        nn.init.normal_(self.fc1.weight, std=in_dim ** -0.5)
        nn.init.normal_(self.fc2.weight, std=hidden ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.ln(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# ─── L_I loss (standard SupCon L_out, τ=0.07) ───────────────────────────────

def li_loss(image_features: torch.Tensor, char_ids: torch.Tensor,
            temperature: float = 0.07) -> torch.Tensor:
    B = image_features.size(0)
    if B < 2:
        return torch.tensor(0.0, device=image_features.device, requires_grad=True)

    sim = image_features @ image_features.t() / temperature

    same_char = (char_ids.unsqueeze(0) == char_ids.unsqueeze(1))
    same_char.fill_diagonal_(False)

    has_pos = same_char.any(dim=1)
    if not has_pos.any():
        return torch.tensor(0.0, device=image_features.device, requires_grad=True)

    not_self = ~torch.eye(B, dtype=torch.bool, device=sim.device)
    denom = torch.logsumexp(sim.masked_fill(~not_self, float('-inf')), dim=1)
    log_prob = sim - denom.unsqueeze(1)
    log_prob_pos = log_prob.masked_fill(~same_char, 0.0)
    num_pos = same_char.float().sum(dim=1).clamp(min=1)
    loss_per_anchor = -log_prob_pos.sum(dim=1) / num_pos
    return loss_per_anchor[has_pos].mean()


def compute_clip_loss_multipos(logits_per_image, logits_per_text, char_ids, row_weights=None):
    B = logits_per_image.size(0)
    if B < 2:
        return torch.tensor(0.0, device=logits_per_image.device)

    pos_mask = (char_ids.unsqueeze(0) == char_ids.unsqueeze(1)).float()

    log_prob_i = F.log_softmax(logits_per_image, dim=1)
    loss_i_per = -(log_prob_i * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)

    log_prob_t = F.log_softmax(logits_per_text, dim=1)
    loss_t_per = -(log_prob_t * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)

    if row_weights is None:
        loss_i = loss_i_per.mean()
        loss_t = loss_t_per.mean()
    else:
        w = row_weights.to(device=logits_per_image.device, dtype=loss_i_per.dtype)
        w = w / w.mean().clamp(min=1e-8)
        loss_i = (loss_i_per * w).mean()
        loss_t = (loss_t_per * w).mean()

    return (loss_i + loss_t) / 2


# ─── CJKCLIP v3 ──────────────────────────────────────────────────────────────

class CJKCLIP(nn.Module):
    """
    v6: ViT patch mean pooling + DeBERTa CLS + 2-layer MLP projectors -> 1024-d shared space.
    """

    def __init__(
        self,
        image_encoder: ViTImageEncoder,
        text_encoder: DeBERTaTextEncoder,
        embed_dim: int = 1024,
        logit_scale_init: float = 0.07,
        learnable_logit_scale: bool = True,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        self.image_projector = MLPProjector(
            in_dim=image_encoder.output_dim,
            out_dim=embed_dim,
        )
        self.text_projector = MLPProjector(
            in_dim=text_encoder.output_dim,
            out_dim=embed_dim,
        )

        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1.0 / logit_scale_init)
        )
        if not learnable_logit_scale:
            self.logit_scale.requires_grad_(False)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        features = self.image_encoder(images)
        features = self.image_projector(features)
        return F.normalize(features, dim=-1)

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        features = self.text_encoder(text)
        features = self.text_projector(features)
        return F.normalize(features, dim=-1)

    def forward(self, images: torch.Tensor, text: torch.Tensor) -> dict:
        return {
            'image_features': self.encode_image(images),
            'text_features': self.encode_text(text),
            'logit_scale': self.logit_scale.exp(),
        }

    def get_params_summary(self) -> str:
        img = sum(p.numel() for p in self.image_encoder.parameters())
        txt = sum(p.numel() for p in self.text_encoder.parameters())
        ip = sum(p.numel() for p in self.image_projector.parameters())
        tp = sum(p.numel() for p in self.text_projector.parameters())
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return (
            f"Model Parameters (v6):\n"
            f"  Image encoder (ViT):      {img/1e6:.1f}M (out={self.image_encoder.output_dim})\n"
            f"  Text encoder (DeBERTa):    {txt/1e6:.1f}M (out={self.text_encoder.output_dim})\n"
            f"  Image projector (MLP):     {ip/1e6:.1f}M\n"
            f"  Text projector  (MLP):     {tp/1e6:.1f}M\n"
            f"  Total: {total/1e6:.1f}M (trainable: {trainable/1e6:.1f}M)"
        )


def build_model(
    ids_dict,
    model_size: str = 'base',
    pretrained: bool = False,
    embed_dim: int = 1024,
    text_context_length: int = 30,
    learnable_logit_scale: bool = True,
    img_size: int = 128,
    patch_size: int = 16,
) -> CJKCLIP:
    """Build v3 model.

    Args:
        model_size: 'base' (~175M) or 'large' (~612M)
        pretrained: use ImageNet-pretrained ViT (recommended True for ViT)
        embed_dim: shared projection space (default 1024)
        text_context_length: max IDS tokens (default 30)
        img_size: input image size (default 128)
        patch_size: ViT patch size (default 16 -> 8x8 = 64 tokens + CLS)
    """
    image_encoder = ViTImageEncoder(
        size=model_size, pretrained=pretrained,
        img_size=img_size, patch_size=patch_size,
    )
    text_encoder = DeBERTaTextEncoder(
        vocab_size=len(ids_dict.token2id),
        size=model_size,
        max_position_embeddings=text_context_length,
        pad_token_id=0,
    )
    model = CJKCLIP(
        image_encoder, text_encoder,
        embed_dim=embed_dim,
        learnable_logit_scale=learnable_logit_scale,
    )
    return model


def test_model():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ids_dictionary import IDSDictionary

    print("=" * 70)
    print("Testing v6 Model (base)")
    print("=" * 70)

    ids_dict = IDSDictionary.load(str(Path(__file__).resolve().parent / 'data'))
    model = build_model(ids_dict, model_size='base', pretrained=False)
    print(model.get_params_summary())

    B = 4
    images = torch.randn(B, 3, 128, 128)
    chars = ['辉', '想', '国', '的']
    text = torch.tensor([
        ids_dict.char_to_ids_with_special(c, max_len=30) for c in chars
    ], dtype=torch.long)

    out = model(images, text)
    img_feat = out['image_features']
    txt_feat = out['text_features']
    logit_scale = out['logit_scale']

    print(f"\nOutput shapes:")
    print(f"  image_features: {img_feat.shape}")
    print(f"  text_features:  {txt_feat.shape}")
    print(f"  logit_scale:    {logit_scale.item():.4f}")

    print(f"\nFeature norms:")
    print(f"  image: {[f'{n:.4f}' for n in img_feat.norm(dim=-1).tolist()]}")
    print(f"  text:  {[f'{n:.4f}' for n in txt_feat.norm(dim=-1).tolist()]}")

    logits_i2t = logit_scale * img_feat @ txt_feat.t()
    logits_t2i = logits_i2t.t()
    char_ids = torch.tensor([0, 0, 1, 2])
    li = li_loss(img_feat, char_ids)
    cl = compute_clip_loss_multipos(logits_i2t, logits_t2i, char_ids)
    print(f"\nLosses: clip={cl.item():.4f}, li={li.item():.4f}")

    # Backward check
    (cl + 0.5 * li).backward()
    print("Backward OK")

    print("\n" + "=" * 70)
    print("Testing v6 Model (large)")
    print("=" * 70)
    model_l = build_model(ids_dict, model_size='large', pretrained=False)
    print(model_l.get_params_summary())

    print("\n" + "=" * 70)
    print("v6 Model tests completed!")
    print("=" * 70)


if __name__ == '__main__':
    test_model()
