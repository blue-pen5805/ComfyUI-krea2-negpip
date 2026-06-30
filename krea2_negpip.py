# ComfyUI custom node: Krea 2 NegPiP
# Implementation module for ComfyUI/custom_nodes/ComfyUI-krea2-negpip.
#
# - CLIP/Qwen3-VL side enables negative prompt weights and emits a sidecar token.
# - DiT side strips the sidecar and stores active token positions in transformer_options.
# - Krea2 Attention.forward is patched lazily once per model object, not every sampler step.
# - Negative token V-flips are vectorized and cached per step/device.

from __future__ import annotations

import copy
import inspect
import math
import numbers
import types
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from einops import rearrange

from comfy import sd1_clip
import comfy.model_management as model_management
import comfy.patcher_extension
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import optimized_attention_masked


WRAPPER_KEY = "krea2_negpip"
KREA2_TEXT_ENCODER_KEY = "qwen3vl_4b"
KREA2_TAP_LAYERS = 12
KREA2_TAP_DIM = 2560
KREA2_FLAT_DIM = KREA2_TAP_LAYERS * KREA2_TAP_DIM
KREA2_SIDECAR_COPY_BASES = tuple(i * KREA2_TAP_DIM for i in range(KREA2_TAP_LAYERS))

IM_START_TOKEN = 151644
USER_TOKEN = 872
NEWLINE_TOKEN = 198
IMAGE_PAD_TOKEN = 151655
EMPTY_THINK_BLOCK = "<think>\n\n</think>\n\n"

# Sidecar marker values. Kept small so an incorrectly paired workflow is less destructive.
SIDECAR_MAGIC_A = 0.12345
SIDECAR_MAGIC_B = -0.23456
SIDECAR_MAGIC_C = 0.34567
SIDECAR_VERSION = 1
SIDECAR_HEADER = 16
SIDECAR_PAIR_WIDTH = 2  # token_index high/low chunks
SIDECAR_VALUE_SCALE = 1.0 / 4096.0
SIDECAR_MIN_SCALE = 1.0e-12
SIDECAR_MARKER_RTOL = 0.05
SIDECAR_CHECKSUM_MOD = 251
SIDECAR_CHECKSUM_TOL = 1
SIDECAR_POSITION_BASE = 128


@dataclass(frozen=True)
class TextBatchPlan:
    rows: list[Any]
    nonunit: int
    negative: int

    @property
    def has_weighted_tokens(self) -> bool:
        return self.nonunit > 0


def _is_plain_int_token(x: Any) -> bool:
    return (not torch.is_tensor(x)) and isinstance(x, numbers.Integral)


def _intermediate(x):
    if x is None:
        return None
    return x.to(model_management.intermediate_device())


def _bounded_float(value, default: float, lo: float, hi: float) -> float:
    try:
        v = float(value)
    except Exception:
        v = default
    if not math.isfinite(v):
        v = default
    return max(lo, min(hi, v))


def _bounded_int(value, default: int, lo: int, hi: int) -> int:
    try:
        v = int(value)
    except Exception:
        v = default
    return max(lo, min(hi, v))


def _find_krea2_user_prompt_start(tok_pairs, seq_len: int, template_end: int = -1) -> int:
    """Match ComfyUI's Krea2TEModel prefix stripping."""
    start = template_end
    auto_detected = start == -1
    if start == -1:
        count_im_start = 0
        for i, v in enumerate(tok_pairs):
            elem = v[0]
            if _is_plain_int_token(elem) and int(elem) == IM_START_TOKEN and count_im_start < 2:
                start = i
                count_im_start += 1
    if start < 0:
        return 0
    if auto_detected and seq_len > (start + 3) and len(tok_pairs) > (start + 2):
        if tok_pairs[start + 1][0] == USER_TOKEN and tok_pairs[start + 2][0] == NEWLINE_TOKEN:
            start += 3
    return max(0, min(start, seq_len))


def _krea2_tokenize_with_weights(self, text, return_word_ids=False, llama_template=None,
                                 images=None, prevent_empty_text=False, thinking=True, **kwargs):
    """
    Krea2Tokenizer delegates to Qwen3VLTokenizer; this keeps the same chat template
    but forces Comfy prompt weighting to remain enabled, including (token:-1.2).
    """
    if images is None:
        images = []

    image = kwargs.pop("image", None)
    kwargs.pop("disable_weights", None)

    if image is not None and len(images) == 0:
        images = [image[i:i + 1] for i in range(image.shape[0])]

    skip_template = text.startswith("<|im_start|>")
    if prevent_empty_text and text == "":
        text = " "

    if skip_template:
        llama_text = text
    else:
        if llama_template is not None:
            template = llama_template
        elif len(images) > 0 and hasattr(self, "llama_template_images"):
            template = self.llama_template_images
        else:
            template = self.llama_template

        if len(images) > 1:
            vision_block = "<|vision_start|><|image_pad|><|vision_end|>"
            template = template.replace(vision_block, vision_block * len(images), 1)
        llama_text = template.format(text)

    if not thinking:
        llama_text += EMPTY_THINK_BLOCK

    tokens = sd1_clip.SD1Tokenizer.tokenize_with_weights(
        self,
        llama_text,
        return_word_ids=return_word_ids,
        disable_weights=False,
        **kwargs,
    )

    key_name = next(iter(tokens))
    embed_count = 0
    for row in tokens[key_name]:
        for i in range(len(row)):
            if row[i][0] == IMAGE_PAD_TOKEN:
                if len(images) > embed_count:
                    row[i] = ({
                        "type": "image",
                        "data": images[embed_count],
                        "original_type": "image",
                    },) + row[i][1:]
                embed_count += 1
    return tokens


def _sidecar_copy_bases(feature_dim: int) -> tuple[int, ...]:
    """
    Store the metadata redundantly at the beginning of every Krea2 tap-layer slice.

    Conditioning Rebalance scales Krea2 conditioning as (B, seq, 12, 2560), so any
    metadata stored in the main tensor is multiplied by both the global multiplier
    and the per-layer gain.  By writing the same metadata to every layer slice and
    parsing it with a scale-invariant marker, the sidecar survives normal rebalance
    use unless the whole conditioning is multiplied by exactly zero.
    """
    if feature_dim == KREA2_FLAT_DIM:
        return KREA2_SIDECAR_COPY_BASES
    return (0,)


def _sidecar_capacity_for_base(feature_dim: int, base: int) -> int:
    if base < 0 or base + SIDECAR_HEADER >= feature_dim:
        return 0
    # Keep each redundant copy inside its own Krea2 tap-layer slice when possible.
    local_end = min(feature_dim, base + KREA2_TAP_DIM) if feature_dim == KREA2_FLAT_DIM else feature_dim
    return max(0, (local_end - base - SIDECAR_HEADER) // SIDECAR_PAIR_WIDTH)


def _sidecar_capacity(feature_dim: int) -> int:
    return max((_sidecar_capacity_for_base(feature_dim, base) for base in _sidecar_copy_bases(feature_dim)), default=0)


def _sidecar_checksum(positions: list[int]) -> int:
    total = 0
    for i, pos in enumerate(positions):
        total += (i + 1) * (int(pos) + 1)
    return total % SIDECAR_CHECKSUM_MOD


def _sidecar_checksum_matches(actual: int, expected: int) -> bool:
    delta = abs(int(actual) - int(expected))
    circular_delta = min(delta, SIDECAR_CHECKSUM_MOD - delta)
    return circular_delta <= SIDECAR_CHECKSUM_TOL


def _make_sidecar(cond: torch.Tensor, negative_positions: list[int]) -> torch.Tensor:
    if not negative_positions:
        return cond
    if cond.ndim != 3 or cond.shape[-1] < SIDECAR_HEADER + SIDECAR_PAIR_WIDTH:
        return cond

    max_capacity = _sidecar_capacity(cond.shape[-1])
    if max_capacity <= 0:
        return cond

    sidecars = []
    for chunk_start in range(0, len(negative_positions), max_capacity):
        pairs = negative_positions[chunk_start:chunk_start + max_capacity]
        sidecar = torch.zeros((cond.shape[0], 1, cond.shape[-1]), dtype=cond.dtype, device=cond.device)
        checksum = _sidecar_checksum(pairs)
        for base in _sidecar_copy_bases(cond.shape[-1]):
            max_pairs = _sidecar_capacity_for_base(cond.shape[-1], base)
            if max_pairs <= 0:
                continue
            copy_pairs = pairs[:max_pairs]
            sidecar[:, 0, base + 0] = SIDECAR_MAGIC_A
            sidecar[:, 0, base + 1] = SIDECAR_MAGIC_B
            sidecar[:, 0, base + 2] = float(len(copy_pairs)) * SIDECAR_VALUE_SCALE
            sidecar[:, 0, base + 3] = float(cond.shape[1] // SIDECAR_POSITION_BASE) * SIDECAR_VALUE_SCALE
            sidecar[:, 0, base + 4] = SIDECAR_MAGIC_C
            sidecar[:, 0, base + 5] = float(SIDECAR_VERSION) * SIDECAR_VALUE_SCALE
            sidecar[:, 0, base + 6] = float(checksum) * SIDECAR_VALUE_SCALE
            sidecar[:, 0, base + 7] = float(cond.shape[1] % SIDECAR_POSITION_BASE) * SIDECAR_VALUE_SCALE

            if copy_pairs:
                offsets = base + SIDECAR_HEADER + torch.arange(len(copy_pairs), device=cond.device) * SIDECAR_PAIR_WIDTH
                values = torch.tensor(copy_pairs, dtype=torch.long, device=cond.device)
                high = (values // SIDECAR_POSITION_BASE).to(dtype=cond.dtype) * SIDECAR_VALUE_SCALE
                low = (values % SIDECAR_POSITION_BASE).to(dtype=cond.dtype) * SIDECAR_VALUE_SCALE
                sidecar[:, 0, offsets] = high
                sidecar[:, 0, offsets + 1] = low
        sidecars.append(sidecar)

    return torch.cat([cond, *sidecars], dim=1)


def _pair_weight(pair) -> float | None:
    if len(pair) < 2:
        return None
    try:
        return float(pair[1])
    except Exception:
        return None


def _embedded_token_width(token: Any) -> int | None:
    if _is_plain_int_token(token):
        return 1
    if torch.is_tensor(token):
        if token.ndim == 0:
            return 1
        return int(token.reshape(-1, token.shape[-1]).shape[0])
    if isinstance(token, dict) and token.get("type") == "embedding" and torch.is_tensor(token.get("data")):
        data = token["data"]
        if data.ndim == 0:
            return 1
        return int(data.reshape(-1, data.shape[-1]).shape[0])
    return None


def _expanded_token_index_map(section, seq_len: int) -> tuple[list[int | None], bool]:
    """
    Map tokenizer pair indexes to encoded hidden-state indexes.

    Comfy expands tensor embeddings and image placeholders inside SDClipModel.process_tokens.
    Tensor embedding widths are visible here; image widths depend on preprocessing, so positions
    after an image placeholder are treated as unknown instead of risking a wrong V flip.
    """
    index_map: list[int | None] = []
    encoded_index = 0
    reliable = True

    for entry in section:
        width = _embedded_token_width(entry[0])
        if width is None or not reliable:
            index_map.append(None)
            reliable = False
            continue

        index_map.append(encoded_index if encoded_index < seq_len else None)
        encoded_index += width

    return index_map, reliable


def _krea2_empty_tokens(clip_model, token_count: int):
    if hasattr(clip_model, "gen_empty_tokens"):
        return clip_model.gen_empty_tokens(clip_model.special_tokens, token_count)
    return sd1_clip.gen_empty_tokens(clip_model.special_tokens, token_count)


def _prepare_krea2_text_batch(pair_sections, clip_model):
    token_rows = []
    max_width = 0
    nonunit = 0
    negative = 0
    image_positions: set[int] = set()

    for section in pair_sections:
        row = [entry[0] for entry in section]
        token_rows.append(row)
        max_width = max(max_width, len(section))
        for i, entry in enumerate(section):
            weight = _pair_weight(entry)
            if isinstance(row[i], dict) and row[i].get("type") == "image":
                image_positions.add(i)
            if weight is None or weight == 1.0:
                continue
            nonunit += 1
            negative += int(weight < 0)

    needs_reference_row = nonunit > 0 or len(token_rows) == 0
    if needs_reference_row:
        if image_positions:
            # Preserve image entries in the reference row so that
            # the encoded sequence length matches across the batch.
            reference = _krea2_empty_tokens(clip_model, max_width)
            for pos in sorted(image_positions):
                if pos < len(reference):
                    for row in token_rows:
                        if pos < len(row) and isinstance(row[pos], dict) and row[pos].get("type") == "image":
                            reference[pos] = row[pos]
                            break
            token_rows.append(reference)
        else:
            token_rows.append(_krea2_empty_tokens(clip_model, max_width))

    return TextBatchPlan(
        rows=token_rows,
        nonunit=nonunit,
        negative=negative,
    )


def _flatten_krea2_taps(tensor: torch.Tensor) -> torch.Tensor:
    batch, layers, seq_len, width = tensor.shape
    return tensor.permute(0, 2, 1, 3).reshape(batch, seq_len, layers * width)


def _apply_krea2_token_magnitudes(
    sample: torch.Tensor,
    reference: torch.Tensor,
    section,
    visible_start: int,
    output_offset: int,
) -> tuple[torch.Tensor, list[int]]:
    encoded_indices = []
    magnitudes = []
    negative_positions: list[int] = []
    index_map, _ = _expanded_token_index_map(section, min(sample.shape[2], reference.shape[2]))
    token_limit = min(len(index_map), len(section))

    for token_index in range(token_limit):
        weight = _pair_weight(section[token_index])
        if weight is None or weight == 1.0:
            continue

        encoded_index = index_map[token_index]
        if encoded_index is None:
            continue

        encoded_indices.append(encoded_index)
        magnitudes.append(abs(weight))

        if weight < 0 and encoded_index >= visible_start:
            negative_positions.append(output_offset + encoded_index - visible_start)

    if not encoded_indices:
        return sample, negative_positions

    changed = sample.clone()
    idx = torch.tensor(encoded_indices, device=sample.device, dtype=torch.long)
    scale = torch.tensor(magnitudes, device=sample.device, dtype=sample.dtype).view(1, 1, -1, 1)
    changed[:, :, idx, :] = torch.lerp(reference[:, :, idx, :], changed[:, :, idx, :], scale)
    return changed, negative_positions


def _build_krea2_conditioning(encoded, pair_sections, has_weighted_tokens: bool, template_end: int):
    raw_context = encoded[0]  # Krea2 raw: (B, 12, seq, 2560)
    if len(pair_sections) == 0:
        return _make_sidecar(_flatten_krea2_taps(raw_context[-1:].clone()), []), []

    reference = raw_context[-1:] if has_weighted_tokens else None
    flattened_sections = []
    negative_positions: list[int] = []
    running_len = 0

    for section_index, section in enumerate(pair_sections):
        current = raw_context[section_index:section_index + 1]
        visible_start = _find_krea2_user_prompt_start(section, current.shape[2], template_end)

        if reference is not None:
            current, row_positions = _apply_krea2_token_magnitudes(
                current,
                reference,
                section,
                visible_start,
                running_len,
            )
            negative_positions.extend(row_positions)

        visible = current[:, :, visible_start:]
        flattened = _flatten_krea2_taps(visible)
        flattened_sections.append(flattened)
        running_len += flattened.shape[1]

    if len(flattened_sections) == 1:
        merged = flattened_sections[0]
    else:
        merged = torch.cat(flattened_sections, dim=1)
    return _make_sidecar(merged, negative_positions), negative_positions


def _build_krea2_extra(encoded, pair_sections, template_end: int, cond_seq_len: int):
    if len(encoded) <= 2 or not isinstance(encoded[2], dict):
        return {}

    extra = dict(encoded[2])
    attention_mask = extra.get("attention_mask")
    if not torch.is_tensor(attention_mask):
        extra.pop("attention_mask", None)
        return extra

    raw_context = encoded[0]
    mask_sections = []
    base_len = 0

    if len(pair_sections) == 0:
        mask = attention_mask[-1:, :raw_context.shape[2]]
        mask_sections.append(mask)
        base_len = int(mask.shape[1])
    else:
        for section_index, section in enumerate(pair_sections):
            visible_start = _find_krea2_user_prompt_start(section, raw_context.shape[2], template_end)
            mask = attention_mask[section_index:section_index + 1, visible_start:raw_context.shape[2]]
            mask_sections.append(mask)
            base_len += int(mask.shape[1])

    if not mask_sections:
        extra.pop("attention_mask", None)
        return extra

    merged_mask = mask_sections[0] if len(mask_sections) == 1 else torch.cat(mask_sections, dim=1)
    sidecar_tokens = max(0, int(cond_seq_len) - base_len)
    if sidecar_tokens > 0:
        sidecar_mask = torch.zeros(
            (merged_mask.shape[0], sidecar_tokens),
            dtype=merged_mask.dtype,
            device=merged_mask.device,
        )
        merged_mask = torch.cat((merged_mask, sidecar_mask), dim=1)

    if merged_mask.sum() == torch.numel(merged_mask):
        extra.pop("attention_mask", None)
    else:
        extra["attention_mask"] = _intermediate(merged_mask)
    return extra


def _make_krea2_negpip_encode_token_weights(cond_stage_model):
    original_encode = getattr(
        cond_stage_model,
        "_krea2_negpip_original_encode_token_weights",
        cond_stage_model.encode_token_weights,
    )

    def encode_token_weights(token_weight_pairs, template_end=-1):
        if KREA2_TEXT_ENCODER_KEY not in token_weight_pairs or not hasattr(cond_stage_model, KREA2_TEXT_ENCODER_KEY):
            try:
                return original_encode(token_weight_pairs, template_end=template_end)
            except TypeError:
                return original_encode(token_weight_pairs)

        clip_model = getattr(cond_stage_model, KREA2_TEXT_ENCODER_KEY)
        pairs = token_weight_pairs[KREA2_TEXT_ENCODER_KEY]
        plan = _prepare_krea2_text_batch(pairs, clip_model)
        encoded = clip_model.encode(plan.rows)
        pooled = encoded[1]
        first_pooled = _intermediate(pooled[0:1]) if pooled is not None else None
        cond, negative_positions = _build_krea2_conditioning(
            encoded,
            pairs,
            plan.has_weighted_tokens,
            template_end,
        )
        extra = _build_krea2_extra(encoded, pairs, template_end, cond.shape[1])

        return _intermediate(cond), first_pooled, extra

    return encode_token_weights


def _patch_clip_for_krea2_negpip(clip):
    cond_stage_model = getattr(clip, "cond_stage_model", None)
    if cond_stage_model is None or not hasattr(cond_stage_model, KREA2_TEXT_ENCODER_KEY):
        raise RuntimeError("Krea2 NegPiP requires CLIPLoader type='krea2'.")

    new_clip = clip.clone() if hasattr(clip, "clone") else copy.copy(clip)
    new_clip.tokenizer = copy.copy(clip.tokenizer)
    new_clip.cond_stage_model = copy.copy(cond_stage_model)

    new_clip.tokenizer.tokenize_with_weights = types.MethodType(
        _krea2_tokenize_with_weights,
        new_clip.tokenizer,
    )

    patched_cond_stage_model = new_clip.cond_stage_model

    if not hasattr(patched_cond_stage_model, "_krea2_negpip_original_encode_token_weights"):
        patched_cond_stage_model._krea2_negpip_original_encode_token_weights = patched_cond_stage_model.encode_token_weights
    patched_cond_stage_model.encode_token_weights = _make_krea2_negpip_encode_token_weights(patched_cond_stage_model)
    return new_clip


def _is_krea2_dm(dm: Any) -> bool:
    return (
        hasattr(dm, "txtfusion")
        and hasattr(dm, "txtmlp")
        and hasattr(dm, "blocks")
        and hasattr(dm, "_unpack_context")
        and int(getattr(dm, "txtlayers", 0)) == KREA2_TAP_LAYERS
        and int(getattr(dm, "txtdim", 0)) == KREA2_TAP_DIM
    )


def _decode_sidecar_row(tail_row: torch.Tensor) -> tuple[list[int], float, int | None] | None:
    """Return (positions, recovered_scale, source_length) for one batch row."""
    feature_dim = int(tail_row.shape[0])
    best = None

    for base in _sidecar_copy_bases(feature_dim):
        max_pairs = _sidecar_capacity_for_base(feature_dim, base)
        if max_pairs <= 0 or base + SIDECAR_HEADER >= feature_dim:
            continue

        a = float(tail_row[base + 0].item())
        b = float(tail_row[base + 1].item())
        c = float(tail_row[base + 4].item())
        if not (math.isfinite(a) and math.isfinite(b) and math.isfinite(c)):
            continue

        scale_a = a / SIDECAR_MAGIC_A
        scale_b = b / SIDECAR_MAGIC_B
        scale_c = c / SIDECAR_MAGIC_C
        denom = max(abs(scale_a), abs(scale_b), SIDECAR_MIN_SCALE)
        if abs(scale_a) < SIDECAR_MIN_SCALE or abs(scale_b) < SIDECAR_MIN_SCALE or abs(scale_c) < SIDECAR_MIN_SCALE:
            continue
        if abs(scale_a - scale_b) / denom > SIDECAR_MARKER_RTOL:
            continue
        if abs(scale_a - scale_c) / max(denom, abs(scale_c)) > SIDECAR_MARKER_RTOL:
            continue

        scale = (scale_a + scale_b + scale_c) / 3.0
        if abs(scale) < SIDECAR_MIN_SCALE or not math.isfinite(scale):
            continue

        try:
            n = int(round(float(tail_row[base + 2].item()) / (scale * SIDECAR_VALUE_SCALE)))
        except Exception:
            continue
        if n < 0 or n > max_pairs:
            continue

        try:
            source_high = int(round(float(tail_row[base + 3].item()) / (scale * SIDECAR_VALUE_SCALE)))
            source_low = int(round(float(tail_row[base + 7].item()) / (scale * SIDECAR_VALUE_SCALE)))
            source_length = source_high * SIDECAR_POSITION_BASE + source_low
        except Exception:
            source_length = None
        if source_length is not None and (
            source_length <= 0
            or source_high < 0
            or source_low < 0
            or source_low >= SIDECAR_POSITION_BASE
        ):
            source_length = None

        try:
            version = int(round(float(tail_row[base + 5].item()) / (scale * SIDECAR_VALUE_SCALE)))
        except Exception:
            continue
        if version != SIDECAR_VERSION:
            continue

        try:
            expected_checksum = int(round(float(tail_row[base + 6].item()) / (scale * SIDECAR_VALUE_SCALE)))
        except Exception:
            continue
        if expected_checksum < 0 or expected_checksum >= SIDECAR_CHECKSUM_MOD:
            continue

        row: list[int] = []
        valid = True
        for i in range(n):
            off = base + SIDECAR_HEADER + i * SIDECAR_PAIR_WIDTH
            try:
                high = int(round(float(tail_row[off].item()) / (scale * SIDECAR_VALUE_SCALE)))
                low = int(round(float(tail_row[off + 1].item()) / (scale * SIDECAR_VALUE_SCALE)))
            except Exception:
                valid = False
                break
            if high < 0 or low < 0 or low >= SIDECAR_POSITION_BASE:
                valid = False
                break
            pos = high * SIDECAR_POSITION_BASE + low
            if pos < 0 or (source_length is not None and pos >= source_length):
                valid = False
                break
            row.append(pos)
        if not valid:
            continue
        if not _sidecar_checksum_matches(_sidecar_checksum(row), expected_checksum):
            continue

        candidate = (row, scale, source_length)
        # Prefer the least numerically tiny surviving copy.
        if best is None or abs(candidate[1]) > abs(best[1]):
            best = candidate

    return best


def _parse_sidecar_token(token_rows: torch.Tensor) -> tuple[list[list[int]], list[float], list[int | None], bool]:
    positions: list[list[int]] = []
    scales: list[float] = []
    source_lengths: list[int | None] = []
    found = False

    for b in range(token_rows.shape[0]):
        decoded = _decode_sidecar_row(token_rows[b])
        if decoded is None:
            positions.append([])
            scales.append(float("nan"))
            source_lengths.append(None)
            continue
        row, scale, source_length = decoded
        positions.append(row)
        scales.append(scale)
        source_lengths.append(source_length)
        found = True

    return positions, scales, source_lengths, found


def _sidecar_candidate_indices(context: torch.Tensor) -> set[int]:
    if context.ndim != 3 or context.shape[1] < 1:
        return set()

    seq_len = int(context.shape[1])
    feature_dim = int(context.shape[2])
    candidates = torch.zeros(seq_len, dtype=torch.bool, device=context.device)
    detached = context.detach()

    for base in _sidecar_copy_bases(feature_dim):
        if _sidecar_capacity_for_base(feature_dim, base) <= 0 or base + SIDECAR_HEADER >= feature_dim:
            continue

        a = detached[:, :, base + 0].float()
        b = detached[:, :, base + 1].float()
        c = detached[:, :, base + 4].float()

        scale_a = a / SIDECAR_MAGIC_A
        scale_b = b / SIDECAR_MAGIC_B
        scale_c = c / SIDECAR_MAGIC_C
        denom = torch.maximum(
            torch.maximum(scale_a.abs(), scale_b.abs()),
            torch.full_like(scale_a, SIDECAR_MIN_SCALE),
        )

        valid = (
            torch.isfinite(scale_a)
            & torch.isfinite(scale_b)
            & torch.isfinite(scale_c)
            & (scale_a.abs() >= SIDECAR_MIN_SCALE)
            & (scale_b.abs() >= SIDECAR_MIN_SCALE)
            & (scale_c.abs() >= SIDECAR_MIN_SCALE)
            & (((scale_a - scale_b).abs() / denom) <= SIDECAR_MARKER_RTOL)
            & (((scale_a - scale_c).abs() / torch.maximum(denom, scale_c.abs())) <= SIDECAR_MARKER_RTOL)
        )
        candidates |= valid.any(dim=0)

    if not bool(candidates.any().item()):
        return set()
    return set(torch.nonzero(candidates, as_tuple=False).flatten().tolist())


def _parse_and_strip_sidecar_full(
    context: torch.Tensor,
) -> tuple[torch.Tensor, list[list[int]] | None, list[float] | None, list[int] | None]:
    if context.ndim != 3 or context.shape[1] < 1 or context.shape[2] < SIDECAR_HEADER:
        return context, None, None, None

    merged_positions: list[list[int]] = [[] for _ in range(context.shape[0])]
    scales: list[float] = []
    keep_indices: list[int] = []
    segment_start = 0
    found_any = False
    candidate_indices = _sidecar_candidate_indices(context)
    if not candidate_indices:
        return context, None, None, None

    for i in range(context.shape[1]):
        if i not in candidate_indices:
            keep_indices.append(i)
            continue

        positions, token_scales, source_lengths, found = _parse_sidecar_token(context[:, i, :].detach())
        if not found:
            keep_indices.append(i)
            continue

        known_lengths = [v for v in source_lengths if v is not None]
        if known_lengths:
            source_length = max(known_lengths)
            current_segment_start = max(0, len(keep_indices) - source_length)
        else:
            current_segment_start = segment_start

        for b, row in enumerate(positions):
            merged_positions[b].extend(current_segment_start + pos for pos in row)
        scales.extend(token_scales)
        found_any = True
        segment_start = len(keep_indices)

    if not found_any:
        return context, None, None, None

    if not keep_indices:
        stripped = context[:, :0, :]
    elif len(keep_indices) == context.shape[1]:
        stripped = context
    else:
        idx = torch.tensor(keep_indices, device=context.device, dtype=torch.long)
        stripped = context.index_select(1, idx)

    return stripped, merged_positions, scales, keep_indices


def _parse_and_strip_sidecar(context: torch.Tensor) -> tuple[torch.Tensor, list[list[int]] | None, list[float] | None]:
    stripped, positions, scales, _ = _parse_and_strip_sidecar_full(context)
    return stripped, positions, scales


def _strip_mask_with_indices(attention_mask: Any, keep_indices: list[int] | None, original_context_len: int):
    if attention_mask is None or keep_indices is None or not torch.is_tensor(attention_mask):
        return attention_mask
    if attention_mask.ndim < 1 or int(attention_mask.shape[-1]) != int(original_context_len):
        if attention_mask.ndim >= 1 and int(attention_mask.shape[-1]) == len(keep_indices):
            return attention_mask
        if attention_mask.ndim >= 1 and int(attention_mask.shape[-1]) < int(original_context_len):
            # ConditioningConcat keeps only conditioning_to extras. Preserve the known prefix mask
            # and treat the concat suffix as valid text tokens after sidecar stripping.
            prefix_len = int(attention_mask.shape[-1])
            kept_prefix_indices = [i for i in keep_indices if i < prefix_len]
            suffix_count = sum(1 for i in keep_indices if i >= prefix_len)

            pieces = []
            if kept_prefix_indices:
                idx = torch.tensor(kept_prefix_indices, device=attention_mask.device, dtype=torch.long)
                pieces.append(attention_mask.index_select(attention_mask.ndim - 1, idx))
            if suffix_count > 0:
                suffix_shape = list(attention_mask.shape)
                suffix_shape[-1] = suffix_count
                pieces.append(torch.ones(suffix_shape, dtype=attention_mask.dtype, device=attention_mask.device))
            if pieces:
                return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=attention_mask.ndim - 1)
        return None
    idx = torch.tensor(keep_indices, device=attention_mask.device, dtype=torch.long)
    return attention_mask.index_select(attention_mask.ndim - 1, idx)


def _get_index_tensors_for_v(v: torch.Tensor, cfg: dict[str, Any]):
    positions = cfg.get("_negative_positions")
    value_strength = float(cfg.get("value_strength", 1.0))
    if not positions or value_strength == 0.0:
        return None

    cache = cfg.setdefault("_index_cache", {})
    key = (v.device, v.shape[0], v.shape[2], v.dtype, value_strength)
    cached = cache.get(key)
    if cached is not None:
        return cached

    bsz = int(v.shape[0])
    seq_len = int(v.shape[2])
    src_rows = len(positions)

    batch_ids = []
    pos_ids = []
    multipliers = []
    for b in range(bsz):
        row = positions[b if src_rows == bsz else (b % src_rows)]
        for pos in row:
            if 0 <= pos < seq_len:
                batch_ids.append(b)
                pos_ids.append(int(pos))
                multipliers.append(-value_strength)

    if not batch_ids:
        cache[key] = None
        return None

    bi = torch.tensor(batch_ids, device=v.device, dtype=torch.long)
    pi = torch.tensor(pos_ids, device=v.device, dtype=torch.long)
    mul = torch.tensor(multipliers, device=v.device, dtype=v.dtype).view(-1, 1, 1)
    cached = (bi, pi, mul)
    cache[key] = cached
    return cached


def _apply_v_flip_inplace(v: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    tensors = _get_index_tensors_for_v(v, cfg)
    if tensors is None:
        return v
    bi, pi, mul = tensors
    # v: B,H,L,D. Advanced-index only the affected token rows instead of materializing a BxL scale tensor.
    v[bi, :, pi, :] = v[bi, :, pi, :] * mul
    return v


def _attention_signature_is_compatible(attn: Any) -> bool:
    try:
        params = inspect.signature(attn.forward).parameters
    except Exception:
        return False

    if "x" not in params and "hidden_states" not in params:
        return False

    known = {
        "x",
        "hidden_states",
        "freqs",
        "freqs_cis",
        "mask",
        "attention_mask",
        "attn_mask",
        "transformer_options",
    }
    for name, param in params.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param.default is inspect.Parameter.empty and name not in known:
            return False
    return True


def _attention_has_expected_krea2_shape(attn: Any) -> bool:
    required = ("wq", "wk", "wv", "gate", "qknorm", "wo", "heads", "kvheads")
    if not all(hasattr(attn, name) for name in required):
        return False
    if not _attention_signature_is_compatible(attn):
        return False
    try:
        heads = int(attn.heads)
        kvheads = int(attn.kvheads)
    except Exception:
        return False
    return heads > 0 and kvheads > 0 and heads % kvheads == 0


def _extract_attention_call(signature: inspect.Signature, args, kwargs):
    bound = signature.bind_partial(*args, **kwargs)
    arguments = bound.arguments
    x = arguments.get("x", arguments.get("hidden_states", None))
    freqs = arguments.get("freqs", arguments.get("freqs_cis", None))
    mask = arguments.get("mask", arguments.get("attention_mask", arguments.get("attn_mask", None)))
    transformer_options = arguments.get("transformer_options", None)
    if transformer_options is None:
        transformer_options = {}
    return x, freqs, mask, transformer_options


def _make_static_attention_forward(attn_module, original_forward, role: str, block_index: int | None):
    original_signature = inspect.signature(original_forward)

    def forward(self, *args, **kwargs):
        try:
            x, freqs, mask, transformer_options = _extract_attention_call(original_signature, args, kwargs)
        except TypeError:
            return original_forward(*args, **kwargs)

        cfg = transformer_options.get(WRAPPER_KEY, None) if transformer_options is not None else None
        if not cfg or not cfg.get("_active", False):
            return original_forward(*args, **kwargs)

        if role == "main":
            start = int(cfg.get("block_start", 0))
            end = int(cfg.get("block_end", 9999))
            stride = max(1, int(cfg.get("block_stride", 1)))
            idx = int(block_index or 0)
            if idx < start or idx > end or ((idx - start) % stride) != 0:
                return original_forward(*args, **kwargs)
        elif role == "txtfusion_refiner":
            if not bool(cfg.get("patch_txtfusion_refiners", False)):
                return original_forward(*args, **kwargs)
        else:
            return original_forward(*args, **kwargs)

        if x is None:
            return original_forward(*args, **kwargs)

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        gate = self.gate(x)

        q = rearrange(q, "B L (H D) -> B H L D", H=self.heads)
        k = rearrange(k, "B L (H D) -> B H L D", H=self.kvheads)
        v = rearrange(v, "B L (H D) -> B H L D", H=self.kvheads)

        v = _apply_v_flip_inplace(v, cfg)

        q, k = self.qknorm(q, k)
        if freqs is not None:
            q, k = apply_rope(q, k, freqs)
        if self.kvheads != self.heads:
            rep = self.heads // self.kvheads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        out = optimized_attention_masked(
            q, k, v, self.heads, mask=mask, skip_reshape=True,
            transformer_options=transformer_options,
        )
        return self.wo(out * F.sigmoid(gate))

    return types.MethodType(forward, attn_module)


def _patch_attention_once(attn, role: str, block_index: int | None = None):
    if getattr(attn, "_krea2_negpip_static_patched", False):
        return False
    if not _attention_has_expected_krea2_shape(attn):
        raise RuntimeError("Krea2 NegPiP cannot patch an unexpected Krea2 attention layout.")
    original = attn.forward
    attn._krea2_negpip_original_forward = original
    attn.forward = _make_static_attention_forward(attn, original, role, block_index)
    attn._krea2_negpip_static_patched = True
    return True


def _ensure_static_model_patches(dm: Any):
    if getattr(dm, "_krea2_negpip_model_patched", False):
        return

    for i, block in enumerate(getattr(dm, "blocks", [])):
        if hasattr(block, "attn"):
            _patch_attention_once(block.attn, "main", i)

    # Patch refiner blocks too, but they stay on the original forward unless the node option is enabled.
    txtfusion = getattr(dm, "txtfusion", None)
    if txtfusion is not None and hasattr(txtfusion, "refiner_blocks"):
        for i, block in enumerate(txtfusion.refiner_blocks):
            if hasattr(block, "attn"):
                _patch_attention_once(block.attn, "txtfusion_refiner", i)

    dm._krea2_negpip_model_patched = True


def krea2_negpip_wrapper(executor, x, timesteps, context, attention_mask=None, transformer_options=None, **kwargs):
    transformer_options = transformer_options or {}
    cfg = transformer_options.get(WRAPPER_KEY, {})
    if not cfg or not cfg.get("enabled", True):
        return executor(x, timesteps, context, attention_mask, transformer_options, **kwargs)

    dm = executor.class_obj
    if not _is_krea2_dm(dm):
        context_without_sidecar, negative_positions, _, keep_indices = _parse_and_strip_sidecar_full(context)
        attention_mask = _strip_mask_with_indices(attention_mask, keep_indices, context.shape[1])
        if negative_positions is not None:
            raise RuntimeError("Krea2 NegPiP conditioning was connected to a non-Krea2 diffusion model.")
        return executor(x, timesteps, context_without_sidecar, attention_mask, transformer_options, **kwargs)

    _ensure_static_model_patches(dm)

    original_context_len = context.shape[1]
    context, negative_positions, _, keep_indices = _parse_and_strip_sidecar_full(context)
    attention_mask = _strip_mask_with_indices(attention_mask, keep_indices, original_context_len)
    if negative_positions is None:
        return executor(x, timesteps, context, attention_mask, transformer_options, **kwargs)

    total_neg = sum(len(r) for r in negative_positions)
    value_strength = _bounded_float(cfg.get("value_strength", 1.0), 1.0, 0.0, 8.0)
    if total_neg == 0 or value_strength == 0.0:
        return executor(x, timesteps, context, attention_mask, transformer_options, **kwargs)

    # Keep the original transformer_options shape but install a per-call active cfg.
    # The patched Attention.forward reads this, so no per-step monkey-patching is needed.
    new_transformer_options = transformer_options.copy()
    active_cfg = dict(cfg)
    active_cfg["_active"] = True
    active_cfg["_negative_positions"] = negative_positions
    active_cfg["_index_cache"] = {}
    active_cfg["value_strength"] = value_strength
    new_transformer_options[WRAPPER_KEY] = active_cfg

    return executor(x, timesteps, context, attention_mask, new_transformer_options, **kwargs)


class ApplyKrea2NegPiP:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "value_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05}),
                "patch_txtfusion_refiners": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "block_start": ("INT", {"default": 0, "min": 0, "max": 999, "step": 1}),
                "block_end": ("INT", {"default": 27, "min": 0, "max": 999, "step": 1}),
                "block_stride": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "apply"
    CATEGORY = "loaders"

    def apply(self, model, clip, value_strength=1.0, patch_txtfusion_refiners=False,
              block_start=0, block_end=27, block_stride=1):
        new_clip = _patch_clip_for_krea2_negpip(clip)
        patched = model.clone()

        value_strength = _bounded_float(value_strength, 1.0, 0.0, 8.0)
        block_start = _bounded_int(block_start, 0, 0, 999)
        block_end = _bounded_int(block_end, 27, 0, 999)
        block_stride = _bounded_int(block_stride, 1, 1, 16)
        if block_end < block_start:
            block_start, block_end = block_end, block_start

        to = patched.model_options.setdefault("transformer_options", {})
        to[WRAPPER_KEY] = {
            "enabled": True,
            "value_strength": value_strength,
            "patch_txtfusion_refiners": bool(patch_txtfusion_refiners),
            "block_start": block_start,
            "block_end": block_end,
            "block_stride": block_stride,
        }

        if hasattr(patched, "remove_wrappers_with_key"):
            patched.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY)

        wrappers = to.get("wrappers", {})
        diffusion_wrappers = wrappers.get(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, {})
        diffusion_wrappers.pop(WRAPPER_KEY, None)

        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            WRAPPER_KEY,
            krea2_negpip_wrapper,
        )

        return patched, new_clip


NODE_CLASS_MAPPINGS = {
    "ApplyKrea2NegPiP": ApplyKrea2NegPiP,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ApplyKrea2NegPiP": "Apply Krea2 NegPiP",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
