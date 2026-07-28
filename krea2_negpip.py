# ComfyUI custom node: Krea 2 NegPiP
# Implementation module for ComfyUI/custom_nodes/ComfyUI-krea2-negpip.
#
# - CLIP/Qwen3-VL side enables negative prompt weights and emits a sidecar token.
# - DiT side strips the sidecar, or falls back to conditioning extra metadata when available.
# - Krea2 block/wv forwards are patched only for the active diffusion call and restored immediately.
# - Negative token V-flips are vectorized and cached per call/device.

from __future__ import annotations

import copy
import inspect
import math
import numbers
import types
from dataclasses import dataclass
from typing import Any

import torch

from comfy import sd1_clip
import comfy.model_management as model_management
import comfy.samplers
import comfy.patcher_extension
import comfy.text_encoders.qwen_vl


WRAPPER_KEY = "krea2_negpip"
NEGATIVE_POSITIONS_EXTRA_KEY = "krea2_negpip_negative_positions"
NEGATIVE_SOURCE_LENGTH_EXTRA_KEY = "krea2_negpip_source_length"
NEGATIVE_SIDECAR_TOKENS_EXTRA_KEY = "krea2_negpip_sidecar_tokens"
NEGATIVE_METADATA_BY_UUID_KEY = "_negative_metadata_by_uuid"
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
_IMAGE_TOKEN_WIDTH_CACHE: dict[tuple[int, int], int | None] = {}


@dataclass(frozen=True)
class TextBatchPlan:
    rows: list[Any]
    reference_indices: list[int | None]


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

    if not thinking and not skip_template:
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

    Simple Krea2 rebalance/scaling nodes can multiply conditioning as (B, seq, 12, 2560).
    By writing the same metadata to every layer slice and parsing it with a scale-invariant
    marker, the sidecar survives simple nonzero scale-only edits.  More complex tensor
    transforms such as normalization, averaging, or clamping can still destroy it.
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
    if isinstance(token, dict) and token.get("type") == "image" and torch.is_tensor(token.get("data")):
        return _qwen3vl_image_token_width(token["data"])
    return None


def _qwen3vl_image_token_width(image: torch.Tensor) -> int | None:
    """Estimate Qwen3-VL image placeholder expansion from ComfyUI's image grid helper."""
    if image.ndim != 4 or image.shape[1] <= 0 or image.shape[2] <= 0:
        return None

    cache_key = (int(image.shape[1]), int(image.shape[2]))
    if cache_key in _IMAGE_TOKEN_WIDTH_CACHE:
        return _IMAGE_TOKEN_WIDTH_CACHE[cache_key]

    try:
        _, grid = comfy.text_encoders.qwen_vl.process_qwen2vl_images(
            image,
            patch_size=16,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
        )
        merge_size = 2
        grid_t = int(grid[0][0].item())
        grid_h = int(grid[0][1].item())
        grid_w = int(grid[0][2].item())
        width = max(1, grid_t * (grid_h // merge_size) * (grid_w // merge_size))
    except Exception:
        width = None

    if len(_IMAGE_TOKEN_WIDTH_CACHE) > 128:
        _IMAGE_TOKEN_WIDTH_CACHE.clear()
    _IMAGE_TOKEN_WIDTH_CACHE[cache_key] = width
    return width


def _expanded_token_index_map(section, seq_len: int) -> tuple[list[int | None], bool]:
    """
    Map tokenizer pair indexes to encoded hidden-state indexes.

    Comfy expands tensor embeddings and image placeholders inside SDClipModel.process_tokens.
    Krea2/Qwen3-VL image widths are derived from the same resize/grid math as the vision
    preprocessor, so TextEncodeKrea2's default image-before-prompt layout can still carry
    negative weights on the text that follows the image.
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


def _section_has_nonunit_weights(section) -> bool:
    return any((_pair_weight(entry) not in (None, 1.0)) for entry in section)


def _make_krea2_reference_row(section, clip_model, max_width: int):
    """
    Build the neutral row used for Comfy-style prompt weighting.

    For text-only prompts this preserves the previous behavior exactly: a plain empty
    token row.  For vision prompts, a fully empty row has a different expanded length
    because it lacks image embeddings, so keep the image/embedding placeholders in the
    neutral row and replace plain text tokens with the tokenizer's empty/pad tokens.
    """
    empty = _krea2_empty_tokens(clip_model, max(len(section), max_width))
    row = []
    for i, entry in enumerate(section):
        token = entry[0]
        row.append(empty[i] if _is_plain_int_token(token) else token)
    if len(row) < max_width:
        row.extend(empty[len(row):max_width])
    return row


def _prepare_krea2_text_batch(pair_sections, clip_model):
    token_rows = []
    reference_indices: list[int | None] = []
    max_width = 0

    for section in pair_sections:
        token_rows.append([entry[0] for entry in section])
        reference_indices.append(None)
        max_width = max(max_width, len(section))

    for i, section in enumerate(pair_sections):
        if _section_has_nonunit_weights(section):
            reference_indices[i] = len(token_rows)
            token_rows.append(_make_krea2_reference_row(section, clip_model, max_width))

    if len(token_rows) == 0:
        token_rows.append(_krea2_empty_tokens(clip_model, max_width))

    return TextBatchPlan(
        rows=token_rows,
        reference_indices=reference_indices,
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


def _build_krea2_conditioning(encoded, pair_sections, reference_indices: list[int | None], template_end: int):
    raw_context = encoded[0]  # Krea2 raw: (B, 12, seq, 2560)
    if len(pair_sections) == 0:
        merged = _flatten_krea2_taps(raw_context[-1:].clone())
        return _make_sidecar(merged, []), [], int(merged.shape[1])

    flattened_sections = []
    negative_positions: list[int] = []
    running_len = 0

    for section_index, section in enumerate(pair_sections):
        current = raw_context[section_index:section_index + 1]
        visible_start = _find_krea2_user_prompt_start(section, current.shape[2], template_end)

        reference_index = reference_indices[section_index] if section_index < len(reference_indices) else None
        if reference_index is not None:
            reference = raw_context[reference_index:reference_index + 1]
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
    return _make_sidecar(merged, negative_positions), negative_positions, int(merged.shape[1])


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
        cond, negative_positions, source_length = _build_krea2_conditioning(
            encoded,
            pairs,
            plan.reference_indices,
            template_end,
        )
        extra = _build_krea2_extra(encoded, pairs, template_end, cond.shape[1])
        if negative_positions:
            extra[NEGATIVE_POSITIONS_EXTRA_KEY] = list(map(int, negative_positions))
            extra[NEGATIVE_SOURCE_LENGTH_EXTRA_KEY] = int(source_length)
            extra[NEGATIVE_SIDECAR_TOKENS_EXTRA_KEY] = max(0, int(cond.shape[1]) - int(source_length))

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


def _decode_sidecar_row(tail_row: torch.Tensor) -> tuple[list[int], int | None] | None:
    """Return (positions, source_length) for one batch row."""
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

        candidate = (row, source_length, scale)
        # Prefer the least numerically tiny surviving copy.
        if best is None or abs(candidate[2]) > abs(best[2]):
            best = candidate

    if best is None:
        return None
    return best[0], best[1]


def _parse_sidecar_token(token_rows: torch.Tensor) -> tuple[list[list[int]], list[int | None], bool]:
    positions: list[list[int]] = []
    source_lengths: list[int | None] = []
    found = False

    for b in range(token_rows.shape[0]):
        decoded = _decode_sidecar_row(token_rows[b])
        if decoded is None:
            positions.append([])
            source_lengths.append(None)
            continue
        row, source_length = decoded
        positions.append(row)
        source_lengths.append(source_length)
        found = True

    return positions, source_lengths, found


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
) -> tuple[torch.Tensor, list[list[int]] | None, list[int] | None]:
    if context.ndim != 3 or context.shape[1] < 1 or context.shape[2] < SIDECAR_HEADER:
        return context, None, None

    merged_positions: list[list[int]] = [[] for _ in range(context.shape[0])]
    keep_indices: list[int] = []
    segment_start = 0
    found_any = False
    candidate_indices = _sidecar_candidate_indices(context)
    if not candidate_indices:
        return context, None, None

    for i in range(context.shape[1]):
        if i not in candidate_indices:
            keep_indices.append(i)
            continue

        positions, source_lengths, found = _parse_sidecar_token(context[:, i, :].detach())
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
        found_any = True
        segment_start = len(keep_indices)

    if not found_any:
        return context, None, None

    if not keep_indices:
        stripped = context[:, :0, :]
    elif len(keep_indices) == context.shape[1]:
        stripped = context
    else:
        idx = torch.tensor(keep_indices, device=context.device, dtype=torch.long)
        stripped = context.index_select(1, idx)

    return stripped, merged_positions, keep_indices


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


def _collect_negative_metadata_by_uuid(conds: Any) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    if not isinstance(conds, list):
        return metadata

    for cond_group in conds:
        if not isinstance(cond_group, list):
            continue
        for cond in cond_group:
            if not isinstance(cond, dict):
                continue
            positions = cond.get(NEGATIVE_POSITIONS_EXTRA_KEY)
            if not positions:
                continue
            cond_uuid = cond.get("uuid")
            if cond_uuid is None:
                continue
            try:
                row = [int(p) for p in positions]
            except Exception:
                continue
            source_length = cond.get(NEGATIVE_SOURCE_LENGTH_EXTRA_KEY)
            try:
                source_length = int(source_length) if source_length is not None else None
            except Exception:
                source_length = None
            sidecar_tokens = cond.get(NEGATIVE_SIDECAR_TOKENS_EXTRA_KEY)
            try:
                sidecar_tokens = int(sidecar_tokens) if sidecar_tokens is not None else None
            except Exception:
                sidecar_tokens = None
            metadata[str(cond_uuid)] = {
                "positions": row,
                "source_length": source_length,
                "sidecar_tokens": sidecar_tokens,
            }
    return metadata


def _inject_negative_metadata_into_calc_args(args: dict[str, Any]) -> dict[str, Any]:
    metadata = _collect_negative_metadata_by_uuid(args.get("conds"))
    if not metadata:
        return args

    model_options = args.get("model_options", {}).copy()
    transformer_options = model_options.get("transformer_options", {}).copy()
    cfg = dict(transformer_options.get(WRAPPER_KEY, {}))
    cfg[NEGATIVE_METADATA_BY_UUID_KEY] = metadata
    transformer_options[WRAPPER_KEY] = cfg
    model_options["transformer_options"] = transformer_options
    args = dict(args)
    args["model_options"] = model_options
    return args


def _make_krea2_negpip_calc_cond_batch(previous):
    def calc_cond_batch(args):
        args = _inject_negative_metadata_into_calc_args(args)

        if previous is not None:
            return previous(args)
        return comfy.samplers.calc_cond_batch(
            args["model"],
            args["conds"],
            args["input"],
            args["sigma"],
            args["model_options"],
        )

    calc_cond_batch._krea2_negpip_original = previous
    return calc_cond_batch


def _krea2_negpip_calc_cond_batch_wrapper(executor, model, conds, x_in, timestep, model_options):
    args = {
        "model": model,
        "conds": conds,
        "input": x_in,
        "sigma": timestep,
        "model_options": model_options,
    }
    args = _inject_negative_metadata_into_calc_args(args)
    return executor(
        args["model"],
        args["conds"],
        args["input"],
        args["sigma"],
        args["model_options"],
    )


def _negative_metadata_from_transformer_options(
    transformer_options: Any,
    context_len: int,
) -> tuple[list[list[int]], int | None] | None:
    if not isinstance(transformer_options, dict):
        return None
    cfg = transformer_options.get(WRAPPER_KEY)
    if not isinstance(cfg, dict):
        return None
    metadata = cfg.get(NEGATIVE_METADATA_BY_UUID_KEY)
    uuids = transformer_options.get("uuids")
    if not isinstance(metadata, dict) or not isinstance(uuids, list) or not uuids:
        return None

    rows: list[list[int]] = []
    found = False
    trim_to_length: int | None = None
    for cond_uuid in uuids:
        item = metadata.get(str(cond_uuid))
        if not isinstance(item, dict):
            rows.append([])
            continue

        source_length = item.get("source_length")
        if source_length is not None:
            try:
                source_length = int(source_length)
                sidecar_tokens = item.get("sidecar_tokens")
                sidecar_tokens = int(sidecar_tokens) if sidecar_tokens is not None else None
                if source_length == int(context_len):
                    pass
                elif (
                    sidecar_tokens is not None
                    and sidecar_tokens > 0
                    and source_length + sidecar_tokens == int(context_len)
                ):
                    trim_to_length = source_length if trim_to_length is None else min(trim_to_length, source_length)
                else:
                    rows.append([])
                    continue
            except Exception:
                rows.append([])
                continue

        try:
            row = [int(p) for p in item.get("positions", []) if 0 <= int(p) < int(context_len)]
        except Exception:
            row = []
        rows.append(row)
        found = found or bool(row)

    return (rows, trim_to_length) if found else None


def _get_index_tensors_for_flat_v(v: torch.Tensor, cfg: dict[str, Any]):
    positions = cfg.get("_negative_positions")
    value_strength = float(cfg.get("value_strength", 1.0))
    if not positions or value_strength == 0.0:
        return None

    cache = cfg.setdefault("_flat_index_cache", {})
    key = (v.device, v.shape[0], v.shape[1], v.dtype, value_strength)
    cached = cache.get(key)
    if cached is not None:
        return cached

    bsz = int(v.shape[0])
    seq_len = int(v.shape[1])
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
    mul = torch.tensor(multipliers, device=v.device, dtype=v.dtype).view(-1, 1)
    cached = (bi, pi, mul)
    cache[key] = cached
    return cached


def _apply_flat_v_flip_inplace(v: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    if not torch.is_tensor(v) or v.ndim != 3:
        return v
    tensors = _get_index_tensors_for_flat_v(v, cfg)
    if tensors is None:
        return v
    bi, pi, mul = tensors
    # v: B,L,(H*D), before the attention implementation rearranges to heads.
    v[bi, pi, :] = v[bi, pi, :] * mul
    return v


def _cfg_active_for_block(cfg: dict[str, Any], role: str, block_index: int | None) -> bool:
    if not cfg or not cfg.get("_active", False):
        return False
    if role == "main":
        start = int(cfg.get("block_start", 0))
        end = int(cfg.get("block_end", 9999))
        stride = max(1, int(cfg.get("block_stride", 1)))
        idx = int(block_index or 0)
        return start <= idx <= end and ((idx - start) % stride) == 0
    if role == "txtfusion_refiner":
        return bool(cfg.get("patch_txtfusion_refiners", False))
    return False


def _patch_wv_for_runtime_v_flip(attn: Any) -> bool:
    wv = getattr(attn, "wv", None)
    if wv is None or getattr(wv, "_krea2_negpip_wv_patched", False):
        return False
    if not callable(getattr(wv, "forward", None)):
        return False

    original_forward = wv.forward

    def forward(self, *args, **kwargs):
        out = original_forward(*args, **kwargs)
        cfg = getattr(attn, "_krea2_negpip_runtime_cfg", None)
        if cfg:
            out = _apply_flat_v_flip_inplace(out, cfg)
        return out

    wv._krea2_negpip_original_forward = original_forward
    wv.forward = types.MethodType(forward, wv)
    wv._krea2_negpip_installed_forward = wv.forward
    wv._krea2_negpip_wv_patched = True
    return True


def _same_bound_method(a: Any, b: Any) -> bool:
    return (
        getattr(a, "__self__", None) is getattr(b, "__self__", None)
        and getattr(a, "__func__", a) is getattr(b, "__func__", b)
    )


def _restore_wv_runtime_v_flip(attn: Any) -> bool:
    wv = getattr(attn, "wv", None)
    if wv is None or not getattr(wv, "_krea2_negpip_wv_patched", False):
        return False
    original = getattr(wv, "_krea2_negpip_original_forward", None)
    installed = getattr(wv, "_krea2_negpip_installed_forward", None)
    if original is not None and (installed is None or _same_bound_method(wv.forward, installed)):
        wv.forward = original
    for name in ("_krea2_negpip_original_forward", "_krea2_negpip_installed_forward", "_krea2_negpip_wv_patched"):
        try:
            delattr(wv, name)
        except AttributeError:
            pass
    try:
        delattr(attn, "_krea2_negpip_runtime_cfg")
    except AttributeError:
        pass
    return True


def _extract_transformer_options_for_block(signature: inspect.Signature, args, kwargs):
    try:
        bound = signature.bind_partial(*args, **kwargs)
    except TypeError:
        return None
    return bound.arguments.get("transformer_options", None)


def _patch_block_for_runtime_cfg(block: Any, attn: Any, role: str, block_index: int | None = None) -> bool:
    if getattr(block, "_krea2_negpip_block_patched", False):
        return False
    if not callable(getattr(block, "forward", None)):
        return False

    _patch_wv_for_runtime_v_flip(attn)
    original_forward = block.forward
    original_signature = inspect.signature(original_forward)

    def forward(self, *args, **kwargs):
        transformer_options = _extract_transformer_options_for_block(original_signature, args, kwargs)
        cfg = transformer_options.get(WRAPPER_KEY, None) if isinstance(transformer_options, dict) else None
        active_cfg = cfg if _cfg_active_for_block(cfg, role, block_index) else None

        previous = getattr(attn, "_krea2_negpip_runtime_cfg", None)
        if active_cfg is not None:
            attn._krea2_negpip_runtime_cfg = active_cfg
        try:
            return original_forward(*args, **kwargs)
        finally:
            if previous is None:
                try:
                    delattr(attn, "_krea2_negpip_runtime_cfg")
                except AttributeError:
                    pass
            else:
                attn._krea2_negpip_runtime_cfg = previous

    block._krea2_negpip_original_forward = original_forward
    block.forward = types.MethodType(forward, block)
    block._krea2_negpip_installed_forward = block.forward
    block._krea2_negpip_block_patched = True
    return True


def _restore_block_runtime_cfg(block: Any) -> bool:
    if not getattr(block, "_krea2_negpip_block_patched", False):
        return False
    original = getattr(block, "_krea2_negpip_original_forward", None)
    installed = getattr(block, "_krea2_negpip_installed_forward", None)
    if original is not None and (installed is None or _same_bound_method(block.forward, installed)):
        block.forward = original
    for name in ("_krea2_negpip_original_forward", "_krea2_negpip_installed_forward", "_krea2_negpip_block_patched"):
        try:
            delattr(block, name)
        except AttributeError:
            pass
    return True


def _install_runtime_model_patches(dm: Any, cfg: dict[str, Any]):
    for i, block in enumerate(getattr(dm, "blocks", [])):
        if hasattr(block, "attn") and _cfg_active_for_block(cfg, "main", i):
            _patch_block_for_runtime_cfg(block, block.attn, "main", i)

    txtfusion = getattr(dm, "txtfusion", None)
    if txtfusion is not None and hasattr(txtfusion, "refiner_blocks"):
        for i, block in enumerate(txtfusion.refiner_blocks):
            if hasattr(block, "attn") and _cfg_active_for_block(cfg, "txtfusion_refiner", i):
                _patch_block_for_runtime_cfg(block, block.attn, "txtfusion_refiner", i)


def _restore_runtime_model_patches(dm: Any):
    for block in getattr(dm, "blocks", []):
        attn = getattr(block, "attn", None)
        if attn is not None:
            _restore_wv_runtime_v_flip(attn)
        _restore_block_runtime_cfg(block)

    txtfusion = getattr(dm, "txtfusion", None)
    if txtfusion is not None and hasattr(txtfusion, "refiner_blocks"):
        for block in txtfusion.refiner_blocks:
            attn = getattr(block, "attn", None)
            if attn is not None:
                _restore_wv_runtime_v_flip(attn)
            _restore_block_runtime_cfg(block)


def krea2_negpip_wrapper(executor, x, timesteps, context, attention_mask=None, ref_latents=None, transformer_options=None, **kwargs):
    transformer_options = transformer_options or {}
    cfg = transformer_options.get(WRAPPER_KEY, {})
    if not cfg or not cfg.get("enabled", True):
        return executor(x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs)

    dm = executor.class_obj
    if not _is_krea2_dm(dm):
        context_without_sidecar, negative_positions, keep_indices = _parse_and_strip_sidecar_full(context)
        attention_mask = _strip_mask_with_indices(attention_mask, keep_indices, context.shape[1])
        if negative_positions is not None:
            raise RuntimeError("Krea2 NegPiP conditioning was connected to a non-Krea2 diffusion model.")
        return executor(x, timesteps, context_without_sidecar, attention_mask, ref_latents, transformer_options, **kwargs)

    original_context_len = context.shape[1]
    context, negative_positions, keep_indices = _parse_and_strip_sidecar_full(context)
    attention_mask = _strip_mask_with_indices(attention_mask, keep_indices, original_context_len)
    if negative_positions is None:
        metadata_fallback = _negative_metadata_from_transformer_options(transformer_options, context.shape[1])
        if metadata_fallback is None:
            return executor(x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs)
        negative_positions, trim_to_length = metadata_fallback
        if trim_to_length is not None and trim_to_length < int(context.shape[1]):
            context = context[:, :trim_to_length, :]
            if torch.is_tensor(attention_mask) and attention_mask.ndim >= 1 and int(attention_mask.shape[-1]) == int(original_context_len):
                attention_mask = attention_mask[..., :trim_to_length]

    total_neg = sum(len(r) for r in negative_positions)
    value_strength = _bounded_float(cfg.get("value_strength", 1.0), 1.0, 0.0, 8.0)
    if total_neg == 0 or value_strength == 0.0:
        return executor(x, timesteps, context, attention_mask, ref_latents, transformer_options, **kwargs)

    # Keep the original transformer_options shape but install a per-call active cfg.
    # The temporary block/wv forward wrappers read this and are restored after this call.
    new_transformer_options = transformer_options.copy()
    active_cfg = dict(cfg)
    active_cfg["_active"] = True
    active_cfg["_negative_positions"] = negative_positions
    active_cfg["_flat_index_cache"] = {}
    active_cfg["value_strength"] = value_strength
    new_transformer_options[WRAPPER_KEY] = active_cfg

    try:
        _install_runtime_model_patches(dm, active_cfg)
        return executor(x, timesteps, context, attention_mask, ref_latents, new_transformer_options, **kwargs)
    finally:
        _restore_runtime_model_patches(dm)


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

        previous_calc = patched.model_options.get("sampler_calc_cond_batch_function")
        previous_calc = getattr(previous_calc, "_krea2_negpip_original", previous_calc)
        if previous_calc is None:
            # Prefer ComfyUI's composable wrapper API.  The legacy sampler_calc hook
            # bypasses CALC_COND_BATCH wrappers, so only use function chaining when a
            # pre-existing custom hook forces that path.
            patched.model_options.pop("sampler_calc_cond_batch_function", None)
            if hasattr(patched, "remove_wrappers_with_key"):
                patched.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.CALC_COND_BATCH, WRAPPER_KEY)
            patched.add_wrapper_with_key(
                comfy.patcher_extension.WrappersMP.CALC_COND_BATCH,
                WRAPPER_KEY,
                _krea2_negpip_calc_cond_batch_wrapper,
            )
        else:
            patched.model_options["sampler_calc_cond_batch_function"] = _make_krea2_negpip_calc_cond_batch(previous_calc)

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
