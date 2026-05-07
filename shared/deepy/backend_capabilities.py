from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BackendCapabilityProfile:
    family: str
    supports_skip_steps: bool
    supports_temporal_upsampling: bool
    supports_sliding_window: bool
    supports_audio_prompt: bool
    supports_image_start: bool
    supports_image_end: bool
    motion_parameter: str
    fast_motion_translation: dict[str, Any]
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "supports_skip_steps": self.supports_skip_steps,
            "supports_temporal_upsampling": self.supports_temporal_upsampling,
            "supports_sliding_window": self.supports_sliding_window,
            "supports_audio_prompt": self.supports_audio_prompt,
            "supports_image_start": self.supports_image_start,
            "supports_image_end": self.supports_image_end,
            "motion_parameter": self.motion_parameter,
            "fast_motion_translation": dict(self.fast_motion_translation),
            "notes": list(self.notes),
        }


_UNKNOWN_BACKEND = BackendCapabilityProfile(
    family="unknown",
    supports_skip_steps=False,
    supports_temporal_upsampling=False,
    supports_sliding_window=False,
    supports_audio_prompt=False,
    supports_image_start=False,
    supports_image_end=False,
    motion_parameter="",
    fast_motion_translation={},
    notes=(
        "Capabilities are unknown for this backend. Inspect exposed settings before applying model-specific controls.",
    ),
)

_BACKEND_PROFILES: tuple[tuple[re.Pattern[str], BackendCapabilityProfile], ...] = (
    (
        re.compile(r"(?:^|[_\s-])wan(?:2(?:\.2|_2)?|22|gp)?|i2v_2_2|t2v_2_2", re.IGNORECASE),
        BackendCapabilityProfile(
            family="wan",
            supports_skip_steps=True,
            supports_temporal_upsampling=True,
            supports_sliding_window=True,
            supports_audio_prompt=False,
            supports_image_start=True,
            supports_image_end=True,
            motion_parameter="motion_amplitude",
            fast_motion_translation={"motion_amplitude": 1.2, "skip_steps_multiplier": 1.75},
            notes=(
                "Wan-family templates commonly expose skip-step caching and motion amplitude controls.",
                "Use Get Default Settings to confirm exact extra_settings labels before overriding them.",
            ),
        ),
    ),
    (
        re.compile(r"ltx", re.IGNORECASE),
        BackendCapabilityProfile(
            family="ltx",
            supports_skip_steps=False,
            supports_temporal_upsampling=False,
            supports_sliding_window=True,
            supports_audio_prompt=False,
            supports_image_start=True,
            supports_image_end=False,
            motion_parameter="input_video_strength",
            fast_motion_translation={"input_video_strength": 0.85},
            notes=(
                "LTX-family templates emphasize video/input strength and windowing rather than Wan skip-step controls.",
            ),
        ),
    ),
    (
        re.compile(r"cogvideo|cog", re.IGNORECASE),
        BackendCapabilityProfile(
            family="cogvideo",
            supports_skip_steps=False,
            supports_temporal_upsampling=False,
            supports_sliding_window=False,
            supports_audio_prompt=False,
            supports_image_start=True,
            supports_image_end=False,
            motion_parameter="frame_blend",
            fast_motion_translation={"frame_blend": 0.5},
            notes=(
                "CogVideo-style backends should not receive Wan-specific skip-step settings unless exposed by the active template.",
            ),
        ),
    ),
    (
        re.compile(r"hunyuan", re.IGNORECASE),
        BackendCapabilityProfile(
            family="hunyuan",
            supports_skip_steps=False,
            supports_temporal_upsampling=False,
            supports_sliding_window=False,
            supports_audio_prompt=False,
            supports_image_start=True,
            supports_image_end=False,
            motion_parameter="motion_strength",
            fast_motion_translation={"motion_strength": 0.7},
            notes=(
                "Hunyuan-family support is treated conservatively until the active template exposes specific controls.",
            ),
        ),
    ),
    (
        re.compile(r"mochi", re.IGNORECASE),
        BackendCapabilityProfile(
            family="mochi",
            supports_skip_steps=False,
            supports_temporal_upsampling=False,
            supports_sliding_window=False,
            supports_audio_prompt=False,
            supports_image_start=False,
            supports_image_end=False,
            motion_parameter="motion_strength",
            fast_motion_translation={"motion_strength": 0.65},
            notes=(
                "Mochi-family support is conservative; confirm template controls before applying motion-specific overrides.",
            ),
        ),
    ),
)


def normalize_backend_descriptor(model_def: dict[str, Any] | None, variant: Any = "") -> str:
    parts: list[str] = []
    if isinstance(model_def, dict):
        for key in ("base_model_type", "model_type", "architecture", "type"):
            value = str(model_def.get(key, "") or "").strip()
            if value:
                parts.append(value)
    variant_text = str(variant or "").strip()
    if variant_text:
        parts.append(variant_text)
    return " ".join(parts)


def get_backend_capability_profile(model_def: dict[str, Any] | None, variant: Any = "") -> BackendCapabilityProfile:
    descriptor = normalize_backend_descriptor(model_def, variant)
    for pattern, profile in _BACKEND_PROFILES:
        if pattern.search(descriptor):
            return profile
    return _UNKNOWN_BACKEND


def describe_backend_capabilities(model_def: dict[str, Any] | None, variant: Any = "") -> dict[str, Any]:
    profile = get_backend_capability_profile(model_def, variant)
    descriptor = normalize_backend_descriptor(model_def, variant)
    result = profile.as_dict()
    result["descriptor"] = descriptor
    result["known"] = profile.family != "unknown"
    result["unsupported_parameter_policy"] = (
        "Do not silently ignore unsupported parameters. Explain the mismatch, suggest exposed alternatives, "
        "or ask the user whether to continue without the unavailable setting."
    )
    return result
