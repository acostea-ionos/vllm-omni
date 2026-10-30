# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from vllm_omni.watermarking.audio_seal import AudioSealWatermarker
from vllm_omni.watermarking.base import (
    AudioWatermarkerBase,
    ImageWatermarkerBase,
    Watermarker,
    WatermarkFailureError,
)
from vllm_omni.watermarking.types import AudioTensor, ImageTensor
from vllm_omni.watermarking.video_seal import PixelSealWatermarker, VideoSealWatermarker

WATERMARKER_REGISTRY: dict[str, dict[str, type[Watermarker]]] = {
    "audio": {"audioseal": AudioSealWatermarker},
    "image": {"videoseal": VideoSealWatermarker, "pixelseal": PixelSealWatermarker},
}

__all__ = [
    "AudioSealWatermarker",
    "AudioTensor",
    "AudioWatermarkerBase",
    "ImageTensor",
    "ImageWatermarkerBase",
    "PixelSealWatermarker",
    "VideoSealWatermarker",
    "WATERMARKER_REGISTRY",
    "WatermarkFailureError",
    "Watermarker",
]
