# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import io
import os
from pathlib import Path

import pytest
import torch
from PIL import Image

from vllm_omni.config.watermarking import WatermarkConfig
from vllm_omni.watermarking import WATERMARKER_REGISTRY, ImageTensor, VideoSealWatermarker

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

NBITS = 256
MESSAGE = "a5" * (NBITS // 8)


class _FakeSeal(torch.nn.Module):
    """Stand-in for a Meta Seal TorchScript model with the same embed/detect interface.

    Stores message bits verbatim in the first row of the red channel, so detection is
    exact on marked images and ~0.5 accurate on unmarked noise.
    """

    def __init__(self, nbits: int = NBITS, broken: bool = False) -> None:
        super().__init__()
        self.nbits = nbits
        self.broken = broken

    @torch.jit.export
    def embed(self, imgs: torch.Tensor, msgs: torch.Tensor) -> torch.Tensor:
        out = imgs.clone()
        out[:, 0, 0, : self.nbits] = msgs
        return out

    @torch.jit.export
    def detect(self, imgs: torch.Tensor) -> torch.Tensor:
        bits = imgs[:, 0, 0, : self.nbits] * 2 - 1
        if self.broken:
            bits = torch.zeros_like(bits)
        return torch.cat([torch.ones_like(bits[:, :1]), bits], dim=1)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return self.detect(imgs)


def _save_fake(tmp_path: Path, **kwargs) -> str:
    path = tmp_path / "fake_seal.jit"
    torch.jit.save(torch.jit.script(_FakeSeal(**kwargs)), str(path))
    return str(path)


@pytest.fixture
def watermarker(tmp_path: Path) -> VideoSealWatermarker:
    watermarker = VideoSealWatermarker(checkpoint=_save_fake(tmp_path), message=MESSAGE)
    yield watermarker
    watermarker.close()


def _noise_image(mode: str = "RGB", size: tuple[int, int] = (320, 256)) -> Image.Image:
    """Build a deterministic RGB or RGBA noise image."""
    generator = torch.Generator().manual_seed(0)
    shape = (size[1], size[0], len(mode))
    return Image.fromarray(torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8).numpy())


def test_image_watermarker_is_registered(tmp_path: Path) -> None:
    """Ensure the videoseal algorithm is selectable through the watermark config."""
    assert WATERMARKER_REGISTRY["image"]["videoseal"] is VideoSealWatermarker
    WatermarkConfig({"image": {"algorithm": "videoseal", "checkpoint": _save_fake(tmp_path)}})


def test_watermark_images_round_trips_pil(watermarker: VideoSealWatermarker) -> None:
    """Ensure PIL images keep size and mode and decode to the configured message."""
    image = _noise_image()
    (marked,) = watermarker.watermark_images("r", [image])

    assert marked.size == image.size
    assert marked.mode == "RGB"
    assert watermarker.is_image_watermarked(marked)
    assert not watermarker.is_image_watermarked(image)


def test_watermark_survives_png_encoding(watermarker: VideoSealWatermarker) -> None:
    """Ensure the mark survives the lossless encoding the image API applies."""
    (marked,) = watermarker.watermark_images("r", [_noise_image()])
    buffer = io.BytesIO()
    marked.save(buffer, format="PNG")
    buffer.seek(0)

    assert watermarker.is_image_watermarked(Image.open(buffer))


def test_watermark_images_preserves_alpha(watermarker: VideoSealWatermarker) -> None:
    """Ensure an alpha band is carried over untouched."""
    image = _noise_image("RGBA")
    (marked,) = watermarker.watermark_images("r", [image])

    assert marked.mode == "RGBA"
    assert marked.getchannel("A").tobytes() == image.getchannel("A").tobytes()
    assert watermarker.is_image_watermarked(marked)


def test_watermark_images_rejects_non_pil(watermarker: VideoSealWatermarker) -> None:
    """Ensure unexpected image payloads fail rather than pass through unmarked."""
    with pytest.raises(TypeError, match="PIL image"):
        watermarker.watermark_images("r", [torch.zeros(3, 8, 8)])


def test_tensor_path_validates_layout(watermarker: VideoSealWatermarker) -> None:
    """Ensure the tensor path rejects layouts other than [batch, 3, height, width]."""
    with pytest.raises(ValueError, match="batch, 3, height, width"):
        watermarker.watermark("r", ImageTensor(torch.rand(3, 256, 256)))


def test_message_hex_round_trips(watermarker: VideoSealWatermarker) -> None:
    """Ensure the configured message is reported back unchanged for detection tooling."""
    assert watermarker.message_hex == MESSAGE


def test_default_message_is_deterministic(tmp_path: Path) -> None:
    """Ensure the fallback message is stable across instances."""
    checkpoint = _save_fake(tmp_path)
    first = VideoSealWatermarker(checkpoint=checkpoint)
    second = VideoSealWatermarker(checkpoint=checkpoint)

    assert first.message_hex == second.message_hex
    assert len(first.message_hex) == NBITS // 4


@pytest.mark.parametrize(
    ("message", "error"),
    [
        ("zz" * (NBITS // 8), "hex digits"),
        ("0x" + "a5" * (NBITS // 8 - 1), "hex digits"),
        ("a5", f"{NBITS // 4} hex digits"),
    ],
)
def test_invalid_message_fails_at_startup(tmp_path: Path, message: str, error: str) -> None:
    """Ensure a bad message is rejected when the server starts, not per request."""
    with pytest.raises(ValueError, match=error):
        VideoSealWatermarker(checkpoint=_save_fake(tmp_path), message=message)


def test_missing_checkpoint_fails_at_startup(tmp_path: Path) -> None:
    """Ensure a missing checkpoint is reported with a pointer to the model."""
    with pytest.raises(ValueError, match="checkpoint"):
        VideoSealWatermarker()
    with pytest.raises(FileNotFoundError):
        VideoSealWatermarker(checkpoint=str(tmp_path / "missing.jit"))


def test_self_check_rejects_model_that_does_not_round_trip(tmp_path: Path) -> None:
    """Ensure a model that cannot recover its own message never serves requests."""
    with pytest.raises(RuntimeError, match="self-check failed"):
        VideoSealWatermarker(checkpoint=_save_fake(tmp_path, broken=True), message=MESSAGE)


@pytest.mark.skipif(
    not os.environ.get("VIDEOSEAL_CHECKPOINT"),
    reason="set VIDEOSEAL_CHECKPOINT to a Meta Seal TorchScript model (e.g. y_256b_img.jit)",
)
def test_real_checkpoint_marks_flux_sized_images() -> None:
    """Ensure a real Meta Seal model marks and detects a FLUX-sized image after PNG encoding."""
    watermarker = VideoSealWatermarker(checkpoint=os.environ["VIDEOSEAL_CHECKPOINT"], message=MESSAGE, num_threads=16)
    try:
        image = _noise_image(size=(1024, 1024))
        (marked,) = watermarker.watermark_images("r", [image])
        buffer = io.BytesIO()
        marked.save(buffer, format="PNG")
        buffer.seek(0)

        assert watermarker.is_image_watermarked(Image.open(buffer))
        assert not watermarker.is_image_watermarked(image)
    finally:
        watermarker.close()
