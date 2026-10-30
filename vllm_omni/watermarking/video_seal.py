# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import hashlib
from pathlib import Path
from typing import ClassVar, Protocol, cast

import numpy as np
import regex as re
import torch
from vllm.logger import init_logger

from vllm_omni.watermarking.base import ImageWatermarkerBase
from vllm_omni.watermarking.types import ImageTensor

logger = init_logger(__name__)

VIDEOSEAL_JIT_URL = "https://dl.fbaipublicfiles.com/videoseal/y_256b_img.jit"


class _SealModel(Protocol):
    """The Meta Seal TorchScript interface: ``detect`` returns ``[B, 1 + nbits]`` predictions."""

    def embed(self, imgs: torch.Tensor, msgs: torch.Tensor) -> torch.Tensor: ...

    def detect(self, imgs: torch.Tensor) -> torch.Tensor: ...


class VideoSealWatermarker(ImageWatermarkerBase):
    """Watermark images with a Meta Seal TorchScript model such as VideoSeal v1.0.


    Configured through ``--watermark-config``, e.g.::

        {"strict": true,
         "modalities": {"image": {"algorithm": "videoseal", "checkpoint": "/models/y_256b_img.jit",
                                  "message": "<hex digits>", "device": "cpu", "num_threads": 16}}}

    With ``"strict": true``, a request whose images cannot be marked fails instead of
    being delivered unmarked.

    Every image is marked with the same configured message, so detection can tell
    that an image came from this deployment, not only that some mark is present.
    """

    # An unmarked image decodes to ~0.5 of a fixed message's bits (std ~0.03 at
    # 256 bits); marked outputs decode above 0.99 after 8-bit quantization.
    DETECTION_BIT_ACCURACY = 0.8
    # The configured algorithm name, used in logs and errors.
    ALGORITHM: ClassVar[str] = "videoseal"
    _DEFAULT_MESSAGE_SEED = b"vllm-omni/videoseal"
    _WARMUP_SIZE = 1024
    _WARMUP_CALLS = 2

    def __init__(
        self,
        checkpoint: str | None = None,
        message: str | None = None,
        device: str = "cpu",
        num_threads: int | None = None,
    ) -> None:
        """Load the TorchScript model, warm it up and verify it round-trips the message."""
        super().__init__()
        if checkpoint is None:
            raise ValueError(
                f"{self.ALGORITHM} watermarking requires a 'checkpoint' path to a TorchScript model, "
                f"e.g. {VIDEOSEAL_JIT_URL}"
            )
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"{self.ALGORITHM} checkpoint not found: {path}")
        if num_threads is not None:
            # torch's intra-op pool is process-wide. On many-core hosts the default
            # oversubscribes: embedding a 1024x1024 image took ~4x longer at 254
            # threads than with a 16-thread cap.
            torch.set_num_threads(num_threads)
        self._device = torch.device(device)
        logger.info("Loading %s watermark model from %s on %s", self.ALGORITHM, path, self._device)
        self._model = cast(_SealModel, torch.jit.load(str(path), map_location=self._device).eval())
        self._message = self._self_check(message)

    @property
    def message_hex(self) -> str:
        """Return the embedded message as hex digits, for detection tooling."""
        return np.packbits(self._message[0].to(torch.uint8).cpu().numpy()).tobytes().hex()

    def _watermark_image(self, data: ImageTensor) -> ImageTensor:
        """Embed the configured message in every image of the batch."""
        pixels = data.pixels
        source = pixels.to(device=self._device, dtype=torch.float32)
        marked = self._model.embed(source, self._message.expand(source.shape[0], -1)).clamp(0, 1)
        return ImageTensor(marked.to(device=pixels.device, dtype=pixels.dtype))

    def _is_image_watermarked(self, data: ImageTensor) -> bool:
        """Return whether every image decodes to the configured message."""
        preds = self._model.detect(data.pixels.to(device=self._device, dtype=torch.float32))
        return self._bit_accuracy(preds, self._message) >= self.DETECTION_BIT_ACCURACY

    def _self_check(self, message: str | None) -> torch.Tensor:
        """Warm the model up and prove it round-trips the configured message.

        TorchScript's first calls run several times slower than steady state, so
        they are paid here instead of by the first request.
        """
        generator = torch.Generator().manual_seed(0)
        probe = torch.rand(1, 3, self._WARMUP_SIZE, self._WARMUP_SIZE, generator=generator).to(self._device)
        with torch.inference_mode():
            nbits = self._model.detect(probe).shape[1] - 1
            msg = self._parse_message(message, nbits).to(self._device)
            for _ in range(self._WARMUP_CALLS):
                marked = self._model.embed(probe, msg).clamp(0, 1).mul(255).round().div(255)
                preds = self._model.detect(marked)
        accuracy = self._bit_accuracy(preds, msg)
        if accuracy < self.DETECTION_BIT_ACCURACY:
            raise RuntimeError(f"{self.ALGORITHM} self-check failed: decoded {accuracy:.3f} of the message bits")
        logger.info("%s ready: %d-bit message, self-check bit accuracy %.3f", self.ALGORITHM, nbits, accuracy)
        return msg

    @classmethod
    def _parse_message(cls, message: str | None, nbits: int) -> torch.Tensor:
        """Convert hex digits to a [1, nbits] float bit tensor."""
        if message is None:
            digest = hashlib.sha256(cls._DEFAULT_MESSAGE_SEED).digest()
            if nbits != len(digest) * 8:
                raise ValueError(
                    f"no default message for a {nbits}-bit model; set 'message' to {nbits // 4} hex digits"
                )
            logger.warning("No watermark message configured; the vllm-omni default does not identify this deployment")
            message = digest.hex()
        if not isinstance(message, str) or not re.fullmatch(r"[0-9a-fA-F]+", message):
            raise ValueError("watermark message must be a string of hex digits")
        if len(message) * 4 != nbits:
            raise ValueError(f"watermark message must be {nbits // 4} hex digits for this model, got {len(message)}")
        bits = np.unpackbits(np.frombuffer(bytes.fromhex(message), dtype=np.uint8))
        return torch.from_numpy(bits).float()[None]

    @staticmethod
    def _bit_accuracy(preds: torch.Tensor, message: torch.Tensor) -> float:
        """Return the lowest per-image fraction of message bits recovered."""
        bits = (preds[:, 1:] > 0).to(message.dtype)
        return (bits == message).float().mean(dim=1).min().item()


class PixelSealWatermarker(VideoSealWatermarker):
    """Watermark images with a TorchScript export of Meta's Pixel Seal.

    Meta publishes Pixel Seal only as a PyTorch checkpoint; the export (see
    ``docs/features/watermarking.md``) has the same interface as VideoSeal's.
    """

    ALGORITHM: ClassVar[str] = "pixelseal"
