# Watermarking

vLLM-Omni supports text watermarking at sampling time in the same way as vLLM, in addition to watermarking generated multimodal outputs as post-processing. Currently, audio and image watermarking are supported, with video watermarking being an active area of integration.

For more details on text watermarking, read vLLM's [text watermarking docs](https://docs.vllm.ai/en/latest/features/watermarking/).

Watermarking multimodal outputs in vLLM Omni requires additional dependencies, which can be installed via the `watermarking` extra:

```bash
pip install 'vllm-omni[watermarking]'
```

## Configuration

Watermarking is configured in the same way as vLLM, but with additional nesting to allow configuration by modality.

As an example, to configure a text watermarker that we'd normally use in vLLM, we pass the config under the `text` modality, as shown below:

```bash
vllm serve MODEL \
  --watermark-config '{"modalities": {"text": {"algorithm":"gumbel","key":42}}}' --omni
```

Note that currently, watermarking is off by default.

### Audio Watermarking with AudioSeal

Audio watermarking is supported through `AudioSeal`. Using the format above, we can configure it for a model that emits audio as shown:

```bash
vllm serve Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --omni \
  --watermark-config '{"modalities": {"audio": {"algorithm": "audioseal"}}}'
```

In the case of offline inference, pass a `WatermarkConfig` as an initialization argument when initializing the `Omni` object:

```python
from vllm_omni import Omni
from vllm_omni.config import WatermarkConfig

omni = Omni(
    model="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    watermark_config=WatermarkConfig({"audio": {"algorithm": "audioseal"}}),
)
```

### Image Watermarking with VideoSeal

Images are watermarked through Meta's [VideoSeal](https://github.com/facebookresearch/videoseal), which embeds a fixed bit message in every generated image after generation, before encoding. It loads a Meta Seal TorchScript model, such as the 256-bit image model [`y_256b_img.jit`](https://dl.fbaipublicfiles.com/videoseal/y_256b_img.jit), so it needs no packages beyond PyTorch. Download the model ahead of time and pass its path; nothing is downloaded at startup.

```bash
vllm serve black-forest-labs/FLUX.2-klein-4B --omni \
  --watermark-config '{"strict": true, "modalities": {"image": {"algorithm": "videoseal", "checkpoint": "/models/y_256b_img.jit", "message": "<64 hex digits>"}}}'
```

The image watermarker accepts these options:

| Option | Default | Description |
| --- | --- | --- |
| `checkpoint` | required | Path to the TorchScript model. |
| `message` | a fixed vLLM-Omni message | Message to embed, as hex digits (64 for a 256-bit model). Set your own so detection identifies your deployment. |
| `device` | `cpu` | Device to run the watermark model on, e.g. `cuda`. |
| `num_threads` | PyTorch default | Caps PyTorch's CPU threads. On many-core hosts the default can make CPU watermarking several times slower. |

At startup the watermarker marks and decodes a probe image, and refuses to start if it cannot recover its message.

#### Pixel Seal

[Pixel Seal](https://github.com/facebookresearch/videoseal) is Meta's newer 256-bit image watermarking model. It is selected with `"algorithm": "pixelseal"` and takes the same options, but Meta publishes it only as a PyTorch checkpoint, so it must first be exported to a TorchScript file with the same `embed`/`detect` interface as `y_256b_img.jit`:

```bash
vllm serve black-forest-labs/FLUX.2-klein-4B --omni \
  --watermark-config '{"strict": true, "modalities": {"image": {"algorithm": "pixelseal", "checkpoint": "/models/pixelseal_y_256b_img.jit", "message": "<64 hex digits>", "device": "cuda"}}}'
```

The export runs once, offline, and needs Meta's `videoseal` package installed from its GitHub repository (the PyPI release does not include the Pixel Seal model card; this recipe was tested at commit `870ca7f`) and the [`checkpoint.pth`](https://dl.fbaipublicfiles.com/videoseal/pixelseal/checkpoint.pth). Serving needs neither. Pixel Seal does not compile with `torch.jit.script`, so it is traced; tracing records devices, so two patches applied during the trace keep the file loadable on both CPU and CUDA:

```python
import contextlib
import re
import sys
import tempfile
from pathlib import Path

import torch
import videoseal
from videoseal.modules.msg_processor import MsgProcessor
from videoseal.utils.cfg import setup_model_from_model_card

checkpoint, out = Path(sys.argv[1]), sys.argv[2]

# Build the model from Pixel Seal's model card, which sets the released watermark strength.
card = (Path(videoseal.__file__).parent / "cards" / "pixelseal.yaml").read_text()
card = re.sub(r"(?m)^checkpoint_path:.*$", f"checkpoint_path: {checkpoint.resolve()}", card)
with tempfile.TemporaryDirectory() as tmp:
    (Path(tmp) / "pixelseal.yaml").write_text(card)
    model = setup_model_from_model_card(Path(tmp) / "pixelseal.yaml").eval().requires_grad_(False)


class MetaSealInterface(torch.nn.Module):
    """The TorchScript interface: embed(imgs, msgs) -> imgs, detect(imgs) -> [B, 1 + nbits]."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def embed(self, imgs, msgs):
        return self.model.embed(imgs, msgs, is_video=False)["imgs_w"]

    def detect(self, imgs):
        preds = self.model.detect(imgs, is_video=False)["preds"]
        return preds.mean(dim=(2, 3)) if preds.ndim > 2 else preds


def binary_msg_forward(self, latents, msg, verbose=False):
    # MsgProcessor.forward for binary messages, with the indices built from msg:
    # torch.arange would record the trace device.
    indices = (2 * (torch.ones_like(msg).cumsum(-1) - 1) + msg).long()
    msg_aux = self.msg_embeddings(indices).sum(dim=-2)
    msg_aux = msg_aux[..., None, None].repeat(1, 1, latents.shape[-2], latents.shape[-1])
    if self.msg_agg == "concat":
        return torch.cat([latents, self.msg_mult * msg_aux], dim=1)
    return latents + self.msg_mult * msg_aux


def device_of(args, kwargs):
    if kwargs.keys() - {"device", "non_blocking"} or len(args) > 1:
        return None
    target = kwargs.get("device", args[0] if args else None)
    return torch.device(target) if isinstance(target, (str, torch.device)) else None


@contextlib.contextmanager
def device_independent_trace():
    # Same-device .to() calls are no-ops when served, so keep them out of the trace.
    tensor_to, module_to, msg_forward = torch.Tensor.to, torch.nn.Module.to, MsgProcessor.forward

    def tensor_to_patched(self, *args, **kwargs):
        target = device_of(args, kwargs)
        return self if target is not None and target.type == self.device.type else tensor_to(self, *args, **kwargs)

    def module_to_patched(self, *args, **kwargs):
        target, current = device_of(args, kwargs), next(self.parameters(), next(self.buffers(), None))
        if target is not None and (current is None or target.type == current.device.type):
            return self
        return module_to(self, *args, **kwargs)

    torch.Tensor.to, torch.nn.Module.to, MsgProcessor.forward = tensor_to_patched, module_to_patched, binary_msg_forward
    try:
        yield
    finally:
        torch.Tensor.to, torch.nn.Module.to, MsgProcessor.forward = tensor_to, module_to, msg_forward


imgs, msgs = torch.rand(1, 3, 512, 512), torch.randint(0, 2, (1, 256)).float()
with device_independent_trace(), torch.inference_mode():
    traced = torch.jit.trace_module(
        MetaSealInterface(model), {"embed": (imgs, msgs), "detect": (imgs,)}, check_trace=False
    )
torch.jit.save(traced, out)
```

Save it as e.g. `export_pixelseal.py` and run `python export_pixelseal.py checkpoint.pth pixelseal_y_256b_img.jit`. The watermarker's startup self-check verifies that the exported file round-trips its message. In our tests, the export's outputs were identical to the eager model on CPU and within one 8-bit level on CUDA, with every message bit recovered. Serving FLUX.2-klein-4B at 1024x1024 on an L40S, it added about 34 ms per request on CUDA and about 340 ms on CPU, so prefer `"device": "cuda"` for Pixel Seal.

## Opting out per request

Requests can opt out by setting `watermarking` to `false`, e.g., in the respective request
bodies for most endpoints, or in the session config for streaming and realtime APIs. For offline requests, you can set
`watermarking=False` on the sampling params.

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "input": "Hello, this audio is not watermarked.",
    "voice": "vivian",
    "watermarking": false
  }' --output unwatermarked.wav
```

Deployments that require watermarking should restrict opt-out to trusted callers only.

## Strict Mode (Multimodal Outputs)

Unlike text, audio and visual outputs typically embed the watermark into the data after generation. In other words, rather than guiding the sampling process as is necessary with text, we simply generate the multimodal response and then apply the watermark to the final result before emitting it.

In addition, production instances may have different requirements; in some cases, if a watermark fails, it's more desirable to emit the unwatermarked data rather than to fully fail the request after the response has been generated. In others, unwatermarked data is not acceptable, and should be considered a fatal error on the request.

To this end, we allow setting `"strict": true` in the watermarking config to fail requests where post-processing time watermarking did not succeed.

## Detection

vLLM Omni does not make guarantees regarding full support for a detection API since it's not necessary at inference time; in such cases, please refer to the documentation of the underlying library being used for how to check its applied watermark.

With that being said, if a watermarker does support detection, you can test it with the `.is_watermarked()` method on the corresponding watermarker, as shown below for the case of `AudioSeal`.

```python
import soundfile
import torch

from vllm_omni.watermarking import AudioSealWatermarker, AudioTensor

# soundfile returns [samples, channels]; the watermarker expects channels first
audio, sample_rate = soundfile.read("watermarked.wav", always_2d=True)
samples = torch.from_numpy(audio.T)

print(AudioSealWatermarker().is_watermarked(AudioTensor(samples, sample_rate)))
```

For images, `is_image_watermarked()` takes a PIL image and checks it for the configured message, so construct the watermarker with the same `checkpoint` and `message` as the deployment:

```python
from PIL import Image

from vllm_omni.watermarking import VideoSealWatermarker

watermarker = VideoSealWatermarker(checkpoint="/models/y_256b_img.jit", message="<64 hex digits>")
print(watermarker.is_image_watermarked(Image.open("generated.png")))
```

Please note that detection support is largely offered as a convenience for local verification, and is usually not ideal for optimal client-side detection.

## Limitations

- Text watermarking requires vLLM's model runner v2.

- AudioSeal requires mono audio inputs; for models emitting stereo outputs, e.g., Stable Audio, we downmix to mono and watermark the mid signal.

- Audio watermarking currently depends on the sampling rate being emitted consistently per model, which may cause it to fail for some models. This is being actively worked on as a fix.

- The image generation endpoints do not accept the `watermarking` request field yet, so online image requests cannot opt out; offline requests can, through the sampling params.

- Image watermarks are embedded before encoding, so lossy formats, resizing and cropping can weaken or remove them. Check robustness for your use case with the detector.
