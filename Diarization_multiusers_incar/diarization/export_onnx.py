"""Export one Nemotron-3-Diarization streaming step to a static-shape ONNX graph for the NPU.

The graph covers everything that runs once per chunk: input layer norm, the 31 Transformer layers, the projection,
the sub-pixel upsampler and the classifier. Feature stacking, the speaker cache and the FIFO stay on the CPU (see
diarizer.py). The step input is padded to a fixed length; padded frames are masked out of attention and zeroed
before the upsampler convolution, so valid frames match an unpadded forward exactly.

With --parts N the layers are split into N consecutive graphs chained through their hidden states, which keeps
each NPU compile smaller.

Usage: python export_onnx.py [--seq-len 704] [--parts 1]
"""

import argparse
from pathlib import Path

import librosa
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from transformers import AutoModelForAudioFrameClassification

MODELS = Path(__file__).parent / "models"
MASK_VALUE = -1e4  # finite so BF16 attention scores cannot turn into NaN


class DiarizationStep(nn.Module):
    """Layers [first_layer, last_layer) of one step; the first part takes embeds, the last one returns probs."""

    def __init__(self, model, seq_len, first_layer=0, last_layer=None):
        super().__init__()
        self.model = model.model
        self.classifier = model.classifier
        audio = self.model.audio_tower
        num_layers = len(audio.layers)
        last_layer = num_layers if last_layer is None else last_layer
        self.layers = audio.layers[first_layer:last_layer]
        self.is_first, self.is_last = first_layer == 0, last_layer == num_layers
        position_ids = torch.arange(seq_len)[None, :]
        cos, sin = audio.rotary_emb(torch.zeros(1, seq_len, audio.config.hidden_size), position_ids)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, hidden, mask):
        audio = self.model.audio_tower
        attention_bias = ((1.0 - mask) * MASK_VALUE)[:, None, None, :]
        if self.is_first:
            hidden = audio.input_layer_norm(hidden)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask=attention_bias, position_embeddings=(self.cos, self.sin))
        if not self.is_last:
            return hidden
        hidden = self.model.proj(audio.layer_norm(hidden)) * mask[..., None]
        return torch.sigmoid(self.classifier(self.model.upsampler(hidden)))


def part_paths(seq_len, parts):
    if parts == 1:
        return [MODELS / f"diar_step_L{seq_len}.onnx"]
    return [MODELS / f"diar_step_L{seq_len}_p{i}of{parts}.onnx" for i in range(parts)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=704)
    parser.add_argument("--parts", type=int, default=1)
    args = parser.parse_args()
    seq_len = args.seq_len

    model = AutoModelForAudioFrameClassification.from_pretrained(MODELS / "hf", attn_implementation="eager").eval()
    step = DiarizationStep(model, seq_len).eval()
    num_layers = len(model.model.audio_tower.layers)
    bounds = [round(i * num_layers / args.parts) for i in range(args.parts + 1)]

    cfg = model.config
    fe = {"n_fft": 512, "hop_length": 160, "win_length": 400, "sampling_rate": 16000, "feature_size": 128}
    mel_filters = librosa.filters.mel(
        sr=fe["sampling_rate"], n_fft=fe["n_fft"], n_mels=fe["feature_size"], fmin=0.0,
        fmax=fe["sampling_rate"] / 2, norm="slaney",
    ).astype(np.float32)
    if not (MODELS / "diar_aux.npz").exists():
        np.savez(
            MODELS / "diar_aux.npz",
            embed_weight=model.model.audio_tower.embedder.projection.weight.detach().numpy(),
            silence_embeds=model.silence_embeds.detach().numpy(),
            mel_filters=mel_filters,
        )

    # a realistic step: 600 valid frames of real-looking embeddings, the rest padding
    torch.manual_seed(0)
    valid = seq_len * 7 // 8
    features = torch.randn(1, valid * 8, 128) * 2 - 8
    embeds = torch.zeros(1, seq_len, 512)
    with torch.no_grad():
        embeds[:, :valid] = model.model.audio_tower.embedder(features)
    mask = torch.zeros(1, seq_len)
    mask[:, :valid] = 1.0

    with torch.inference_mode():
        padded = step(embeds, mask)
        unpadded = DiarizationStep(model, valid)(embeds[:, :valid], mask[:, :valid])
    pad_err = (padded[0, : valid * 8] - unpadded[0]).abs().max().item()
    print(f"padding equivalence: max |padded - unpadded| = {pad_err:.2e}")

    hidden = embeds
    for i, out_path in enumerate(part_paths(seq_len, args.parts)):
        part = DiarizationStep(model, seq_len, bounds[i], bounds[i + 1]).eval()
        torch.onnx.export(
            part, (hidden, mask), out_path, input_names=["embeds" if i == 0 else "hidden", "mask"],
            output_names=["probs" if part.is_last else "hidden"], opset_version=17, dynamo=False,
            do_constant_folding=True,
        )
        onnx.checker.check_model(str(out_path))
        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        hidden = torch.from_numpy(sess.run(None, {sess.get_inputs()[0].name: hidden.numpy(), "mask": mask.numpy()})[0])
        ops = sorted({n.op_type for n in onnx.load(str(out_path)).graph.node})
        print(f"exported {out_path.name} (layers {bounds[i]}-{bounds[i + 1] - 1}, "
              f"{out_path.stat().st_size / 1e6:.0f} MB), ops: {ops}")

    onnx_err = np.abs(hidden[0, : valid * 8].numpy() - padded[0, : valid * 8].numpy()).max()
    print(f"ONNX vs torch max err {onnx_err:.2e}")
    print(f"config: chunk={cfg.chunk_length} right_context={cfg.chunk_right_context} fifo={cfg.fifo_length} "
          f"update_period={cfg.speaker_cache_update_period} spkcache={cfg.streaming_config.speaker_cache_length}")


if __name__ == "__main__":
    main()
