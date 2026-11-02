"""Run the official Transformers model in streaming mode (as in the model card) and save its probabilities.

Usage: python reference_stream_hf.py audio/synthetic_5spk.wav [--mode low_latency]
Writes <audio>.stream_<mode>.ref.npz. The last chunk (no look-ahead) is left out: a live stream never ends.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor

MODEL_DIR = Path(__file__).parent / "models" / "hf"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("--mode", default="low_latency")
    args = parser.parse_args()

    torch.set_num_threads(8)
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    model = AutoModelForAudioFrameClassification.from_pretrained(MODEL_DIR, attn_implementation="eager").eval()
    processor.set_streaming_mode(args.mode)
    audio, sr = sf.read(args.audio, dtype="float32")

    def chunks():
        yield processor(audio[: processor.num_samples_first_audio_chunk], sampling_rate=sr,
                        is_streaming=True, is_first_audio_chunk=True)
        mel_frame_idx = processor.num_mel_frames_per_step
        start = processor.audio_chunk_start(mel_frame_idx)
        while (end := start + processor.num_samples_per_audio_chunk) <= len(audio):
            yield processor(audio[start:end], sampling_rate=sr, is_streaming=True, is_first_audio_chunk=False)
            mel_frame_idx += processor.num_mel_frames_per_step
            start = processor.audio_chunk_start(mel_frame_idx)

    speaker_cache, logits = None, []
    t = time.perf_counter()
    with torch.inference_mode():
        for inputs in chunks():
            outputs = model(**inputs, speaker_cache=speaker_cache)
            logits.append(outputs.logits)
            speaker_cache = outputs.speaker_cache
    elapsed = time.perf_counter() - t
    probs = torch.cat(logits, dim=1).sigmoid()[0].numpy()
    out = Path(args.audio).with_suffix(f".stream_{args.mode}.ref.npz")
    np.savez(out, probs=probs)
    print(f"{args.audio}: {len(logits)} chunks, {len(probs)} frames, {processor.streaming_latency_ms} ms latency, "
          f"{elapsed:.1f}s on CPU -> {out.name}")


if __name__ == "__main__":
    main()
