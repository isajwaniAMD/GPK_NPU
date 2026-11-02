"""Live (streaming) Nemotron-3-Diarization on the NPU, in the model card's low-latency mode (1.04 s buffer).

Feed audio as it arrives; every 0.72 s of audio one step of 9 new encoder frames plus 4 look-ahead frames runs on
the 544-frame step graph (diar_step_L544.onnx), attending to the speaker cache and FIFO. The output timeline is the
same as the Transformers streaming mode of Nemotron3DiarizationForAudioFrameClassification.

Usage (file fed as a live stream):
  python stream_diarizer.py audio/synthetic_5spk.wav --device npu --compare audio/synthetic_5spk.stream_low_latency.ref.npz
"""

import argparse
import threading
import time

import numpy as np

import diarizer as dz

STREAMING_MODES = {  # (chunk_len, right_context) in 80 ms encoder frames, from processor_config.json
    "low_latency": (9, 4),
    "very_low_latency": (6, 2),
    "ultra_low_latency": (3, 1),
}
STREAM_FIFO_LEN, STREAM_UPDATE_PERIOD = 264, 222
FRAME_SECONDS = dz.HOP / dz.SAMPLE_RATE


class StreamingDiarizer:
    def __init__(self, device="npu", mode="low_latency", seq_len=544, threads=4):
        self.chunk_len, self.right_context = STREAMING_MODES[mode]
        if dz.SPKCACHE_LEN + STREAM_FIFO_LEN + self.chunk_len + self.right_context > seq_len:
            raise ValueError(f"{mode} needs a step graph of at least "
                             f"{dz.SPKCACHE_LEN + STREAM_FIFO_LEN + self.chunk_len + self.right_context} frames")
        self.net = dz.Diarizer(device, seq_len=seq_len, threads=threads)
        self.front_end = self.net.front_end
        hidden = self.net.embed_weight_t.shape[1]
        self.cache = dz.SpeakerCache(hidden, fifo_length=STREAM_FIFO_LEN, update_period=STREAM_UPDATE_PERIOD)
        self.step_embeds = np.zeros((1, seq_len, hidden), dtype=np.float32)
        self.step_mask = np.zeros((1, seq_len), dtype=np.float32)

        self._emphasized = np.zeros(dz.N_FFT // 2, dtype=np.float32)  # the centering zeros
        self._buffer_start = -(dz.N_FFT // 2)  # sample index of _emphasized[0]
        self._last_sample = None
        self._num_features = 0  # mel frames computed so far
        self._pending_features = np.zeros((0, self.front_end.mel_filters_t.shape[1]), dtype=np.float32)
        self._embeds = np.zeros((0, hidden), dtype=np.float32)  # embeds not yet consumed by a step

        self._probs = []  # (72, 8) blocks of 10 ms speaker probabilities
        self.num_output_frames = 0
        self.step_seconds = []
        self._lock = threading.Lock()

    @property
    def latency_seconds(self):
        return (self.chunk_len + self.right_context) * dz.SUBSAMPLING * FRAME_SECONDS

    def feed(self, samples):
        """Adds audio samples (16 kHz mono float32) and runs every step whose audio is complete."""
        samples = np.asarray(samples, dtype=np.float32)
        if not len(samples):
            return
        previous = samples[:1] if self._last_sample is None else self._last_sample
        emphasized = samples - dz.PREEMPHASIS * np.concatenate([previous, samples[:-1]])
        if self._last_sample is None:
            emphasized[0] = samples[0]
        self._last_sample = samples[-1:]
        self._emphasized = np.concatenate([self._emphasized, emphasized])
        self._compute_features()
        while len(self._embeds) >= self.chunk_len + self.right_context:
            self._step()

    def _compute_features(self):
        # frame i covers emphasized samples [i * HOP - N_FFT / 2, i * HOP + N_FFT / 2)
        buffer_end = self._buffer_start + len(self._emphasized)
        num_complete = (buffer_end - dz.N_FFT // 2) // dz.HOP + 1
        if num_complete > self._num_features:
            first = self._num_features * dz.HOP - dz.N_FFT // 2 - self._buffer_start
            span = self._emphasized[first:]
            count = num_complete - self._num_features
            frames = np.lib.stride_tricks.as_strided(
                span, shape=(count, dz.N_FFT), strides=(span.strides[0] * dz.HOP, span.strides[0])
            )
            features = self.front_end.features(frames, np.ones(count, dtype=bool))
            self._num_features = num_complete
            keep_from = self._num_features * dz.HOP - dz.N_FFT // 2 - self._buffer_start
            self._emphasized = self._emphasized[keep_from:].copy()
            self._buffer_start += keep_from
            self._pending_features = np.concatenate([self._pending_features, features])

        num_groups = len(self._pending_features) // dz.SUBSAMPLING
        if num_groups:
            stacked = self._pending_features[: num_groups * dz.SUBSAMPLING].reshape(num_groups, -1)
            self._pending_features = self._pending_features[num_groups * dz.SUBSAMPLING :]
            self._embeds = np.concatenate([self._embeds, stacked @ self.net.embed_weight_t])

    def _step(self):
        t = time.perf_counter()
        chunk = self._embeds[: self.chunk_len + self.right_context]
        cached = self.cache.get_embeds()
        num_cached, num_valid = len(cached), len(cached) + len(chunk)
        self.step_embeds[0, :num_cached] = cached
        self.step_embeds[0, num_cached:num_valid] = chunk
        self.step_embeds[0, num_valid:] = 0.0
        self.step_mask[0, :num_valid] = 1.0
        self.step_mask[0, num_valid:] = 0.0

        probs = self.net._run_step(self.step_embeds, self.step_mask)[0][: num_valid * dz.SUBSAMPLING]
        self.cache.update(self.step_embeds[0, :num_valid], probs, self.net.silence_embeds, self.chunk_len,
                          self.step_mask[0, :num_valid])
        new = probs[num_cached * dz.SUBSAMPLING : (num_cached + self.chunk_len) * dz.SUBSAMPLING].copy()
        self._embeds = self._embeds[self.chunk_len :]
        with self._lock:
            self._probs.append(new)
            self.num_output_frames += len(new)
        self.step_seconds.append(time.perf_counter() - t)

    def probs(self):
        with self._lock:
            return np.concatenate(self._probs) if self._probs else np.zeros((0, dz.NUM_SPEAKERS), np.float32)

    def covered_until(self):
        """Seconds of audio that have speaker probabilities so far."""
        return self.num_output_frames * FRAME_SECONDS

    def speakers_between(self, start, end, threshold=0.5):
        """Per-speaker share of active frames in [start, end) seconds, largest first: [(speaker, share), ...]."""
        probs = self.probs()[int(start / FRAME_SECONDS) : int(end / FRAME_SECONDS)]
        if not len(probs):
            return []
        active = (probs > threshold).sum(axis=0)
        total = active.sum()
        if not total:
            return []
        order = np.argsort(-active)
        return [(int(s), float(active[s] / total)) for s in order if active[s]]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio")
    parser.add_argument("--device", choices=["npu", "gpu", "cpu"], default="npu")
    parser.add_argument("--mode", choices=list(STREAMING_MODES), default="low_latency")
    parser.add_argument("--block-ms", type=int, default=100, help="size of the audio blocks fed in")
    parser.add_argument("--compare", help="reference from reference_stream_hf.py")
    args = parser.parse_args()

    audio = dz.load_audio(args.audio)
    stream = StreamingDiarizer(args.device, args.mode)
    print(f"Loaded in {stream.net.load_seconds:.2f}s, providers: {stream.net.providers}")
    if args.device == "npu":
        for key, binary in zip(stream.net.cache_keys, stream.net.npu_binaries()):
            print(f"NPU binary: {binary}" if binary else f"WARNING: no compiled NPU binary for {key}")

    block = dz.SAMPLE_RATE * args.block_ms // 1000
    start = time.perf_counter()
    for i in range(0, len(audio), block):
        stream.feed(audio[i : i + block])
    elapsed = time.perf_counter() - start
    steps = np.array(stream.step_seconds) * 1000
    duration = len(audio) / dz.SAMPLE_RATE
    print(f"\n{args.audio}: {duration:.1f}s fed in {args.block_ms} ms blocks, {len(steps)} steps, "
          f"{elapsed:.1f}s total (RTFx {duration / elapsed:.0f})")
    print(f"  buffer latency {stream.latency_seconds:.2f}s + step compute: median {np.median(steps):.0f} ms, "
          f"p95 {np.percentile(steps, 95):.0f} ms, max {steps.max():.0f} ms "
          f"(budget {stream.chunk_len * dz.SUBSAMPLING * FRAME_SECONDS * 1000:.0f} ms per step)")

    probs = stream.probs()
    if args.compare:
        ref = np.load(args.compare)["probs"]
        n = min(len(ref), len(probs))
        agree = ((ref[:n] > 0.5) == (probs[:n] > 0.5)).all(axis=1).mean()
        print(f"vs reference ({n} frames): max |diff| {np.abs(ref[:n] - probs[:n]).max():.4f}, "
              f"mean {np.abs(ref[:n] - probs[:n]).mean():.5f}, identical decisions {agree * 100:.2f}%")
    segments = dz.extract_segments(probs, np.ones(len(probs), dtype=bool))
    for s in segments[:10]:
        print(f"  {s['start']:7.2f} {s['end']:7.2f} speaker_{s['speaker']}")


if __name__ == "__main__":
    main()
