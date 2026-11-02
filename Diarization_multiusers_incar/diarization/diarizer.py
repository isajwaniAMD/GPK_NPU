"""Nemotron-3-Diarization on AMD Ryzen AI: numpy front end and speaker cache, ONNX Runtime per-chunk network.

The per-chunk network (diar_step_L*.onnx, see export_onnx.py) runs on the NPU through the VitisAI EP (BF16), on
the iGPU through DirectML, or on the CPU. Everything else mirrors the Transformers implementation of
Nemotron3DiarizationForAudioFrameClassification in offline mode.

Usage:
  python diarizer.py audio.wav --device npu
  python diarizer.py audio.wav --device cpu --compare audio.ref.npz
"""

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

MODELS = Path(__file__).parent / "models"

SAMPLE_RATE = 16000
N_FFT, HOP, WIN = 512, 160, 400
PREEMPHASIS = 0.97
LOG_GUARD = 2.0**-24
SUBSAMPLING = 8
NUM_SPEAKERS = 8

# offline-style operating point from the model card, in 80 ms encoder frames
CHUNK_LEN, RIGHT_CONTEXT, FIFO_LEN, UPDATE_PERIOD, SPKCACHE_LEN = 340, 40, 40, 300, 264
# mel frames per background feature block (a multiple of SUBSAMPLING)
EMBED_BLOCK = 1024


class MelFrontEnd:
    def __init__(self, mel_filters):
        self.mel_filters_t = mel_filters.T.astype(np.float32)
        window = np.zeros(N_FFT, dtype=np.float32)
        left = (N_FFT - WIN) // 2
        window[left : left + WIN] = np.hanning(WIN).astype(np.float32)
        self.window = window

    def frames(self, audio):
        """STFT frames (a strided view) and the valid-frame mask."""
        audio = audio.astype(np.float32)
        emphasized = np.concatenate([audio[:1], audio[1:] - PREEMPHASIS * audio[:-1]])
        padded = np.pad(emphasized, (N_FFT // 2, N_FFT // 2))
        num_frames = 1 + (len(padded) - N_FFT) // HOP
        frames = np.lib.stride_tricks.as_strided(
            padded, shape=(num_frames, N_FFT), strides=(padded.strides[0] * HOP, padded.strides[0])
        )
        return frames, np.arange(num_frames) < len(audio) // HOP

    def features(self, frames, mask):
        spectrum = np.fft.rfft(frames * self.window, axis=1)
        power = (spectrum.real**2 + spectrum.imag**2).astype(np.float32)
        features = np.log(power @ self.mel_filters_t + LOG_GUARD)
        features[~mask] = 0.0
        return features

    def __call__(self, audio):
        """Log-mel features (num_frames, 128) and their valid-frame mask, as NemotronAsrStreamingFeatureExtractor."""
        frames, mask = self.frames(audio)
        return self.features(frames, mask), mask


class SpeakerCache:
    """Arrival-Order Speaker Cache and FIFO queue, a numpy port of Nemotron3DiarizationSpeakerCache (batch 1)."""

    def __init__(self, hidden_size, fifo_length=FIFO_LEN, update_period=UPDATE_PERIOD):
        self.fifo_length = fifo_length
        self.update_period = update_period
        self.cache_length = SPKCACHE_LEN
        self.num_silence_frames = 1
        self.threshold = 0.25
        self.latest_frames_boost = 0.05
        budget = self.cache_length // NUM_SPEAKERS - self.num_silence_frames
        self.min_positive_scores = math.floor(budget * 0.5)
        self.num_strong_boosted = math.floor(budget * 0.75)
        self.num_weak_boosted = math.floor(budget * 1.5)

        self.embeds = np.zeros((self.cache_length, hidden_size), dtype=np.float32)
        self.probs = np.zeros((self.cache_length, NUM_SPEAKERS), dtype=np.float32)
        self.fifo = np.zeros((fifo_length, hidden_size), dtype=np.float32)
        self.num_cache = 0
        self.num_fifo = 0
        self.is_compressed = False

    def get_embeds(self):
        return np.concatenate([self.embeds[: self.num_cache], self.fifo[: self.num_fifo]])

    def update(self, step_embeds, step_probs, silence_embeds, num_chunk_frames, mask):
        num_cache, num_fifo = self.num_cache, self.num_fifo
        probs = step_probs.reshape(-1, SUBSAMPLING, NUM_SPEAKERS).mean(axis=1) * mask[:, None]

        chunk_start = num_cache + num_fifo
        chunk_embeds = step_embeds[chunk_start : chunk_start + num_chunk_frames]
        fifo_embeds = np.concatenate([self.fifo[:num_fifo], chunk_embeds])

        num_popped = 0
        if len(fifo_embeds) > self.fifo_length:
            num_popped = min(max(self.update_period, len(fifo_embeds) - self.fifo_length), len(fifo_embeds))
        if num_popped:
            fifo_probs = probs[num_cache : num_cache + len(fifo_embeds)]
            stored_probs = self.probs[:num_cache] if self.is_compressed else probs[:num_cache]
            cache_embeds = np.concatenate([self.embeds[:num_cache], fifo_embeds[:num_popped]])
            cache_probs = np.concatenate([stored_probs, fifo_probs[:num_popped]])
            fifo_embeds = fifo_embeds[num_popped:]
            if len(cache_embeds) > self.cache_length:
                cache_embeds, cache_probs = self._compress(cache_embeds, cache_probs, silence_embeds)
                self.is_compressed = True
            self.num_cache = len(cache_embeds)
            self.embeds[: self.num_cache] = cache_embeds
            self.probs[: self.num_cache] = cache_probs

        self.num_fifo = len(fifo_embeds)
        self.fifo[: self.num_fifo] = fifo_embeds

    def _frame_scores(self, probs):
        log_probs = np.log(np.maximum(probs, self.threshold))
        log_complements = np.log(np.maximum(1.0 - probs, self.threshold))
        scores = log_probs - log_complements + log_complements.sum(axis=1, keepdims=True) - math.log(0.5)
        is_speech = probs > 0.5
        scores[~is_speech] = -np.inf
        is_positive = scores > 0
        has_enough_positive = is_positive.sum(axis=0, keepdims=True) >= self.min_positive_scores
        scores[~is_positive & is_speech & has_enough_positive] = -np.inf
        return scores

    @staticmethod
    def _boost(scores, num_boosted, boost):
        top = np.argpartition(-scores, num_boosted - 1, axis=0)[:num_boosted]
        np.add.at(scores, (top, np.arange(scores.shape[1])[None, :]), boost)
        return scores

    def _compress(self, embeds, probs, silence_embeds):
        num_frames = len(embeds)
        scores = self._frame_scores(probs)
        scores[self.cache_length :] += self.latest_frames_boost
        scores = self._boost(scores, self.num_strong_boosted, -2.0 * math.log(0.5))
        scores = self._boost(scores, self.num_weak_boosted, -math.log(0.5))
        scores = np.concatenate([scores, np.full((self.num_silence_frames, NUM_SPEAKERS), np.inf, np.float32)])
        embeds = np.concatenate([embeds, silence_embeds[None, :]])
        probs = np.concatenate([probs, np.zeros((1, NUM_SPEAKERS), np.float32)])

        num_scored = num_frames + self.num_silence_frames
        sentinel = num_scored * NUM_SPEAKERS
        flat_scores = scores.T.reshape(-1)
        top = np.argpartition(-flat_scores, self.cache_length - 1)[: self.cache_length]
        top = np.where(flat_scores[top] == -np.inf, sentinel, top)
        top.sort()
        frame_indices = np.where(top == sentinel, num_frames, np.minimum(top % num_scored, num_frames))
        return embeds[frame_indices], probs[frame_indices]


def setup_vaiml_path():
    lib = Path(sys.prefix) / "Lib" / "site-packages" / "flexml" / "flexml_extras" / "lib"
    if (lib / "vaiml.dll").exists():
        os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")
        os.add_dll_directory(str(lib))


class Diarizer:
    def __init__(self, device="npu", seq_len=704, threads=8, parts=1, vai_config="vai_ep_config_accurate.json"):
        import onnxruntime as ort

        aux = np.load(MODELS / "diar_aux.npz")
        self.embed_weight_t = aux["embed_weight"].T.copy()
        self.silence_embeds = aux["silence_embeds"]
        self.front_end = MelFrontEnd(aux["mel_filters"])
        self.seq_len = seq_len
        self.device = device

        if parts == 1:
            names = [f"diar_step_L{seq_len}"]
        else:
            names = [f"diar_step_L{seq_len}_p{i}of{parts}" for i in range(parts)]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = threads
        opts.log_severity_level = 3

        self.cache_dir = MODELS / "vai_cache"
        variant = Path(vai_config).stem.replace("vai_ep_config", "")
        self.cache_keys = [name.replace("diar_step", "nemotron_diar") + "_bf16" + variant for name in names]
        if device == "npu":
            setup_vaiml_path()
            self.cache_dir.mkdir(exist_ok=True)

        start = time.perf_counter()
        self.sessions = []
        for name, cache_key in zip(names, self.cache_keys):
            if device == "npu":
                providers = [
                    ("VitisAIExecutionProvider", {
                        "config_file": str(MODELS / vai_config),
                        "cache_dir": str(self.cache_dir),
                        "cache_key": cache_key,
                        "enable_cache_file_io_in_mem": "0",
                    }),
                    ("CPUExecutionProvider", {}),
                ]
            elif device == "gpu":
                providers = [("DmlExecutionProvider", {}), ("CPUExecutionProvider", {})]
            else:
                providers = [("CPUExecutionProvider", {})]
            self.sessions.append(ort.InferenceSession(
                str(MODELS / f"{name}.onnx"), opts,
                providers=[p for p, _ in providers], provider_options=[o for _, o in providers],
            ))
        self.load_seconds = time.perf_counter() - start
        self.providers = self.sessions[0].get_providers()

    def npu_binaries(self):
        """The compiled NPU binary of every part; a missing one means that part fell back to the CPU."""
        return [next((self.cache_dir / key).glob("*.rai"), None) for key in self.cache_keys]

    def _run_step(self, embeds, mask):
        hidden = embeds
        for session in self.sessions:
            hidden = session.run(None, {session.get_inputs()[0].name: hidden, "mask": mask})[0]
        return hidden

    def _embed_blocks(self, audio, executor):
        """Starts computing the encoder input embeddings block by block in the background."""
        frames, feature_mask = self.front_end.frames(audio)
        num_frames = len(frames)
        num_embeds = -(-num_frames // SUBSAMPLING)
        embeds = np.empty((num_embeds, self.embed_weight_t.shape[1]), dtype=np.float32)

        def work(i):
            features = self.front_end.features(frames[i : i + EMBED_BLOCK], feature_mask[i : i + EMBED_BLOCK])
            padding = -len(features) % SUBSAMPLING
            stacked = np.pad(features, ((0, padding), (0, 0))).reshape(-1, SUBSAMPLING * features.shape[1])
            embeds[i // SUBSAMPLING : i // SUBSAMPLING + len(stacked)] = stacked @ self.embed_weight_t

        futures = [executor.submit(work, i) for i in range(0, num_frames, EMBED_BLOCK)]
        return embeds, feature_mask, futures

    def diarize(self, audio):
        """Per-speaker activity probabilities (num_frames, 8), one frame every 10 ms, and stage timings."""
        timings = {"features": 0.0, "network": 0.0, "cache": 0.0}
        with ThreadPoolExecutor(max_workers=1) as executor:
            embeds, feature_mask, futures = self._embed_blocks(audio, executor)
            probs = self._diarize_embeds(embeds, feature_mask, futures, timings)
        return probs, feature_mask, timings

    def _diarize_embeds(self, embeds, feature_mask, futures, timings):
        num_frames = len(feature_mask)
        embed_mask = feature_mask[::SUBSAMPLING].astype(np.float32)
        cache = SpeakerCache(embeds.shape[1])
        step_embeds = np.zeros((1, self.seq_len, embeds.shape[1]), dtype=np.float32)
        step_mask = np.zeros((1, self.seq_len), dtype=np.float32)
        outputs, num_steps, num_ready = [], 0, 0
        for start in range(0, len(embeds), CHUNK_LEN):
            end = min(start + CHUNK_LEN, len(embeds))
            chunk_end = min(end + RIGHT_CONTEXT, len(embeds))
            t = time.perf_counter()
            needed = -(-chunk_end * SUBSAMPLING // EMBED_BLOCK)
            while num_ready < min(needed, len(futures)):
                futures[num_ready].result()
                num_ready += 1
            timings["features"] += time.perf_counter() - t

            t = time.perf_counter()
            cached = cache.get_embeds()
            num_cached, num_valid = len(cached), len(cached) + chunk_end - start
            step_embeds[0, :num_cached] = cached
            step_embeds[0, num_cached:num_valid] = embeds[start:chunk_end]
            step_embeds[0, num_valid:] = 0.0
            step_mask[0, :num_cached] = 1.0
            step_mask[0, num_cached:num_valid] = embed_mask[start:chunk_end]
            step_mask[0, num_valid:] = 0.0
            timings["cache"] += time.perf_counter() - t

            t = time.perf_counter()
            probs = self._run_step(step_embeds, step_mask)[0]
            timings["network"] += time.perf_counter() - t

            t = time.perf_counter()
            probs = probs[: num_valid * SUBSAMPLING]
            cache.update(step_embeds[0, :num_valid], probs, self.silence_embeds, end - start, step_mask[0, :num_valid])
            outputs.append(probs[num_cached * SUBSAMPLING : (num_cached + end - start) * SUBSAMPLING])
            timings["cache"] += time.perf_counter() - t
            num_steps += 1

        timings["steps"] = num_steps
        return np.concatenate(outputs)[:num_frames]


def extract_segments(probs, mask, threshold=0.5):
    """Speech segments per speaker, as Nemotron3DiarizationProcessor.extract_speaker_dict."""
    active = (probs > threshold) & mask[:, None]
    changes = np.diff(np.pad(active.astype(np.int8), ((1, 1), (0, 0))), axis=0)
    segments = []
    for speaker in range(probs.shape[1]):
        starts = np.flatnonzero(changes[:, speaker] == 1)
        ends = np.flatnonzero(changes[:, speaker] == -1)
        segments += [
            {"start": round(s * HOP / SAMPLE_RATE, 2), "end": round(e * HOP / SAMPLE_RATE, 2), "speaker": speaker}
            for s, e in zip(starts, ends)
        ]
    return sorted(segments, key=lambda s: (s["start"], s["speaker"]))


def load_audio(path):
    try:
        import soundfile as sf

        audio, sr = sf.read(path, dtype="float32")
    except ImportError:
        import wave

        with wave.open(str(path), "rb") as w:
            if w.getsampwidth() != 2:
                raise ValueError(f"{path}: only 16-bit PCM WAV is supported without soundfile")
            sr, channels = w.getframerate(), w.getnchannels()
            audio = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
            audio = audio.reshape(-1, channels)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path}: expected {SAMPLE_RATE} Hz audio, got {sr} Hz")
    return audio


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio")
    parser.add_argument("--device", choices=["npu", "gpu", "cpu"], default="npu")
    parser.add_argument("--seq-len", type=int, default=704)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--parts", type=int, default=1, help="number of chained step graphs (export_onnx.py --parts)")
    parser.add_argument("--vai-config", default="vai_ep_config_accurate.json", help="VitisAI EP config file in models/")
    parser.add_argument("--runs", type=int, default=1, help="timed runs after one warm-up run")
    parser.add_argument("--compare", help="reference .ref.npz from reference_hf.py")
    parser.add_argument("--json", help="write segments to this file")
    parser.add_argument("--save", help="write the per-frame probabilities to this .npz (for eval_der.py)")
    args = parser.parse_args()

    audio = load_audio(args.audio)
    duration = len(audio) / SAMPLE_RATE
    diarizer = Diarizer(args.device, args.seq_len, args.threads, args.parts, args.vai_config)
    print(f"Loaded in {diarizer.load_seconds:.2f}s, providers: {diarizer.providers}")
    if args.device == "npu":
        for key, binary in zip(diarizer.cache_keys, diarizer.npu_binaries()):
            print(f"NPU binary: {binary} ({binary.stat().st_size / 1e6:.0f} MB)" if binary
                  else f"WARNING: no compiled NPU binary for {key}, that part is running on the CPU")

    probs, mask, timings = diarizer.diarize(audio)
    best = None
    for _ in range(args.runs):
        start = time.perf_counter()
        probs, mask, timings = diarizer.diarize(audio)
        elapsed = time.perf_counter() - start
        if best is None or elapsed < best[0]:
            best = (elapsed, timings)
    elapsed, timings = best

    segments = extract_segments(probs, mask)
    speakers = sorted({s["speaker"] for s in segments})
    print(f"\n{args.audio}: {duration:.1f}s audio, {len(segments)} segments, speakers {speakers}")
    print(f"Device {args.device}: {elapsed * 1000:.0f} ms total, RTFx {duration / elapsed:.0f} "
          f"({timings['steps']} steps of {args.seq_len} frames)")
    for stage in ("features", "network", "cache"):
        print(f"  {stage:8s} {timings[stage] * 1000:8.1f} ms  ({timings[stage] / elapsed * 100:4.1f}%)")
    per_step = timings["network"] / timings["steps"] * 1000
    print(f"  network per step: {per_step:.1f} ms")
    for s in segments[:12]:
        print(f"  {s['start']:7.2f} {s['end']:7.2f} speaker_{s['speaker']}")

    if args.compare:
        ref = np.load(args.compare)
        n = min(len(ref["probs"]), len(probs))
        ref_active, active = ref["probs"][:n] > 0.5, probs[:n] > 0.5
        agreement = (ref_active == active).all(axis=1)[mask[:n]].mean()
        print(f"\nvs reference: max |prob diff| {np.abs(ref['probs'][:n] - probs[:n]).max():.4f}, "
              f"mean {np.abs(ref['probs'][:n] - probs[:n]).mean():.5f}, "
              f"frames with identical speaker decisions {agreement * 100:.2f}%")
        if "features" in ref:
            print(f"  mel max diff {np.abs(ref['features'][:n] - diarizer.front_end(audio)[0][:n]).max():.2e}")

    if args.json:
        Path(args.json).write_text(json.dumps(segments, indent=1))
    if args.save:
        np.savez(args.save, probs=probs)


if __name__ == "__main__":
    main()
