"""
Parakeet TDT 0.6B ONNX inference pipeline.

Supports two execution modes:
  - CPU: Standard ONNX Runtime CPU Execution Provider
  - NPU: AMD Ryzen AI NPU via VitisAI Execution Provider

The VitisAI EP automatically partitions the ONNX graph, running
NPU-supported operators on the NPU and falling back to CPU for the rest.
"""

import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class Transcriber:
    """
    ONNX Runtime inference pipeline for Parakeet TDT 0.6B.

    Loads encoder and decoder ONNX models and runs inference
    using either NPU (VitisAI EP) or CPU execution provider.
    """

    # Max encoded frames the model supports (limited by attention positional encoding).
    # The Conformer attention bias is 1750 frames. We stay safely below that.
    MAX_ENCODED_FRAMES = 1700
    # Corresponding max mel frames (before 8x subsampling): 1700 * 8 = 13600
    # Corresponding max audio duration: 13600 * 160 / 16000 = 136 seconds (~2.3 min)
    MAX_CHUNK_SECONDS = 120  # 2 minutes, conservative
    MAX_CHUNK_SAMPLES = MAX_CHUNK_SECONDS * 16000
    OVERLAP_SAMPLES = 2 * 16000  # 2 seconds overlap between chunks
    ENCODER_FRAME_SECONDS = 0.08  # 10 ms mel hop x 8x encoder subsampling

    def __init__(
        self,
        models_dir: str = "./models",
        device: str = "cpu",
        decoder_device: str = "auto",
        debug: bool = False,
        profile: bool = False,
    ):
        """
        Initialize the transcriber.

        Args:
            models_dir: Path to directory containing ONNX model files.
            device: Execution device - "cpu", "npu", or "gpu".
            decoder_device: Decoder execution device - "auto", "cpu", "gpu".
                "auto" = CPU (safest for the tiny per-step LSTM).
                "gpu" = DirectML on AMD Radeon iGPU (experimental).
            debug: Enable verbose debug logging.
            profile: Enable ORT profiling to capture op placement per EP.
                Adds ~2x overhead to encoder inference -- use for diagnostics only.
        """
        self.decoder_device = decoder_device.lower()
        self.profile = profile

        # Ensure flexml/vaiml DLLs are findable for BF16 NPU compilation
        if device.lower() == "npu":
            self._setup_vaiml_path()

        # Import onnxruntime here so we get a clear error if it's missing
        try:
            import onnxruntime as ort

            self.ort = ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required. Install with: pip install onnxruntime\n"
                "For NPU support, use the onnxruntime from the ryzen-ai conda environment."
            )

        self.models_dir = Path(models_dir)
        self.device = device.lower()
        self.debug = debug

        if debug:
            logging.getLogger("inference").setLevel(logging.DEBUG)
            logger.setLevel(logging.DEBUG)

        # Model parameters (from config.json and model inspection)
        self.features_size = 128
        self.subsampling_factor = 8
        self.encoder_dim = 1024
        self.state_dim = 640
        self.num_lstm_layers = 2
        self.num_duration_classes = 5  # TDT uses 5 duration classes (0-4)
        self.max_tokens_per_step = 10
        self.blank_idx = 8192

        # Static model config (for NPU)
        self._static_config = None
        self._fixed_frames = None  # Fixed mel frame count for static encoder

        # Load config
        self._load_config()

        # Load static model config if using NPU
        if self.device == "npu":
            self._load_static_config()

        # Load vocabulary
        self.vocab = {}
        self.vocab_size = 0
        self._load_vocab()

        # Initialize mel filterbank
        from .mel import MelFilterbank

        self.mel = MelFilterbank(n_mels=self.features_size, sample_rate=16000)

        # Create ONNX Runtime sessions
        self._encoder_session = None
        self._decoder_session = None
        self._create_sessions()

        logger.info(
            "Transcriber initialized: device=%s, vocab_size=%d, encoder_dim=%d, static=%s",
            self.device,
            self.vocab_size,
            self.encoder_dim,
            self._fixed_frames is not None,
        )

    @staticmethod
    def _setup_vaiml_path():
        """Add flexml DLL directory to PATH so vaiml.dll can be found.

        Searches multiple locations since the package name (flexml-lite)
        differs from the import name (flexml), and CONDA_PREFIX may not
        always be set (e.g. when running from an IDE terminal).
        """
        import sys

        flexml_dir = None
        # Try both possible import names
        for module_name in ("flexml", "flexml_lite"):
            try:
                mod = __import__(module_name)
                if mod.__file__:
                    flexml_dir = Path(mod.__file__).parent
                elif hasattr(mod, "__path__"):
                    # Namespace package -- use the first search path
                    flexml_dir = Path(list(mod.__path__)[0])
                break
            except ImportError:
                continue

        # Build search paths for vaiml.dll
        search_dirs = []

        # 1. From the flexml module location
        if flexml_dir:
            search_dirs.append(flexml_dir / "flexml_extras" / "lib")
            # Also check parent in case we got a sub-package
            search_dirs.append(flexml_dir.parent / "flexml_extras" / "lib")

        # 2. From CONDA_PREFIX env var
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        if conda_prefix:
            search_dirs.append(
                Path(conda_prefix) / "Lib" / "site-packages" / "flexml" / "flexml_extras" / "lib"
            )

        # 3. From sys.prefix (works even when CONDA_PREFIX isn't set)
        search_dirs.append(
            Path(sys.prefix) / "Lib" / "site-packages" / "flexml" / "flexml_extras" / "lib"
        )

        for d in search_dirs:
            vaiml_path = d / "vaiml.dll"
            if vaiml_path.exists():
                dir_str = str(d)
                if dir_str not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = dir_str + os.pathsep + os.environ.get("PATH", "")
                    logger.info("Added flexml lib to PATH: %s", dir_str)
                # Also use os.add_dll_directory on Python 3.8+
                try:
                    os.add_dll_directory(dir_str)
                except (OSError, AttributeError):
                    pass
                return

        logger.warning(
            "vaiml.dll not found. BF16 NPU compilation will fail. "
            "Ensure flexml-lite is installed in the Ryzen AI conda environment. "
            "Searched: %s",
            [str(d) for d in search_dirs],
        )

    def _load_config(self):
        """Load model configuration from config.json."""
        config_path = self.models_dir / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
            self.features_size = config.get("features_size", self.features_size)
            self.subsampling_factor = config.get(
                "subsampling_factor", self.subsampling_factor
            )
            logger.debug("Config loaded: features_size=%d, subsampling=%d",
                         self.features_size, self.subsampling_factor)
        else:
            logger.warning("config.json not found, using defaults")

    def _load_static_config(self):
        """Load static model configuration for NPU mode."""
        config_path = self.models_dir / "static_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                self._static_config = json.load(f)
            self._fixed_frames = self._static_config["fixed_frames"]
            chunk_sec = self._static_config["chunk_seconds"]
            # Override chunk sizes based on static model config
            Transcriber.MAX_CHUNK_SECONDS = chunk_sec
            Transcriber.MAX_CHUNK_SAMPLES = chunk_sec * 16000
            logger.info(
                "Static model config: %ds chunks, %d fixed mel frames, encoded_len=%d",
                chunk_sec,
                self._fixed_frames,
                self._static_config["encoded_len"],
            )
        else:
            logger.warning(
                "NPU mode but no static_config.json found! "
                "Run: python -m inference.tools.convert_static --models-dir %s\n"
                "Without static models, VitisAI EP will fall back to CPU.",
                self.models_dir,
            )

    def _load_vocab(self):
        """Load SentencePiece vocabulary from vocab.txt."""
        vocab_path = self.models_dir / "vocab.txt"
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")

        with open(vocab_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue
                token, id_str = parts
                try:
                    token_id = int(id_str)
                except ValueError:
                    continue
                # Replace SentencePiece marker with space
                token = token.replace("\u2581", " ")
                self.vocab[token_id] = token

        self.vocab_size = len(self.vocab)
        logger.debug("Vocab loaded: %d tokens, blank_idx=%d", self.vocab_size, self.blank_idx)

    def _find_model_file(self, base_name_int8: str, base_name_fp32: str) -> str:
        """Find model file, preferring static version for NPU."""
        # On NPU, prefer static-shape models from static_config
        if self.device == "npu" and self._static_config:
            # First, check if static_config has explicit model filenames
            config_encoder = self._static_config.get("encoder_model", "")
            config_decoder = self._static_config.get("decoder_model", "")

            # Match the requested model type (encoder vs decoder)
            is_encoder = "encoder" in base_name_fp32
            explicit_name = config_encoder if is_encoder else config_decoder

            if explicit_name:
                explicit_path = self.models_dir / explicit_name
                if explicit_path.exists():
                    logger.info("Using static model from config: %s", explicit_name)
                    return str(explicit_path)

            # Fallback: derive filename from precision
            config_precision = self._static_config.get("precision", "int8")
            if config_precision == "fp32":
                static_name = base_name_fp32.replace(".onnx", ".fp32.static.onnx")
            else:
                static_name = base_name_int8.replace(".onnx", ".static.onnx")

            static_path = self.models_dir / static_name
            if static_path.exists():
                logger.info("Using static model: %s", static_name)
                return str(static_path)

            # Try the other precision's static model as fallback
            alt_static_fp32 = base_name_fp32.replace(".onnx", ".fp32.static.onnx")
            alt_static_int8 = base_name_int8.replace(".onnx", ".static.onnx")
            for candidate in [alt_static_fp32, alt_static_int8]:
                alt_path = self.models_dir / candidate
                if alt_path.exists():
                    logger.info("Using static model (fallback): %s", candidate)
                    return str(alt_path)

            logger.warning(
                "No static model found for NPU. "
                "Run: python -m inference.tools.convert_static --precision fp32\n"
                "Falling back to dynamic model."
            )

        # Fall back to dynamic models: prefer FP32 for BF16 path, then INT8
        fp32_path = self.models_dir / base_name_fp32
        int8_path = self.models_dir / base_name_int8

        if self.device == "npu":
            # For NPU BF16 path, prefer FP32 models
            if fp32_path.exists():
                return str(fp32_path)
            elif int8_path.exists():
                return str(int8_path)
        else:
            # For CPU, prefer INT8 (faster)
            if int8_path.exists():
                return str(int8_path)
            elif fp32_path.exists():
                return str(fp32_path)

        raise FileNotFoundError(
            f"Model not found: tried {int8_path} and {fp32_path}. "
            f"Run: python -m inference.tools.download_models --precision fp32 --output-dir {self.models_dir}"
        )

    def _get_session_options(self) -> "ort.SessionOptions":
        """Create ONNX Runtime session options."""
        opts = self.ort.SessionOptions()
        opts.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3  # Error only (suppress graph transformer spam)
        return opts

    def _get_providers(self, model_name: str = "", model_path: str = ""):
        """
        Get execution providers based on device setting.

        Returns list of (provider_name, options) tuples.

        Supported devices:
          - "cpu": CPU only
          - "npu": VitisAI EP (Ryzen AI NPU) via BF16 compilation
          - "gpu": DirectML EP (integrated AMD Radeon GPU) - handles dynamic shapes
        """
        available = self.ort.get_available_providers()
        logger.debug("Available ONNX Runtime providers: %s", available)

        if self.device == "npu":
            if "VitisAIExecutionProvider" not in available:
                logger.warning(
                    "VitisAIExecutionProvider not available! Available: %s. "
                    "Make sure you activated the ryzen-ai conda environment. "
                    "Falling back to CPU.",
                    available,
                )
                return [("CPUExecutionProvider", {})]

            # Determine if model is FP32 (use BF16 path) or INT8 (use target path)
            is_fp32 = model_path and "int8" not in Path(model_path).name.lower()

            if is_fp32:
                # BF16 compilation path: use config_file for FP32 -> BF16 on NPU
                config_file = str(self.models_dir / "vai_ep_config.json")
                if not Path(config_file).exists():
                    logger.error(
                        "BF16 config file not found: %s. "
                        "Cannot compile FP32 model for NPU.",
                        config_file,
                    )
                    return [("CPUExecutionProvider", {})]

                cache_dir = str(self.models_dir / "vai_cache")
                os.makedirs(cache_dir, exist_ok=True)

                vai_options = {
                    "config_file": config_file,
                    "cache_dir": cache_dir,
                    "cache_key": (model_name or "parakeet") + "_bf16",
                    "enable_cache_file_io_in_mem": "0",
                }

                logger.info(
                    "Using VitisAI EP BF16 path for %s (config=%s, cache=%s)",
                    model_name,
                    config_file,
                    cache_dir,
                )
            else:
                # INT8 compilation path: use target option
                cache_dir = str(self.models_dir / "vai_cache")
                os.makedirs(cache_dir, exist_ok=True)

                vai_options = {
                    "target": "X2",  # X2 for STX/KRK (Ryzen AI 300/Strix)
                    "cache_dir": cache_dir,
                    "cache_key": model_name or "parakeet",
                    "enable_cache_file_io_in_mem": "0",
                }

                logger.info(
                    "Using VitisAI EP INT8 path for %s (target=X2, cache=%s)",
                    model_name,
                    cache_dir,
                )

            return [
                ("VitisAIExecutionProvider", vai_options),
                ("CPUExecutionProvider", {}),
            ]

        elif self.device == "gpu":
            if "DmlExecutionProvider" not in available:
                logger.warning(
                    "DmlExecutionProvider not available! Available: %s. "
                    "Falling back to CPU.",
                    available,
                )
                return [("CPUExecutionProvider", {})]

            logger.info("Using DirectML EP for %s (AMD Radeon iGPU)", model_name)
            return [
                ("DmlExecutionProvider", {}),
                ("CPUExecutionProvider", {}),
            ]

        else:
            return [("CPUExecutionProvider", {})]

    def _create_sessions(self):
        """Create ONNX Runtime inference sessions for encoder and decoder.

        The encoder (large Conformer, ~600M params) runs on the target device
        (NPU/GPU/CPU). The decoder (tiny 27-node LSTM, called ~130x per chunk)
        always runs on CPU to avoid per-timestep NPU<->CPU transfer overhead.
        """
        encoder_path = self._find_model_file(
            "encoder-model.int8.onnx", "encoder-model.onnx"
        )
        decoder_path = self._find_model_file(
            "decoder_joint-model.int8.onnx", "decoder_joint-model.onnx"
        )

        # --- Encoder: runs on target device (NPU/GPU/CPU) ---
        logger.info("Loading encoder: %s", encoder_path)
        t0 = time.perf_counter()

        encoder_opts = self._get_session_options()
        if self.profile:
            encoder_opts.enable_profiling = True
            encoder_opts.profile_file_prefix = os.path.join(
                tempfile.gettempdir(), "parakeet_enc_profile"
            )
        encoder_providers = self._get_providers("parakeet_encoder", encoder_path)

        self._encoder_session = self.ort.InferenceSession(
            encoder_path,
            sess_options=encoder_opts,
            providers=[p[0] for p in encoder_providers],
            provider_options=[p[1] for p in encoder_providers],
        )

        t1 = time.perf_counter()
        logger.info("Encoder loaded in %.2fs", t1 - t0)

        # Log which providers are actually being used
        if self.debug:
            active = self._encoder_session.get_providers()
            logger.debug("Encoder active providers: %s", active)
            for inp in self._encoder_session.get_inputs():
                logger.debug("  Encoder input: %s type=%s shape=%s", inp.name, inp.type, inp.shape)
            for out in self._encoder_session.get_outputs():
                logger.debug("  Encoder output: %s type=%s shape=%s", out.name, out.type, out.shape)

        self._encoder_profiling_active = self.profile
        self._op_placement = None

        # --- Decoder: CPU by default, optionally iGPU (DirectML) ---
        # The decoder is a tiny LSTM (27 nodes) called per-timestep (~130x).
        # CPU is the safe default. iGPU (DirectML) is experimental --
        # the per-kernel-launch overhead may dominate for these tiny tensors,
        # but the iGPU runs in parallel with NPU and CPU.
        decoder_opts = self._get_session_options()

        if self.decoder_device == "gpu":
            available = self.ort.get_available_providers()
            if "DmlExecutionProvider" in available:
                decoder_providers = [
                    ("DmlExecutionProvider", {}),
                    ("CPUExecutionProvider", {}),
                ]
                dec_label = "iGPU (DirectML)"
            else:
                logger.warning(
                    "DmlExecutionProvider not available for decoder. "
                    "Falling back to CPU. Available: %s", available,
                )
                decoder_providers = [("CPUExecutionProvider", {})]
                dec_label = "CPU (DML unavailable)"
        else:
            # "auto" or "cpu" -> CPU
            decoder_providers = [("CPUExecutionProvider", {})]
            dec_label = "CPU"

        logger.info("Loading decoder: %s (%s)", decoder_path, dec_label)
        t0 = time.perf_counter()

        self._decoder_session = self.ort.InferenceSession(
            decoder_path,
            sess_options=decoder_opts,
            providers=[p[0] for p in decoder_providers],
            provider_options=[p[1] for p in decoder_providers],
        )

        t1 = time.perf_counter()
        logger.info("Decoder loaded in %.2fs (%s)", t1 - t0, dec_label)

        if self.debug:
            active = self._decoder_session.get_providers()
            logger.debug("Decoder active providers: %s", active)
            for inp in self._decoder_session.get_inputs():
                logger.debug("  Decoder input: %s type=%s shape=%s", inp.name, inp.type, inp.shape)
            for out in self._decoder_session.get_outputs():
                logger.debug("  Decoder output: %s type=%s shape=%s", out.name, out.type, out.shape)

        # Build a dtype map for decoder inputs so we cast correctly
        self._decoder_input_dtypes = {}
        _onnx_to_numpy = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(int32)": np.int32,
            "tensor(int64)": np.int64,
        }
        for inp in self._decoder_session.get_inputs():
            self._decoder_input_dtypes[inp.name] = _onnx_to_numpy.get(
                inp.type, np.float32
            )

        # Pre-allocate decoder buffers for the hot loop
        self._init_decoder_buffers()

    def transcribe(
        self,
        audio_data: bytes,
        audio_format: str = ".wav",
        language: str = "en",
    ) -> str:
        """
        Transcribe audio data to text.

        Automatically chunks long audio (>9 min) to stay within the model's
        maximum sequence length, with 2-second overlap between chunks.

        Args:
            audio_data: Raw audio file bytes.
            audio_format: Audio format extension (e.g. ".wav").
            language: ISO-639-1 language code (accepted but model is English-primary).

        Returns:
            Transcribed text string.
        """
        from .audio import parse_wav

        total_start = time.perf_counter()

        # Reset per-stage timing accumulator
        self._last_timings = {"mel": 0.0, "encoder": 0.0, "decoder": 0.0,
                               "decoder_steps": 0, "tokens": 0}

        # Step 1: Parse audio
        t0 = time.perf_counter()
        if audio_format in (".wav",):
            waveform = parse_wav(audio_data)
        else:
            raise ValueError(
                f"Unsupported format: {audio_format}. Convert to WAV first: "
                f"ffmpeg -i input{audio_format} -ar 16000 -ac 1 output.wav"
            )

        t1 = time.perf_counter()
        duration_sec = len(waveform) / 16000.0
        logger.debug(
            "Audio parsed: %d samples (%.2fs) in %.3fs",
            len(waveform),
            duration_sec,
            t1 - t0,
        )

        if len(waveform) < 1600:  # Less than 100ms
            logger.debug("Audio too short: %d samples", len(waveform))
            return ""

        # Step 2: Chunk long audio if needed
        if len(waveform) > self.MAX_CHUNK_SAMPLES:
            chunks = self._split_into_chunks(waveform)
            logger.info(
                "Audio too long (%.1fs), splitting into %d chunks of ~%ds each",
                duration_sec,
                len(chunks),
                self.MAX_CHUNK_SECONDS,
            )
        else:
            chunks = [waveform]

        # Step 3: Process chunks
        # For multi-chunk audio, pipeline encoder (NPU) and decoder (CPU):
        # While NPU encodes chunk N+1, CPU decodes chunk N in parallel.
        if len(chunks) > 1:
            all_texts = [self._stitch_chunks(self._process_chunks_pipelined(chunks))]
        else:
            all_texts = []
            chunk_start = time.perf_counter()
            chunk_text = self._transcribe_chunk(chunks[0], 0, 1)
            chunk_elapsed = time.perf_counter() - chunk_start
            if chunk_text:
                all_texts.append(chunk_text)
                logger.debug(
                    "Chunk 1/1 (%.1fs audio) transcribed in %.3fs: '%s'",
                    len(chunks[0]) / 16000.0,
                    chunk_elapsed,
                    chunk_text[:80] + "..." if len(chunk_text) > 80 else chunk_text,
                )

        # Combine chunk texts
        text = " ".join(all_texts)
        # Clean up any double spaces from joining
        text = re.sub(r"\s{2,}", " ", text).strip()

        total_elapsed = time.perf_counter() - total_start
        rtf = total_elapsed / duration_sec if duration_sec > 0 else 0
        logger.info(
            "Transcribed %.2fs audio in %.3fs (RTF=%.3f, device=%s, chunks=%d): '%s'",
            duration_sec,
            total_elapsed,
            rtf,
            self.device,
            len(chunks),
            text[:100] + "..." if len(text) > 100 else text,
        )

        return text

    STITCH_ANCHOR_TOLERANCE_S = 0.3  # max timestamp difference for the same word seen by two chunks

    def _stitch_chunks(self, chunk_tokens: list) -> str:
        """Join overlapping chunk transcripts using token timestamps.

        Neighboring chunks both transcribe their overlap. Each joint is cut at an anchor word that both chunks
        transcribed identically at about the same time, the one closest to the overlap's midpoint: the earlier
        chunk keeps the words before it, the later chunk the anchor and the words after it. Without an anchor,
        each chunk keeps the words that start on its side of the midpoint.

        Args:
            chunk_tokens: per chunk, in order, (token IDs, encoder frame of each token).
        """
        step = (self.MAX_CHUNK_SAMPLES - self.OVERLAP_SAMPLES) / 16000
        overlap = self.OVERLAP_SAMPLES / 16000
        kept = []
        for i, (tokens, frames) in enumerate(chunk_tokens):
            words = self._words_with_times(tokens, frames, offset=i * step)
            if i == 0:
                kept = words
                continue
            overlap_start, overlap_end = i * step, i * step + overlap
            midpoint = overlap_start + overlap / 2
            cut_prev, cut_next = None, None
            best_distance = float("inf")
            for a, (text_a, start_a, _) in enumerate(kept):
                if not (text_a and overlap_start <= start_a < overlap_end):
                    continue
                for b, (text_b, start_b, _) in enumerate(words):
                    if start_b >= overlap_end:
                        break
                    if text_a == text_b and abs(start_a - start_b) <= self.STITCH_ANCHOR_TOLERANCE_S:
                        distance = abs((start_a + start_b) / 2 - midpoint)
                        if distance < best_distance:
                            best_distance, cut_prev, cut_next = distance, a, b
            if cut_prev is None:
                cut_prev = next((a for a, w in enumerate(kept) if w[1] >= midpoint), len(kept))
                cut_next = next((b for b, w in enumerate(words) if w[1] >= midpoint), len(words))
            kept = kept[:cut_prev] + words[cut_next:]
        return self._tokens_to_text([token for _, _, word_tokens in kept for token in word_tokens])

    def _words_with_times(self, tokens: list, frames: list, offset: float) -> list:
        """Group tokens into words: (normalized text, start time in seconds from the audio start, token IDs)."""
        words = []
        for token, frame in zip(tokens, frames):
            piece = self.vocab.get(token, "")
            if piece.startswith(" ") or not words:
                words.append(["", offset + frame * self.ENCODER_FRAME_SECONDS, []])
            words[-1][0] += re.sub(r"[^\w']", "", piece).lower()
            words[-1][2].append(token)
        return [tuple(w) for w in words]

    def _split_into_chunks(self, waveform: np.ndarray) -> list:
        """Split a long waveform into overlapping chunks."""
        chunks = []
        step = self.MAX_CHUNK_SAMPLES - self.OVERLAP_SAMPLES
        offset = 0

        while offset < len(waveform):
            end = min(offset + self.MAX_CHUNK_SAMPLES, len(waveform))
            chunk = waveform[offset:end]
            # Skip very short trailing chunks (< 0.5s)
            if len(chunk) >= 8000:
                chunks.append(chunk)
            offset += step

        return chunks

    def _process_chunks_pipelined(self, chunks: list) -> list:
        """Process multiple chunks with encoder/decoder pipelining.

        While the NPU encodes chunk N+1, the CPU decodes chunk N.
        For K chunks this hides (K-1) decoder invocations behind encoder time.

        Timeline:
            Chunk 0: [mel+encode]─────────────[decode]
            Chunk 1:              [mel+encode]─────────────[decode]
            Chunk 2:                           [mel+encode]─────────────[decode]
                                  ^── decoder for chunk 0 runs here, overlapped
        """
        import concurrent.futures

        total = len(chunks)
        all_tokens = [([], [])] * total  # Pre-sized for ordered results: (tokens, frames) per chunk
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        pending_decode_future = None
        pending_decode_idx = None

        for chunk_idx, chunk in enumerate(chunks):
            # Step 1: Mel features (CPU, fast)
            t0 = time.perf_counter()
            features = self.mel.extract(chunk)
            mel_time = time.perf_counter() - t0

            if features is None or len(features) == 0:
                logger.warning("Chunk %d: no features extracted", chunk_idx)
                # Still need to collect any pending decode
                if pending_decode_future is not None:
                    all_tokens[pending_decode_idx] = pending_decode_future.result()
                    pending_decode_future = None
                continue

            self._last_timings["mel"] += mel_time

            # Step 2: Encoder (NPU) -- this is the long operation
            t0 = time.perf_counter()
            encoder_out, encoded_len = self._run_encoder(features)
            encoder_time = time.perf_counter() - t0
            self._last_timings["encoder"] += encoder_time

            logger.debug(
                "Chunk %d/%d: mel=%.1fms enc=%.1fms encoded_len=%d",
                chunk_idx + 1, total, mel_time * 1000, encoder_time * 1000, encoded_len,
            )

            # Step 3: Collect previous chunk's decode result (if any)
            if pending_decode_future is not None:
                all_tokens[pending_decode_idx] = pending_decode_future.result()
                pending_decode_future = None

            # Step 4: Launch decoder in background thread
            # (runs on CPU, overlaps with next chunk's encoder on NPU)
            pending_decode_idx = chunk_idx
            pending_decode_future = executor.submit(
                self._decode_and_time, encoder_out, encoded_len, chunk_idx, total,
            )

        # Collect the last chunk's decode result
        if pending_decode_future is not None:
            all_tokens[pending_decode_idx] = pending_decode_future.result()

        executor.shutdown(wait=False)
        return all_tokens

    def _decode_and_time(self, encoder_out, encoded_len, chunk_idx, total_chunks) -> tuple:
        """Run decoder with timing, for use in pipelined processing. Returns (tokens, frames)."""
        t0 = time.perf_counter()
        frames = []
        tokens = self._tdt_decode(encoder_out, encoded_len, frames)
        decoder_time = time.perf_counter() - t0

        self._last_timings["decoder"] += decoder_time
        self._last_timings["decoder_steps"] += encoded_len
        self._last_timings["tokens"] += len(tokens)

        logger.debug(
            "Chunk %d/%d decode: %d tokens in %.1fms",
            chunk_idx + 1, total_chunks, len(tokens), decoder_time * 1000,
        )
        return tokens, frames

    def _transcribe_chunk(self, waveform: np.ndarray, chunk_idx: int = 0, total_chunks: int = 1) -> str:
        """Transcribe a single audio chunk (must be within model max length).

        Timing data is accumulated in self._last_timings dict for benchmarking.
        """
        # Extract mel features
        t0 = time.perf_counter()
        features = self.mel.extract(waveform)
        if features is None or len(features) == 0:
            logger.warning("Chunk %d: no features extracted", chunk_idx)
            return ""
        mel_time = time.perf_counter() - t0
        logger.debug(
            "Chunk %d/%d mel: %d frames x %d features in %.3fs",
            chunk_idx + 1, total_chunks,
            features.shape[0], features.shape[1], mel_time,
        )

        # Run encoder
        t0 = time.perf_counter()
        encoder_out, encoded_len = self._run_encoder(features)
        encoder_time = time.perf_counter() - t0
        logger.debug(
            "Chunk %d/%d encoder: shape %s, encoded_len=%d in %.3fs",
            chunk_idx + 1, total_chunks,
            encoder_out.shape, encoded_len, encoder_time,
        )

        # Run TDT decoder
        t0 = time.perf_counter()
        tokens = self._tdt_decode(encoder_out, encoded_len)
        decoder_time = time.perf_counter() - t0
        logger.debug(
            "Chunk %d/%d decoder: %d tokens in %.3fs",
            chunk_idx + 1, total_chunks, len(tokens), decoder_time,
        )

        # Accumulate per-stage timings
        if not hasattr(self, '_last_timings'):
            self._last_timings = {"mel": 0.0, "encoder": 0.0, "decoder": 0.0,
                                   "decoder_steps": 0, "tokens": 0}
        self._last_timings["mel"] += mel_time
        self._last_timings["encoder"] += encoder_time
        self._last_timings["decoder"] += decoder_time
        self._last_timings["decoder_steps"] += encoded_len
        self._last_timings["tokens"] += len(tokens)

        return self._tokens_to_text(tokens)

    def _run_encoder(self, features: np.ndarray) -> tuple:
        """
        Run the encoder model.

        Args:
            features: Mel features of shape [num_frames, n_mels].

        Returns:
            Tuple of (encoder_output, encoded_length).
            encoder_output shape: [1, encoder_dim, encoded_len]
        """
        num_frames = features.shape[0]
        actual_frames = num_frames

        # For static models (NPU), pad to the fixed frame count
        if self._fixed_frames is not None:
            if num_frames > self._fixed_frames:
                # Truncate (shouldn't happen if chunking is correct)
                logger.warning(
                    "Truncating %d frames to %d for static model",
                    num_frames, self._fixed_frames,
                )
                features = features[: self._fixed_frames]
                actual_frames = self._fixed_frames
            elif num_frames < self._fixed_frames:
                # Pad with zeros
                pad_amount = self._fixed_frames - num_frames
                features = np.pad(
                    features,
                    ((0, pad_amount), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
                logger.debug(
                    "Padded %d -> %d frames for static encoder",
                    num_frames, self._fixed_frames,
                )
            num_frames = self._fixed_frames

        # Prepare input: [batch=1, n_mels, num_frames] (transposed)
        input_data = features.T[np.newaxis, :, :].astype(np.float32)  # [1, 128, num_frames]
        length_data = np.array([actual_frames], dtype=np.int64)

        outputs = self._encoder_session.run(
            None,
            {
                "audio_signal": input_data,
                "length": length_data,
            },
        )

        encoder_out = outputs[0]  # [1, encoder_dim, encoded_len]
        encoded_len = int(outputs[1][0])  # scalar

        if self._encoder_profiling_active:
            self._parse_encoder_profile()

        return encoder_out, encoded_len

    def _parse_encoder_profile(self):
        """Parse ORT profiling output to summarize encoder op placement across EPs.

        Called once after the first encoder run when profile=True.
        """
        if not self._encoder_profiling_active:
            return
        self._encoder_profiling_active = False

        try:
            profile_path = self._encoder_session.end_profiling()
            with open(profile_path, "r") as f:
                profile_data = json.load(f)

            ep_counts = {}
            op_types_by_ep = {}

            for event in profile_data:
                if event.get("cat") != "Node":
                    continue
                name = event.get("name", "")
                if "_kernel_time" not in name:
                    continue

                args = event.get("args", {})
                provider = args.get("provider", "unknown")
                op_name = args.get("op_name", "unknown")

                ep_counts[provider] = ep_counts.get(provider, 0) + 1
                if provider not in op_types_by_ep:
                    op_types_by_ep[provider] = {}
                op_types_by_ep[provider][op_name] = (
                    op_types_by_ep[provider].get(op_name, 0) + 1
                )

            total = sum(ep_counts.values())
            self._op_placement = {
                "total_ops": total,
                "by_provider": ep_counts,
                "op_types_by_provider": op_types_by_ep,
            }

            logger.info("Encoder op placement (%d total ops):", total)
            for ep, count in sorted(ep_counts.items(), key=lambda x: -x[1]):
                pct = count / total * 100 if total > 0 else 0
                logger.info("  %s: %d ops (%.1f%%)", ep, count, pct)

            if self.debug:
                for ep in sorted(ep_counts.keys()):
                    ops = op_types_by_ep.get(ep, {})
                    for op, cnt in sorted(ops.items(), key=lambda x: -x[1])[:10]:
                        short_ep = ep.replace("ExecutionProvider", "")
                        logger.debug("    [%s] %s: %d", short_ep, op, cnt)

            try:
                os.remove(profile_path)
            except OSError:
                pass

        except Exception as e:
            logger.warning("Failed to parse encoder profile: %s", e)

    def get_op_placement(self) -> Optional[dict]:
        """Return encoder op placement data from profiling.

        Available after the first transcription when profile=True.
        Returns dict with keys: total_ops, by_provider, op_types_by_provider.
        Returns None if profiling was not enabled or hasn't run yet.
        """
        return getattr(self, '_op_placement', None)

    def _init_decoder_buffers(self):
        """Pre-allocate reusable buffers for the decoder hot loop.

        Called once after sessions are created. Avoids per-step numpy
        array allocation in _tdt_decode, which runs ~130-14K iterations.

        Also sets up ORT IO Binding to bypass Python dict / numpy->ORT
        conversion overhead on each of those iterations.
        """
        self._dec_enc_slice = np.empty(
            (1, self.encoder_dim, 1), dtype=np.float32
        )
        self._dec_targets = np.array([[self.blank_idx]], dtype=np.int32)
        self._dec_target_length = np.array([1], dtype=np.int32)
        self._dec_state1 = np.zeros(
            (self.num_lstm_layers, 1, self.state_dim), dtype=np.float32
        )
        self._dec_state2 = np.zeros(
            (self.num_lstm_layers, 1, self.state_dim), dtype=np.float32
        )
        # Cache input/output names to avoid repeated attribute lookups
        self._dec_input_names = [i.name for i in self._decoder_session.get_inputs()]
        self._dec_output_names = [o.name for o in self._decoder_session.get_outputs()]

        # IO Binding: benchmarking shows the dict path is fastest for both
        # CPU and DML decoders. DML IO binding with GPU-resident states was
        # tested but the per-call bind/OrtValue overhead exceeds savings
        # for these tiny LSTM tensors.
        self._use_io_binding = False
        self._use_dml_io_binding = False
        self._dec_io_binding = None

    def _tdt_decode(self, encoder_out: np.ndarray, encoded_len: int, frames: Optional[list] = None) -> list:
        """
        Run Token-and-Duration Transducer (TDT) greedy decoding.

        Optimized hot loop: all input arrays are pre-allocated and mutated
        in-place. No per-step numpy allocation or Python dict creation.

        Args:
            encoder_out: Encoder output of shape [1, encoder_dim, encoded_len].
            encoded_len: Actual encoded sequence length.
            frames: If given, the encoder frame index of each decoded token is appended to it.

        Returns:
            List of decoded token IDs.
        """
        tokens = []
        timestep = 0
        emitted_tokens = 0
        blank_idx = self.blank_idx
        vocab_size = self.vocab_size
        max_tokens = self.max_tokens_per_step

        # Reset pre-allocated buffers
        enc_slice = self._dec_enc_slice
        targets = self._dec_targets
        target_length = self._dec_target_length
        state1 = self._dec_state1
        state2 = self._dec_state2
        targets[0, 0] = blank_idx
        state1[:] = 0.0
        state2[:] = 0.0

        # Pre-squeeze encoder output for faster indexing: [encoder_dim, encoded_len]
        enc_2d = encoder_out[0]  # [encoder_dim, encoded_len] -- no copy, just view

        _argmax = np.argmax

        if self._use_io_binding:
            # ─── IO Binding path (zero-copy) ───────────────────────────
            io = self._dec_io_binding
            session = self._decoder_session
            input_names = self._dec_input_names

            while timestep < encoded_len:
                enc_slice[0, :, 0] = enc_2d[:, timestep]

                # Bind numpy arrays directly (avoids dict creation + ORT value conversion)
                io.bind_cpu_input(input_names[0], enc_slice)
                io.bind_cpu_input(input_names[1], targets)
                io.bind_cpu_input(input_names[2], target_length)
                io.bind_cpu_input(input_names[3], state1)
                io.bind_cpu_input(input_names[4], state2)

                session.run_with_iobinding(io)
                outputs = io.copy_outputs_to_cpu()

                logits = outputs[0].ravel()
                token = int(_argmax(logits[:vocab_size]))
                step = int(_argmax(logits[vocab_size:]))

                if token != blank_idx:
                    np.copyto(state1, outputs[2])
                    np.copyto(state2, outputs[3])
                    tokens.append(token)
                    if frames is not None:
                        frames.append(timestep)
                    targets[0, 0] = token
                    emitted_tokens += 1

                if step > 0:
                    timestep += step
                    emitted_tokens = 0
                elif token == blank_idx or emitted_tokens >= max_tokens:
                    timestep += 1
                    emitted_tokens = 0
        else:
            # ─── Dict fallback path ────────────────────────────────────
            session_run = self._decoder_session.run
            input_names = self._dec_input_names
            feed = {
                input_names[0]: enc_slice,
                input_names[1]: targets,
                input_names[2]: target_length,
                input_names[3]: state1,
                input_names[4]: state2,
            }

            while timestep < encoded_len:
                enc_slice[0, :, 0] = enc_2d[:, timestep]

                outputs = session_run(None, feed)

                logits = outputs[0].ravel()
                token = int(_argmax(logits[:vocab_size]))
                step = int(_argmax(logits[vocab_size:]))

                if token != blank_idx:
                    np.copyto(state1, outputs[2])
                    np.copyto(state2, outputs[3])
                    tokens.append(token)
                    if frames is not None:
                        frames.append(timestep)
                    targets[0, 0] = token
                    emitted_tokens += 1

                if step > 0:
                    timestep += step
                    emitted_tokens = 0
                elif token == blank_idx or emitted_tokens >= max_tokens:
                    timestep += 1
                    emitted_tokens = 0

        return tokens

    def _tokens_to_text(self, tokens: list) -> str:
        """Convert token IDs to text string."""
        parts = []
        for tok_id in tokens:
            text = self.vocab.get(tok_id, "")
            # Skip special tokens like <unk>, <s>, etc.
            if text.startswith("<") and text.endswith(">"):
                continue
            parts.append(text)

        text = "".join(parts)

        # Clean up spacing
        text = re.sub(r"^\s+|\s+$", "", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    def get_last_timings(self) -> Optional[dict]:
        """Return per-stage timings from the last transcribe() call.

        Returns dict with keys: mel, encoder, decoder (seconds),
        decoder_steps, tokens (counts). Returns None if no transcription
        has been run yet.
        """
        return getattr(self, '_last_timings', None)

    def close(self):
        """Clean up ONNX Runtime sessions."""
        self._encoder_session = None
        self._decoder_session = None
        logger.info("Transcriber closed")

    def get_info(self) -> dict:
        """Return information about the transcriber configuration."""
        available_providers = self.ort.get_available_providers()
        active_encoder = (
            self._encoder_session.get_providers() if self._encoder_session else []
        )
        active_decoder = (
            self._decoder_session.get_providers() if self._decoder_session else []
        )

        return {
            "device": self.device,
            "models_dir": str(self.models_dir),
            "vocab_size": self.vocab_size,
            "encoder_dim": self.encoder_dim,
            "available_providers": available_providers,
            "encoder_providers": active_encoder,
            "decoder_providers": active_decoder,
            "onnxruntime_version": self.ort.__version__,
        }
