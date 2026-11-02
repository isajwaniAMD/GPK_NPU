"""
WAV audio parsing and resampling.

Ported from the Go implementation in achetronic/parakeet.
Supports 8/16/24/32-bit PCM and 32-bit float WAV files.
Handles stereo-to-mono conversion and resampling to 16kHz.
"""

import struct
import logging
import numpy as np

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000


def parse_wav(data: bytes) -> np.ndarray:
    """
    Parse a WAV file and return float32 samples normalized to [-1, 1] at 16kHz mono.

    Args:
        data: Raw WAV file bytes.

    Returns:
        numpy array of float32 samples at 16kHz, mono.

    Raises:
        ValueError: If the file is not a valid WAV or uses an unsupported format.
    """
    if len(data) < 44:
        raise ValueError("WAV file too small")

    # Check RIFF header
    if data[0:4] != b"RIFF":
        raise ValueError("Not a RIFF file")
    if data[8:12] != b"WAVE":
        raise ValueError("Not a WAVE file")

    # Parse chunks
    offset = 12
    audio_format = None
    num_channels = None
    sample_rate = None
    bits_per_sample = None

    while offset < len(data) - 8:
        chunk_id = data[offset : offset + 4].decode("ascii", errors="replace")
        chunk_size = struct.unpack_from("<I", data, offset + 4)[0]

        if chunk_id == "fmt ":
            if chunk_size < 16:
                raise ValueError("fmt chunk too small")
            audio_format = struct.unpack_from("<H", data, offset + 8)[0]
            num_channels = struct.unpack_from("<H", data, offset + 10)[0]
            sample_rate = struct.unpack_from("<I", data, offset + 12)[0]
            # byte_rate at offset+16, block_align at offset+20 (unused)
            bits_per_sample = struct.unpack_from("<H", data, offset + 22)[0]

        elif chunk_id == "data":
            if audio_format is None:
                raise ValueError("data chunk found before fmt chunk")

            data_start = offset + 8
            data_end = min(data_start + chunk_size, len(data))
            audio_data = data[data_start:data_end]

            logger.debug(
                "WAV: format=%d, channels=%d, sampleRate=%d, bitsPerSample=%d, dataSize=%d",
                audio_format,
                num_channels,
                sample_rate,
                bits_per_sample,
                len(audio_data),
            )

            # Convert to float32 mono
            samples = _convert_to_float32(
                audio_data, audio_format, num_channels, bits_per_sample
            )

            # Resample to 16kHz if needed
            if sample_rate != TARGET_SAMPLE_RATE:
                logger.debug(
                    "Resampling from %d Hz to %d Hz (%d -> %d samples)",
                    sample_rate,
                    TARGET_SAMPLE_RATE,
                    len(samples),
                    int(len(samples) * TARGET_SAMPLE_RATE / sample_rate),
                )
                samples = _resample(samples, sample_rate, TARGET_SAMPLE_RATE)

            return samples

        # Move to next chunk (chunks are 2-byte aligned)
        offset += 8 + chunk_size
        if chunk_size % 2 != 0:
            offset += 1

    raise ValueError("No data chunk found in WAV file")


def _convert_to_float32(
    data: bytes, audio_format: int, num_channels: int, bits_per_sample: int
) -> np.ndarray:
    """Convert raw audio bytes to float32 samples normalized to [-1, 1]."""
    if audio_format not in (1, 3):  # 1=PCM, 3=IEEE float
        raise ValueError(
            f"Unsupported audio format: {audio_format} (only PCM and IEEE float supported)"
        )

    bytes_per_sample = bits_per_sample // 8
    num_samples = len(data) // (bytes_per_sample * num_channels)

    if bits_per_sample == 16 and audio_format == 1:
        # Fast path for 16-bit PCM (most common)
        raw = np.frombuffer(data[: num_samples * num_channels * 2], dtype=np.int16)
        raw = raw.reshape(-1, num_channels).astype(np.float32)
        samples = raw.mean(axis=1) / 32768.0

    elif bits_per_sample == 32 and audio_format == 3:
        # Fast path for 32-bit float
        raw = np.frombuffer(data[: num_samples * num_channels * 4], dtype=np.float32)
        samples = raw.reshape(-1, num_channels).mean(axis=1)

    elif bits_per_sample == 8:
        # Unsigned 8-bit PCM
        raw = np.frombuffer(data[: num_samples * num_channels], dtype=np.uint8)
        raw = raw.reshape(-1, num_channels).astype(np.float32)
        samples = raw.mean(axis=1) / 128.0 - 1.0

    elif bits_per_sample == 24:
        # 24-bit PCM - needs manual unpacking
        samples = np.zeros(num_samples, dtype=np.float32)
        for i in range(num_samples):
            channel_sum = 0.0
            for ch in range(num_channels):
                off = (i * num_channels + ch) * 3
                if off + 3 > len(data):
                    break
                b0, b1, b2 = data[off], data[off + 1], data[off + 2]
                val = b0 | (b1 << 8) | (b2 << 16)
                if val & 0x800000:
                    val |= ~0xFFFFFF  # sign extend
                channel_sum += val / 8388608.0
            samples[i] = channel_sum / num_channels

    elif bits_per_sample == 32 and audio_format == 1:
        # 32-bit signed PCM
        raw = np.frombuffer(data[: num_samples * num_channels * 4], dtype=np.int32)
        raw = raw.reshape(-1, num_channels).astype(np.float64)
        samples = (raw.mean(axis=1) / 2147483648.0).astype(np.float32)

    else:
        raise ValueError(f"Unsupported bits per sample: {bits_per_sample}")

    return samples


def _resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample audio using linear interpolation."""
    if src_rate == dst_rate:
        return samples

    ratio = src_rate / dst_rate
    new_len = int(len(samples) / ratio)
    indices = np.arange(new_len) * ratio
    lo = indices.astype(np.int64)
    hi = np.minimum(lo + 1, len(samples) - 1)
    frac = (indices - lo).astype(np.float32)
    result = samples[lo] * (1 - frac) + samples[hi] * frac
    return result
