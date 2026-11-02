"""
Mel filterbank feature extraction.

Ported from the Go implementation in achetronic/parakeet.
Uses NeMo-compatible parameters for 128-bin mel features:
  - n_fft=512, hop_length=160 (10ms), win_length=400 (25ms)
  - Hann window, power spectrum, log mel energy
  - Per-feature mean/variance normalization
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class MelFilterbank:
    """Compute mel-scale filterbank features from audio waveforms."""

    def __init__(
        self,
        n_mels: int = 128,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
    ):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # Pre-compute the mel filterbank matrix
        self.filterbank = self._create_mel_filterbank()  # [n_mels, n_fft//2+1]

        # Pre-compute the Hann window
        self.window = np.hanning(win_length).astype(np.float64)

    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _create_mel_filterbank(self) -> np.ndarray:
        """Create triangular mel filterbank matrix [n_mels, n_fft//2+1]."""
        num_bins = self.n_fft // 2 + 1
        mel_min = self._hz_to_mel(0)
        mel_max = self._hz_to_mel(self.sample_rate / 2)

        # Linearly spaced mel points
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)

        # Convert to frequency bin indices
        hz_points = np.array([self._mel_to_hz(m) for m in mel_points])
        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)

        # Build filterbank
        filterbank = np.zeros((self.n_mels, num_bins), dtype=np.float64)
        for i in range(self.n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]

            # Rising slope
            for j in range(left, min(center, num_bins)):
                if center > left:
                    filterbank[i, j] = (j - left) / (center - left)

            # Falling slope
            for j in range(center, min(right, num_bins)):
                if right > center:
                    filterbank[i, j] = (right - j) / (right - center)

        return filterbank

    def extract(self, samples: np.ndarray) -> np.ndarray:
        """
        Compute mel filterbank features from audio samples.

        Vectorized: all frames are windowed, FFT'd, and projected through
        the mel filterbank in bulk numpy operations (no Python loops).

        Args:
            samples: float32 numpy array of audio samples at 16kHz.

        Returns:
            numpy array of shape [num_frames, n_mels] with normalized log mel features.
            Returns None if audio is too short.
        """
        num_frames = (len(samples) - self.win_length) // self.hop_length + 1
        if num_frames <= 0:
            logger.debug(
                "Mel: not enough samples for even one frame (samples=%d, winLength=%d)",
                len(samples),
                self.win_length,
            )
            return None

        # Build frame indices: [num_frames, win_length]
        frame_starts = np.arange(num_frames) * self.hop_length
        frame_indices = frame_starts[:, np.newaxis] + np.arange(self.win_length)

        # Extract all frames at once and apply window
        frames = samples[frame_indices].astype(np.float64) * self.window  # [num_frames, win_length]

        # Zero-pad to n_fft if needed
        if self.win_length < self.n_fft:
            pad_width = self.n_fft - self.win_length
            frames = np.pad(frames, ((0, 0), (0, pad_width)), mode="constant")

        # Vectorized FFT -> power spectrum for all frames at once
        spectrum = np.fft.rfft(frames, n=self.n_fft, axis=1)  # [num_frames, n_fft//2+1]
        power = np.abs(spectrum) ** 2  # [num_frames, n_fft//2+1]

        # Apply mel filterbank: [num_frames, n_fft//2+1] @ [n_fft//2+1, n_mels] -> [num_frames, n_mels]
        mel_energies = power @ self.filterbank.T
        mel_energies = np.maximum(mel_energies, 1e-10)
        features = np.log(mel_energies).astype(np.float32)

        # Per-feature normalization (zero mean, unit variance)
        self._normalize(features)

        return features

    @staticmethod
    def _normalize(features: np.ndarray) -> None:
        """In-place per-feature mean/variance normalization."""
        if len(features) == 0:
            return

        means = features.mean(axis=0)
        stds = features.std(axis=0)
        stds = np.maximum(stds, 1e-10)

        features -= means
        features /= stds
