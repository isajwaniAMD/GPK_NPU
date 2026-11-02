"""Describe the default microphone and record a few seconds of all its channels, to check what live capture gets.

Usage: python diag_mic.py [seconds]   (speak during the recording)
"""

import sys

import numpy as np
import sounddevice as sd


def main():
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    device = sd.query_devices(sd.default.device[0], "input")
    host = sd.query_hostapis(device["hostapi"])["name"]
    print(f"default input: {device['name']} | host API {host} | {device['max_input_channels']} ch | "
          f"native rate {device['default_samplerate']:.0f} Hz")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            print(f"  [{i:2d}] {dev['name'][:48]:48s} {sd.query_hostapis(dev['hostapi'])['name']:24s} "
                  f"ch={dev['max_input_channels']} rate={dev['default_samplerate']:.0f}")

    channels = min(int(device["max_input_channels"]), 4)
    print(f"\nrecording {seconds:.0f}s at 16 kHz, {channels} channels...")
    audio = sd.rec(int(16000 * seconds), samplerate=16000, channels=channels, dtype="float32")
    sd.wait()
    rms = np.sqrt((audio**2).mean(axis=0))
    peak = np.abs(audio).max(axis=0)
    print("per-channel RMS :", " ".join(f"{r:.4f}" for r in rms))
    print("per-channel peak:", " ".join(f"{p:.3f}" for p in peak))
    if channels > 1:
        corr = np.corrcoef(audio.T)
        print("channel correlation:\n" + "\n".join("  " + " ".join(f"{c:5.2f}" for c in row) for row in corr))
        mono = audio.mean(axis=1)
        print(f"mean of channels RMS {np.sqrt((mono**2).mean()):.4f} vs channel 0 RMS {rms[0]:.4f} "
              "(much lower means the channels partly cancel when averaged)")
    clipped = (np.abs(audio) >= 0.999).mean() * 100
    print(f"clipped samples: {clipped:.3f}%")


if __name__ == "__main__":
    main()
