"""Transcribe a long file with Parakeet and count immediately repeated phrases (chunk-stitching duplicates).

Usage (from C:\\Dev\\Parakeet): python diarization/diag_repeats.py ted.wav out.txt [--device npu]
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def immediate_repeats(text, min_words=3, max_words=12):
    """(start word index, phrase) for every phrase of min_words..max_words words directly followed by itself."""
    words = [re.sub(r"[^\w']", "", w).lower() for w in text.split()]
    found, i = [], 0
    while i < len(words):
        for n in range(max_words, min_words - 1, -1):
            if words[i : i + n] == words[i + n : i + 2 * n] and all(words[i : i + n]):
                found.append((i, " ".join(words[i : i + n])))
                i += n
                break
        else:
            i += 1
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio")
    parser.add_argument("out")
    parser.add_argument("--device", default="npu")
    args = parser.parse_args()

    from inference import Transcriber

    transcriber = Transcriber(models_dir=str(ROOT / "models"), device=args.device,
                              decoder_device="gpu" if args.device == "npu" else "auto")
    text = transcriber.transcribe(Path(args.audio).read_bytes(), audio_format=".wav")
    transcriber.close()
    Path(args.out).write_text(text, encoding="utf-8")
    repeats = immediate_repeats(text)
    print(f"{len(text.split())} words, {len(repeats)} immediate repeats -> {args.out}")
    for _, phrase in repeats[:15]:
        print(f"  {phrase}")


if __name__ == "__main__":
    main()
