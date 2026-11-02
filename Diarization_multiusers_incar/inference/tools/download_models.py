#!/usr/bin/env python3
"""
Download Parakeet TDT 0.6B ONNX models from HuggingFace.

Models by NVIDIA, ONNX conversion by Ivan Stupakov (@istupakov).
Source: https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx
"""

import argparse
import os
import sys
import urllib.request
import time

BASE_URL = "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main"

# Common files needed for all model types
COMMON_FILES = [
    ("config.json", "Model configuration"),
    ("vocab.txt", "SentencePiece vocabulary"),
    ("nemo128.onnx", "Mel filterbank ONNX model"),
]

# INT8 quantized models (recommended, ~670MB total)
INT8_FILES = [
    ("encoder-model.int8.onnx", "Quantized encoder (~652MB)"),
    ("decoder_joint-model.int8.onnx", "Quantized TDT decoder (~18MB)"),
]

# FP32 full precision models (~2.5GB total)
FP32_FILES = [
    ("encoder-model.onnx", "Full precision encoder (~2.4GB)"),
    ("encoder-model.onnx.data", "Encoder external data"),
    ("decoder_joint-model.onnx", "Full precision TDT decoder (~72MB)"),
]


def format_size(size_bytes: int) -> str:
    """Format bytes into human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def download_file(url: str, dest: str, description: str) -> bool:
    """Download a file with progress reporting."""
    if os.path.exists(dest):
        size = os.path.getsize(dest)
        print(f"  [SKIP] {os.path.basename(dest)} already exists ({format_size(size)})")
        return True

    print(f"  [DOWN] {description}: {os.path.basename(dest)}")
    tmp_dest = dest + ".tmp"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "parakeet-downloader/1.0"})
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 1024 * 1024  # 1MB chunks
            start_time = time.time()

            with open(tmp_dest, "wb") as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        pct = downloaded / total_size * 100
                        elapsed = time.time() - start_time
                        speed = downloaded / elapsed if elapsed > 0 else 0
                        eta = (total_size - downloaded) / speed if speed > 0 else 0
                        print(
                            f"\r         {format_size(downloaded)} / {format_size(total_size)} "
                            f"({pct:.1f}%) - {format_size(speed)}/s - ETA {eta:.0f}s",
                            end="",
                            flush=True,
                        )
                    else:
                        print(f"\r         {format_size(downloaded)} downloaded", end="", flush=True)

            print()  # newline after progress

        # Rename tmp to final
        os.replace(tmp_dest, dest)
        elapsed = time.time() - start_time
        print(f"         Done in {elapsed:.1f}s")
        return True

    except Exception as e:
        print(f"\n  [ERR]  Failed to download {os.path.basename(dest)}: {e}")
        if os.path.exists(tmp_dest):
            os.remove(tmp_dest)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download Parakeet TDT 0.6B ONNX models from HuggingFace"
    )
    parser.add_argument(
        "--precision",
        choices=["int8", "fp32"],
        default="int8",
        help="Model precision: int8 (recommended, ~670MB) or fp32 (~2.5GB). Default: int8",
    )
    parser.add_argument(
        "--output-dir",
        default="./models",
        help="Output directory for model files. Default: ./models",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    model_files = INT8_FILES if args.precision == "int8" else FP32_FILES
    all_files = COMMON_FILES + model_files

    total_files = len(all_files)
    print(f"Parakeet TDT 0.6B ONNX Model Downloader")
    print(f"========================================")
    print(f"Precision : {args.precision}")
    print(f"Output    : {os.path.abspath(args.output_dir)}")
    print(f"Files     : {total_files}")
    print()

    # Remove existing files if --force
    if args.force:
        for filename, _ in all_files:
            dest = os.path.join(args.output_dir, filename)
            if os.path.exists(dest):
                os.remove(dest)
                print(f"  Removed {filename}")
        print()

    success = 0
    failed = 0

    for filename, description in all_files:
        url = f"{BASE_URL}/{filename}"
        dest = os.path.join(args.output_dir, filename)
        if download_file(url, dest, description):
            success += 1
        else:
            failed += 1

    print()
    print(f"Download complete: {success} succeeded, {failed} failed")

    if failed > 0:
        print("Some files failed to download. Re-run the script to retry.")
        sys.exit(1)
    else:
        print(f"Models ready in: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
