#!/usr/bin/env python3
"""
Optimize the FP32 static encoder model for NPU compilation.

Uses ONNX Runtime's graph optimization to fold constants (turning dynamically-
computed pad values into actual constants), then fuses Pad+Conv pairs.

This is needed because:
1. The Conformer encoder computes padding amounts dynamically from input shapes
2. With static shapes, these become constant computations
3. ORT's constant folding resolves them to actual values
4. Then we can fuse the Pad ops into Conv ops (required for NPU/AIE compiler)
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def fold_constants_with_ort(input_path, output_path):
    """Use ONNX Runtime to fold constants in the model."""
    import onnxruntime as ort

    print("  Running ONNX Runtime constant folding...")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.optimized_model_filepath = output_path

    # Create session just to trigger optimization + save
    session = ort.InferenceSession(
        input_path,
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    del session
    print(f"  Saved optimized model: {output_path}")


def count_ops(model):
    """Count op types in model."""
    from collections import Counter
    return Counter(n.op_type for n in model.graph.node)


def fuse_pad_conv_in_model(model):
    """Fuse Pad+Conv pairs. Pad values should be constants after ORT folding."""
    # Build lookup maps
    init_map = {}
    for init in model.graph.initializer:
        init_map[init.name] = init

    const_node_map = {}
    for node in model.graph.node:
        if node.op_type == "Constant":
            for out in node.output:
                const_node_map[out] = node

    output_to_consumers = {}
    for node in model.graph.node:
        for inp in node.input:
            if inp not in output_to_consumers:
                output_to_consumers[inp] = []
            output_to_consumers[inp].append(node)

    def get_value(name):
        if name in init_map:
            return numpy_helper.to_array(init_map[name])
        if name in const_node_map:
            n = const_node_map[name]
            for attr in n.attribute:
                if attr.name == "value" and attr.t is not None:
                    return numpy_helper.to_array(attr.t)
        return None

    nodes_to_remove = set()
    fused = 0

    for node in model.graph.node:
        if node.op_type != "Pad":
            continue

        pad_output = node.output[0]
        consumers = output_to_consumers.get(pad_output, [])
        if len(consumers) != 1 or consumers[0].op_type != "Conv":
            continue

        conv_node = consumers[0]

        # Check mode
        mode = "constant"
        for attr in node.attribute:
            if attr.name == "mode":
                mode = attr.s.decode() if attr.s else "constant"
        if mode != "constant":
            continue

        # Get pads value
        if len(node.input) < 2 or not node.input[1]:
            continue
        pads_val = get_value(node.input[1])
        if pads_val is None:
            print(f"  Still can't resolve pads for {node.name}, skipping")
            continue

        pads = pads_val.flatten().astype(int).tolist()
        ndim = len(pads) // 2
        begin_pads = pads[:ndim]
        end_pads = pads[ndim:]

        # Skip if padding on batch/channel dims
        if any(p != 0 for p in begin_pads[:2]) or any(p != 0 for p in end_pads[:2]):
            continue

        spatial_begin = begin_pads[2:]
        spatial_end = end_pads[2:]

        # Get Conv's current pads
        conv_pads = None
        for attr in conv_node.attribute:
            if attr.name == "pads":
                conv_pads = list(attr.ints)
        n_spatial = len(spatial_begin)
        if conv_pads is None:
            conv_pads = [0] * (2 * n_spatial)

        # Merge
        new_pads = []
        for i in range(n_spatial):
            new_pads.append(int(spatial_begin[i]) + int(conv_pads[i]))
        for i in range(n_spatial):
            new_pads.append(int(spatial_end[i]) + int(conv_pads[n_spatial + i]))

        # Update Conv
        found = False
        for attr in conv_node.attribute:
            if attr.name == "pads":
                del attr.ints[:]
                attr.ints.extend(new_pads)
                found = True
                break
        if not found:
            conv_node.attribute.append(onnx.helper.make_attribute("pads", new_pads))

        # Rewire
        pad_input = node.input[0]
        for i, inp in enumerate(conv_node.input):
            if inp == pad_output:
                conv_node.input[i] = pad_input

        nodes_to_remove.add(node.name or id(node))
        fused += 1
        if fused <= 3:
            print(f"  Fused: {node.name} pads={pads} -> Conv new_pads={new_pads}")

    if fused > 3:
        print(f"  ... and {fused - 3} more")

    # Remove fused Pad nodes
    remaining = [n for n in model.graph.node if (n.name or id(n)) not in nodes_to_remove]
    del model.graph.node[:]
    model.graph.node.extend(remaining)

    return fused


def main():
    parser = argparse.ArgumentParser(description="Optimize encoder model for NPU")
    parser.add_argument("--models-dir", default="./models")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    static_encoder = models_dir / "encoder-model.fp32.static.onnx"

    if not static_encoder.exists():
        print(f"ERROR: Static encoder not found: {static_encoder}")
        print("Run: python -m inference.tools.convert_static --precision fp32")
        sys.exit(1)

    print("NPU Model Optimizer")
    print("====================")
    print()

    # Step 1: Use ORT to fold constants
    optimized_path = str(models_dir / "encoder-model.fp32.static.opt.onnx")
    print("[1/3] Constant folding with ONNX Runtime...")
    fold_constants_with_ort(str(static_encoder), optimized_path)

    # Step 2: Load optimized model and fuse Pad+Conv
    print()
    print("[2/3] Fusing Pad -> Conv patterns...")

    # Check if optimized model uses external data
    has_ext = os.path.exists(optimized_path + ".data")
    model = onnx.load(optimized_path, load_external_data=True)

    ops_before = count_ops(model)
    print(f"  Pad ops before fusion: {ops_before.get('Pad', 0)}")

    fused = fuse_pad_conv_in_model(model)
    print(f"  Fused: {fused} Pad->Conv pairs")

    ops_after = count_ops(model)
    print(f"  Pad ops after fusion: {ops_after.get('Pad', 0)}")

    # Step 3: Save
    print()
    print("[3/3] Saving optimized model...")
    final_path = str(models_dir / "encoder-model.fp32.static.npu.onnx")

    # Check if we need external data
    total_size = sum(
        len(t.raw_data) for t in model.graph.initializer if t.raw_data
    )
    if total_size > 1_500_000_000 or has_ext:
        data_filename = Path(final_path).name + ".data"
        print(f"  Saving with external data: {data_filename}")
        onnx.save(
            model, final_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=data_filename,
            size_threshold=0,
        )
        data_full = Path(final_path).parent / data_filename
        data_mb = data_full.stat().st_size / (1024 * 1024) if data_full.exists() else 0
        onnx_mb = os.path.getsize(final_path) / (1024 * 1024)
        print(f"  Done (ONNX: {onnx_mb:.1f} MB + data: {data_mb:.1f} MB)")
    else:
        onnx.save(model, final_path)
        size_mb = os.path.getsize(final_path) / (1024 * 1024)
        print(f"  Done ({size_mb:.1f} MB)")

    # Update static_config to point to the new file
    import json
    config_path = models_dir / "static_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        config["encoder_model"] = Path(final_path).name
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\n  Updated {config_path} -> encoder: {Path(final_path).name}")

    print()
    print("Done! Test with:")
    print("  ASR_LLM_inference/run_pipeline.sh")


if __name__ == "__main__":
    main()
