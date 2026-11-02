#!/usr/bin/env python3
"""
Eliminate Pad->Reshape->Slice patterns in Conformer self-attention.

These 24 Pad ops (one per layer) implement NeMo's '_relative_shift' trick
for relative positional attention. Each does:
    Pad(input, [0,0,0,1,0,0,0,0]) -> Reshape -> Slice

The Pad inserts a single zero column on the left of the last dimension.
Since the model has static shapes, we can run shape inference to find
the exact tensor dimensions and see if the VAIML compiler handles them.

This script:
1. Runs ONNX shape inference to get all tensor dimensions
2. Analyzes the Pad->Reshape->Slice chains
3. Optionally replaces each chain with equivalent ops that avoid Pad

Usage:
    python -m inference.tools.fuse_attn_pads              # Analyze only
    python -m inference.tools.fuse_attn_pads --apply      # Apply and save
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper, TensorProto, helper


def get_shape(model, tensor_name):
    """Get static shape of a tensor from value_info or graph input/output."""
    for vi in list(model.graph.value_info) + list(model.graph.input) + list(model.graph.output):
        if vi.name == tensor_name:
            dims = []
            for d in vi.type.tensor_type.shape.dim:
                if d.dim_value > 0:
                    dims.append(d.dim_value)
                elif d.dim_param:
                    dims.append(d.dim_param)
                else:
                    dims.append(0)
            return dims
    return None


def analyze_pad_chains(model):
    """Find and analyze all Pad->Reshape->Slice chains in self_attn."""
    init_map = {}
    for init in model.graph.initializer:
        try:
            init_map[init.name] = numpy_helper.to_array(init)
        except Exception:
            init_map[init.name] = None

    # Build node maps
    output_to_node = {}
    for node in model.graph.node:
        for out in node.output:
            output_to_node[out] = node

    input_to_consumers = {}
    for node in model.graph.node:
        for inp in node.input:
            if inp not in input_to_consumers:
                input_to_consumers[inp] = []
            input_to_consumers[inp].append(node)

    chains = []
    for node in model.graph.node:
        if node.op_type != "Pad" or "self_attn" not in (node.name or ""):
            continue

        pad_node = node
        pad_input = pad_node.input[0]
        pad_output = pad_node.output[0]

        # Get pads values
        pads = None
        if len(pad_node.input) > 1:
            pads_name = pad_node.input[1]
            # Find the node or initializer that produces the pads
            if pads_name in init_map and init_map[pads_name] is not None:
                pads = init_map[pads_name]
            elif pads_name in output_to_node:
                # It might be a Cast or Concat of constants
                producer = output_to_node[pads_name]
                if producer.op_type == "Cast":
                    src = producer.input[0]
                    if src in init_map and init_map[src] is not None:
                        pads = init_map[src].astype(np.int64)

        # Find consumer Reshape
        consumers = input_to_consumers.get(pad_output, [])
        reshape_node = None
        for c in consumers:
            if c.op_type == "Reshape":
                reshape_node = c
                break

        # Find consumer of Reshape (should be Slice)
        slice_node = None
        if reshape_node:
            reshape_out = reshape_node.output[0]
            for c in input_to_consumers.get(reshape_out, []):
                if c.op_type == "Slice":
                    slice_node = c
                    break

        # Get shapes
        pad_in_shape = get_shape(model, pad_input)
        pad_out_shape = get_shape(model, pad_output)

        chain = {
            "pad_node": pad_node,
            "reshape_node": reshape_node,
            "slice_node": slice_node,
            "pads": pads,
            "pad_input_shape": pad_in_shape,
            "pad_output_shape": pad_out_shape,
        }

        # Get reshape target shape
        if reshape_node and len(reshape_node.input) > 1:
            shape_name = reshape_node.input[1]
            if shape_name in init_map and init_map[shape_name] is not None:
                chain["reshape_shape"] = init_map[shape_name]

        chains.append(chain)

    return chains


def main():
    parser = argparse.ArgumentParser(description="Analyze/fuse attention Pad ops")
    parser.add_argument("--model", default="models/encoder-model.fp32.static.npu.onnx",
                        help="Path to encoder ONNX model")
    parser.add_argument("--apply", action="store_true", help="Apply fusions and save")
    parser.add_argument("--shape-inference", action="store_true",
                        help="Run ONNX shape inference first (slow but gives shapes)")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: {model_path} not found")
        sys.exit(1)

    print(f"Loading {model_path}...")
    model = onnx.load(str(model_path), load_external_data=True)
    print(f"  Nodes: {len(model.graph.node)}")

    if args.shape_inference:
        print("Running shape inference (this may take a while)...")
        try:
            model = onnx.shape_inference.infer_shapes(model, data_prop=True)
            print("  Shape inference complete")
        except Exception as e:
            print(f"  Shape inference failed: {e}")
            print("  Continuing with available shapes...")

    # Analyze
    chains = analyze_pad_chains(model)
    print(f"\nFound {len(chains)} Pad->Reshape->Slice chains in self_attn:")
    print()

    for i, chain in enumerate(chains):
        pad = chain["pad_node"]
        name = pad.name or f"chain_{i}"
        pads = chain["pads"]

        has_reshape = chain["reshape_node"] is not None
        has_slice = chain["slice_node"] is not None
        complete = has_reshape and has_slice

        print(f"  [{i:2d}] {name}")
        print(f"       Pads: {pads}")
        print(f"       Input shape:  {chain['pad_input_shape']}")
        print(f"       Output shape: {chain['pad_output_shape']}")
        if "reshape_shape" in chain:
            print(f"       Reshape to:   {chain['reshape_shape']}")
        print(f"       Chain: Pad -> {'Reshape' if has_reshape else '?'} -> {'Slice' if has_slice else '?'}")
        print(f"       Complete: {'YES' if complete else 'NO'}")
        print()

    # Summary
    complete = sum(1 for c in chains if c["reshape_node"] and c["slice_node"])
    print(f"Summary: {complete}/{len(chains)} chains are complete Pad->Reshape->Slice")

    if all(c["pads"] is not None for c in chains):
        unique_pads = set(tuple(c["pads"].tolist()) for c in chains if c["pads"] is not None)
        print(f"Unique pad patterns: {len(unique_pads)}")
        for p in unique_pads:
            print(f"  {list(p)}")

    if not args.apply:
        print("\nRun with --apply to fuse these chains (requires --shape-inference first)")
        return

    print("\nFusion not yet implemented -- the analysis above helps determine")
    print("if these Pads are causing VAIML partition splits.")
    print("If they are, we can replace each Pad->Reshape->Slice with")
    print("a Concat([zeros, input], axis=-1) -> Reshape -> Slice")
    print("which avoids the Pad op entirely.")


if __name__ == "__main__":
    main()
