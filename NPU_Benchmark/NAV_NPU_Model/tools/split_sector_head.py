"""Cut the sector_risk Gemm out of nav_perception.onnx (the Vitis AI compiler fails on its [5120,7] shape).

Writes models/nav_perception_npu.onnx, which outputs `sector_feat` [1,5120] instead of `sector_risk`, and
models/sector_head.npz with the Gemm weights; NavPerception applies it on the CPU.
"""
import sys
import numpy as np
import onnx
from onnx import numpy_helper as nh

src, dst, head = sys.argv[1:4]
m = onnx.load(src)
g = m.graph
gemm = next(n for n in g.node if "sector_risk" in n.output)
assert gemm.op_type == "Gemm" and not any(a.name in ("transA", "transB") and a.i for a in gemm.attribute)
init = {i.name: i for i in g.initializer}
w, b = nh.to_array(init[gemm.input[1]]), nh.to_array(init[gemm.input[2]])
feat = gemm.input[0]

out = next(o for o in g.output if o.name == "sector_risk")
g.node.remove(gemm)
for name in (gemm.input[1], gemm.input[2]):
    g.initializer.remove(init[name])
g.output.remove(out)
vi = onnx.helper.make_tensor_value_info("sector_feat", onnx.TensorProto.FLOAT, [1, w.shape[0]])
g.output.append(vi)
for n in g.node:                      # rename the Reshape output to the new graph output
    n.output[:] = ["sector_feat" if o == feat else o for o in n.output]
    n.input[:] = ["sector_feat" if i == feat else i for i in n.input]
onnx.checker.check_model(m)
onnx.save(m, dst)
np.savez(head, w=w, b=b)
print("wrote", dst, head, "head", w.shape)
