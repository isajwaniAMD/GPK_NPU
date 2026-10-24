"""
PyTorch twin of nav_perception.onnx.

Same graph, same maths, and the weights are loaded FROM the ONNX file, so the NPU run and the
iGPU run use identical weights. The ONNX build in build_nav_model.py is the source of truth;
if you change the architecture there, change it here too (load_onnx() checks every tensor shape
and will refuse to load a mismatched file).

Input / output names and shapes are the same as the ONNX model:
  forward(images[1,3,2,128,160], valid_mask[1,1,128,160], pose[1,4,4], sdf_volume[1,1,32,32,32],
          query_pts[1,256,3]) -> dict(depth, conf, anchor_conf, anchor_idx, anchor_depth,
                                      sector_risk, scene_logits, sdf, sdf_grad, collision_cost)

Precision: the perception network runs in `dtype` (fp32 / fp16 / bf16). The pose transform and the
SDF lookup always run in fp32, as does the top-k, so the planner-facing geometry is not degraded by
the reduced precision. (On the NPU the same part is expected to run on the CPU.)

Two switches change the kernel mix, to match what the GPU stack would do:
  attn="manual"  MatMul + masked_fill + softmax, identical to the ONNX graph (default)
  attn="sdpa"    torch scaled_dot_product_attention (fused attention kernel, like the 7 fused MHA kernels)
  sdf="gather"   trilinear lookup by index gathers, identical to the ONNX graph (default)
  sdf="grid"     torch grid_sample (one fused trilinear kernel)
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

H, W = 128, 160
HO, WO = H // 2, W // 2
NT = (H // 16) * (W // 16)
D, HEADS, DH, DFF, NBLK = 128, 4, 32, 256, 7
S, EXTENT = 32, 4.0
NQ, KTOP = 256, 64
NEG = -1.0e4


class Linear(nn.Module):
    """y = x @ W + b with W stored [in, out] exactly as in the ONNX file."""

    def __init__(self, din, dout):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(din, dout))
        self.b = nn.Parameter(torch.zeros(dout))

    def forward(self, x):
        return torch.matmul(x, self.w.to(x.dtype)) + self.b.to(x.dtype)


class LN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.b = nn.Parameter(torch.zeros(d))
        self.d = d

    def forward(self, x):
        return F.layer_norm(x, (self.d,), self.g.to(x.dtype), self.b.to(x.dtype), 1e-5)


class Block(nn.Module):
    def __init__(self, act, attn):
        super().__init__()
        self.ln1, self.q, self.k, self.v, self.o = LN(D), Linear(D, D), Linear(D, D), Linear(D, D), Linear(D, D)
        self.ln2, self.f1, self.f2 = LN(D), Linear(D, DFF), Linear(DFF, D)
        self.act, self.attn = act, attn

    def forward(self, x, mask):                 # mask: [1,1,1,NT] bool, True = masked
        z = self.ln1(x)
        sp = lambda t: t.reshape(1, NT, HEADS, DH).transpose(1, 2)        # [1,4,80,32]
        q, k, v = sp(self.q(z)), sp(self.k(z)), sp(self.v(z))
        if self.attn == "sdpa":
            ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=~mask)
        else:
            sc = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(DH))
            sc = sc.masked_fill(mask, NEG)
            ctx = torch.matmul(torch.softmax(sc, dim=-1), v)
        x = x + self.o(ctx.transpose(1, 2).reshape(1, NT, D))
        z = self.f1(self.ln2(x))
        z = F.gelu(z) if self.act == "gelu" else F.relu(z)
        return x + self.f2(z)


class NavPerceptionTorch(nn.Module):
    def __init__(self, attn="manual", sdf="gather", dtype=torch.float32):
        super().__init__()
        self.attn_mode, self.sdf_mode, self.dtype = attn, sdf, dtype
        c3 = lambda i, o: nn.Conv3d(i, o, (1, 3, 3), padding=(0, 1, 1))
        c2 = lambda i, o, k, s=1: nn.Conv2d(i, o, k, s, k // 2)
        bn = lambda c: nn.BatchNorm2d(c, eps=1e-5)
        ct = lambda i, o, k=4, s=2, p=1: nn.ConvTranspose2d(i, o, k, s, p)
        # creation order == ONNX initializer order (checked in load_onnx)
        self.s1, self.s2 = c3(3, 32), c3(32, 32)
        self.e1, self.b1 = c2(64, 64, 3), bn(64)
        self.e2, self.b2 = c2(64, 96, 3), bn(96)
        self.e3 = c2(96, 96, 5)
        self.e4 = c2(96, 128, 1)
        self.e5, self.b5 = c2(128, 128, 3, 2), bn(128)
        self.pos = nn.Parameter(torch.zeros(1, NT, D))
        self.blocks = nn.ModuleList([Block("gelu" if i < 3 else "relu", attn) for i in range(NBLK)])
        self.lnf = LN(D)
        self.u1 = ct(128, 96)
        self.d1, self.db1 = c2(192, 96, 3), bn(96)
        self.u2 = ct(96, 64)
        self.d2, self.db2 = c2(128, 64, 3), bn(64)
        self.u3 = ct(64, 32)
        self.d3 = c2(96, 32, 3)
        self.u4 = ct(32, 32, 1, 1, 0)
        self.h_depth, self.h_conf = c2(32, 1, 3), c2(32, 1, 1)
        self.risk = Linear(HO * WO, 7)
        self.g1, self.g2 = Linear(D, 37), Linear(37, 7)
        self.e_vox = 2.0 / (S - 1)      # one voxel in normalised units; offsets are built in fp32 in _sdf

    # ---- weights from ONNX ----------------------------------------------------------
    def load_onnx(self, path):
        import onnx
        from onnx import numpy_helper as nh
        m = onnx.load(path)
        inits = {i.name: nh.to_array(i) for i in m.graph.initializer}
        isinit = lambda n: n in inits
        groups, pend_g = [], None
        nodes = list(m.graph.node)
        for k, n in enumerate(nodes):
            t = n.op_type
            if t in ("Conv", "ConvTranspose"):
                groups.append(("conv", inits[n.input[1]], inits[n.input[2]]))
            elif t == "BatchNormalization":
                groups.append(("bn", *[inits[x] for x in n.input[1:5]]))
            elif t == "Gemm":
                groups.append(("lin", inits[n.input[1]], inits[n.input[2]]))
            elif t == "MatMul" and isinit(n.input[1]) and inits[n.input[1]].ndim == 2:
                nxt = nodes[k + 1]
                assert nxt.op_type == "Add" and any(isinit(x) for x in nxt.input), "expected bias Add after MatMul"
                groups.append(("lin", inits[n.input[1]], inits[[x for x in nxt.input if isinit(x)][0]]))
            elif t == "LayerNormalization":
                groups.append(("ln", inits[n.input[1]], inits[n.input[2]]))
            elif t == "Mul" and any(x.startswith("g_") for x in n.input):
                pend_g = inits[[x for x in n.input if x.startswith("g_")][0]]
            elif t == "Add" and pend_g is not None and any(x.startswith("b_") for x in n.input):
                groups.append(("ln", pend_g, inits[[x for x in n.input if x.startswith("b_")][0]]))
                pend_g = None
            elif t == "Add" and any(x.startswith("pos_") for x in n.input):
                groups.append(("pos", inits[[x for x in n.input if x.startswith("pos_")][0]]))
        order = []
        for mod in self.modules():
            if mod is self:
                continue
            if isinstance(mod, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d)):
                order.append(("conv", mod))
            elif isinstance(mod, nn.BatchNorm2d):
                order.append(("bn", mod))
            elif isinstance(mod, Linear):
                order.append(("lin", mod))
            elif isinstance(mod, LN):
                order.append(("ln", mod))
        # self.pos sits between the encoder and the blocks in ONNX order
        idx = next(i for i, (_, mod) in enumerate(order) if mod is self.blocks[0].ln1)
        order.insert(idx, ("pos", self))
        # modules() visits attributes in assignment order; ONNX visits Q,K,V after ln1 and o after attention:
        # for Block that is ln1,q,k,v,o,ln2,f1,f2, the same as the assignment order above.
        assert len(order) == len(groups), f"parameter groups: torch {len(order)} vs onnx {len(groups)}"
        with torch.no_grad():
            for (kind, mod), grp in zip(order, groups):
                assert kind == grp[0], f"order mismatch: torch {kind} vs onnx {grp[0]}"
                T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()
                if kind == "conv":
                    w = T(grp[1]).reshape(mod.weight.shape)
                    mod.weight.copy_(w); mod.bias.copy_(T(grp[2]))
                elif kind == "bn":
                    mod.weight.copy_(T(grp[1])); mod.bias.copy_(T(grp[2]))
                    mod.running_mean.copy_(T(grp[3])); mod.running_var.copy_(T(grp[4]))
                elif kind == "lin":
                    assert tuple(grp[1].shape) == tuple(mod.w.shape), f"linear shape {grp[1].shape} vs {tuple(mod.w.shape)}"
                    mod.w.copy_(T(grp[1])); mod.b.copy_(T(grp[2]))
                elif kind == "ln":
                    mod.g.copy_(T(grp[1])); mod.b.copy_(T(grp[2]))
                else:
                    self.pos.copy_(T(grp[1]))
        return self

    # ---- standalone weights (no onnx package needed) -----------------------------------
    @classmethod
    def from_weights(cls, path="nav_perception.pt", attn="manual", sdf="gather"):
        m = cls(attn, sdf)
        m.load_state_dict(torch.load(path, map_location="cpu"))
        return m.eval()

    def save_weights(self, path="models/nav_perception.pt"):
        torch.save({k: v.detach().cpu().float() for k, v in self.state_dict().items()}, path)

    @staticmethod
    def example_inputs(device="cpu", dtype=torch.float32):
        """Random inputs with the right shapes (replace with real OKVIS2 data)."""
        g = torch.Generator().manual_seed(0)
        images = torch.rand(1, 3, 2, H, W, generator=g) * 255
        valid = torch.ones(1, 1, H, W)
        pose = torch.eye(4).unsqueeze(0)
        vol = torch.rand(1, 1, S, S, S, generator=g) * 4 - 1
        pts = torch.rand(1, NQ, 3, generator=g) * 2 - 1
        return [t.to(device) for t in (images, valid, pose, vol, pts)]

    # ---- forward --------------------------------------------------------------------
    def _perception(self, images, valid_mask):
        dt = self.dtype
        x = ((images * (1 / 255.0) - 0.45) / 0.225).to(dt)
        x = F.relu(self.s1(x))
        x = F.relu(self.s2(x))
        try:
            x = F.avg_pool3d(x, (1, 2, 2), (1, 2, 2))             # 3-D pooling, as in the ONNX graph
            x = x.reshape(1, 64, HO, WO)
        except (NotImplementedError, RuntimeError):               # some CPU builds lack 3-D pooling in bf16/fp16
            x = F.avg_pool2d(x.reshape(1, 64, H, W), 2, 2)        # identical result
        skip0 = x
        x = F.relu(self.b1(self.e1(x)))
        x = F.max_pool2d(x, 2, 2)
        skip1 = x
        x = F.relu(self.b2(self.e2(x)))
        x = F.relu(self.e3(x))
        x = F.avg_pool2d(x, 3, 2, 1, count_include_pad=False)     # ONNX AveragePool excludes padding
        skip2 = x
        x = F.relu(self.e4(x))
        x = F.relu(self.b5(self.e5(x)))
        tokv = F.avg_pool2d(valid_mask, 16, 16).reshape(1, NT)
        mask = (tokv < 0.5).reshape(1, 1, 1, NT)
        t = x.reshape(1, D, NT).transpose(1, 2) + self.pos.to(dt)
        for blk in self.blocks:
            t = blk(t, mask)
        t = self.lnf(t)
        fmap = t.transpose(1, 2).reshape(1, D, H // 16, W // 16)
        d = F.relu(self.u1(fmap))
        d = F.relu(self.db1(self.d1(torch.cat([d, skip2], 1))))
        d = F.relu(self.u2(d))
        d = F.relu(self.db2(self.d2(torch.cat([d, skip1], 1))))
        d = F.relu(self.u3(d))
        d = F.relu(self.d3(torch.cat([d, skip0], 1)))
        d = F.relu(self.u4(d))
        depth = torch.clamp(torch.abs(self.h_depth(d)) * 10.0, 0.1, 80.0)
        conf = torch.sigmoid(self.h_conf(d))
        cflat = conf.reshape(1, HO * WO)
        a_conf, a_idx = torch.topk(cflat.float(), KTOP, dim=-1)
        a_depth = depth.reshape(1, HO * WO).float().gather(1, a_idx)
        risk = self.risk(cflat).float()
        gp = fmap.mean(dim=(2, 3))
        scene = self.g2(F.relu(self.g1(gp))).float()
        return depth.float(), conf.float(), a_conf, a_idx, a_depth, risk, scene

    def _sdf(self, pose, sdf_volume, query_pts):
        R, tr = pose[:, :3, :3], pose[:, :3, 3].unsqueeze(1)
        pv = torch.matmul(query_pts, R.transpose(1, 2)) + tr
        pn = pv * (1.0 / EXTENT)
        e = self.e_vox
        offs = torch.tensor([[0, 0, 0], [e, 0, 0], [-e, 0, 0], [0, e, 0], [0, -e, 0], [0, 0, e], [0, 0, -e]],
                            dtype=torch.float32, device=pn.device).reshape(1, 7, 1, 3)
        q = (pn.unsqueeze(1) + offs).reshape(7 * NQ, 3)
        if self.sdf_mode == "grid":
            g = q.reshape(1, 1, 1, 7 * NQ, 3)
            v = F.grid_sample(sdf_volume, g, mode="bilinear", padding_mode="border", align_corners=True)
            val = v.reshape(7, NQ)
        else:
            vf = sdf_volume.reshape(S ** 3)
            u = ((q + 1.0) * ((S - 1) / 2.0)).clamp(0.0, S - 1 - 1e-3)
            fl = torch.floor(u)
            fr = u - fl
            i0 = fl.long()
            base = i0[:, 2] * (S * S) + i0[:, 1] * S + i0[:, 0]
            acc = 0
            for dz in (0, 1):
                for dy in (0, 1):
                    for dx in (0, 1):
                        w = ((1 - fr[:, 0]) if dx == 0 else fr[:, 0]) * ((1 - fr[:, 1]) if dy == 0 else fr[:, 1]) \
                            * ((1 - fr[:, 2]) if dz == 0 else fr[:, 2])
                        acc = acc + w * vf[base + (dz * S * S + dy * S + dx)]
            val = acc.reshape(7, NQ)
        sc = 1.0 / (2 * self.e_vox * EXTENT)
        grad = torch.stack([(val[1] - val[2]) * sc, (val[3] - val[4]) * sc, (val[5] - val[6]) * sc], dim=1)
        sdf = val[0]
        cost = torch.clamp(0.4 - sdf, min=0.0).sum().reshape(1, 1)
        return sdf.reshape(1, NQ), grad.reshape(1, NQ, 3), cost

    @torch.no_grad()
    def forward(self, images, valid_mask, pose, sdf_volume, query_pts):
        depth, conf, a_conf, a_idx, a_depth, risk, scene = self._perception(images, valid_mask)
        sdf, grad, cost = self._sdf(pose.float(), sdf_volume.float(), query_pts.float())
        return {"depth": depth, "conf": conf, "anchor_conf": a_conf, "anchor_idx": a_idx,
                "anchor_depth": a_depth, "sector_risk": risk, "scene_logits": scene,
                "sdf": sdf, "sdf_grad": grad, "collision_cost": cost}

    def to_precision(self, dtype):
        """Cast the perception weights; keep the BatchNorm statistics in fp32-safe form."""
        self.dtype = dtype
        self.to(dtype)
        return self


if __name__ == "__main__":
    # Same file runs on Strix Halo (ROCm build of PyTorch) and Jetson Thor (CUDA build): both use torch.cuda.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = NavPerceptionTorch.from_weights("models/nav_perception.pt").to(dev)      # fp32
    # model.to_precision(torch.float16).to(dev)                              # or torch.bfloat16
    out = model(*NavPerceptionTorch.example_inputs(dev))
    print("device:", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "")
    for k, v in out.items():
        print(f"{k:15s} {tuple(v.shape)} {v.dtype}")
