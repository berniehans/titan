#!/usr/bin/env python
"""Build the Phase 6.2 CPU-reference-bank synthetic GGUF model + its FP32 reference.

Deterministic, stdlib + numpy only. Produces two committed artifacts under
``tests/fixtures/synthetic/``:

  * ``synthetic_min.gguf``  - a minimal REAL v3 GGUF with the llama.cpp/GGUF
    tensor naming convention and the same quant formats the real Qwen3 fixture
    carries (Q4_K attentions + Q8_0 mid-FFN + F32 embed/head), over a tiny
    2-layer, hidden=256, vocab=16 model. The forward bank consumes ``blk.0.*``.
  * ``reference.json``      - the CPU authority's expected output for a single
    forward step (one token at position 0): the layer-0 activation vector and
    the 16 logits, computed here INDEPENDENTLY (this file, numpy float32)
    following ggml.c semantics. The Rust ``forward_cpu`` bank must reproduce
    these bit-exactly (see the Rust test comments for the controlled-constant
    design that makes that exact).

Design of the controlled known-constants model (why logits are bit-exact):

  * Single token at position 0  -> RoPE theta=0, cos=1/sin=0 exactly (identity).
  * Single-token attention       -> softmax over one element = 1.0 exactly; the
    only exp() call is exp(0.0)=1.0, exact in every IEEE libm.
  * The feed-forward DOWN weight is exactly zero, so the FFN branch (silu on
    real nonzero gate values) contributes exactly 0 to the residual. silu itself
    is validated against hand-computed constants in the Rust unit tests; here it
    is present but gated out of the exact logits path so no libm (expf/cosf/
    powf - which are NOT IEEE-correctly-rounded and differ between runtimes)
    reaches the compared output.
  * Everything else (RMSNorm with f64 accumulation + correctly-rounded sqrt/
    reciprocal, fp32 matmul via dequant->dot, residual adds) is determined by
    IEEE-correctly-rounded fp32/f64 arithmetic in a fixed order, so numpy-fp32
    and Rust-fp32 agree bit-for-bit when run in the same order.

Regenerate (deterministic):
    uv run python tools/build_synthetic_gguf.py
"""

import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SYN = ROOT / "tests" / "fixtures" / "synthetic"
OUT_GGUF = SYN / "synthetic_min.gguf"
OUT_REF = SYN / "reference.json"

# --------------------------------------------------------------------------- #
# Model hyperparameters (must match engine-core ModelConfig keys)
# --------------------------------------------------------------------------- #
H = 256          # hidden_size
N_HEAD = 2       # query heads
HEAD_DIM = 128   # head dim (key_length / value_length)
N_KV = 1         # GQA kv heads
FFN = 256        # feed-forward / intermediate
VOCAB = 16
N_LAYER = 2      # tensor fixture has 2 layers; forward bank uses layer 0
CTX = 64
EPS = 1e-5
FREQ_BASE = 1000000.0

TOKEN_ID = 3     # token we run the reference for
POS = 0

T_F32, T_F16, T_Q8_0, T_Q4_K = 0, 1, 8, 12

# --------------------------------------------------------------------------- #
# fp16 / block helpers (GGUF LE, matching ggml-quants / engine-core dequant)
# --------------------------------------------------------------------------- #
def f16_bytes(x: float) -> bytes:
    return struct.pack("<H", np.float16(x).view(np.uint16).item())


def f32_bytes(vals) -> bytes:
    return np.asarray(vals, dtype=np.float32).astype("<f4").tobytes()


def dequant_q4k_block(blk: bytes) -> np.ndarray:
    """Matches engine-core dequant::dequant_q4k_cpu exactly (fp32)."""
    d = np.float32(np.frombuffer(blk[0:2], "<f2")[0])
    dmin = np.float32(np.frombuffer(blk[2:4], "<f2")[0])
    scales = blk[4:16]
    qs = blk[16:144]
    out = []
    is_idx = 0
    for jgrp in range(0, 256, 64):
        for sub in (0, 1):
            j = is_idx + sub
            if j < 4:
                sc = np.float32(scales[j] & 63)
                mn = np.float32(scales[j + 4] & 63)
            else:
                sc = np.float32((scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4))
                mn = np.float32((scales[j + 4] >> 4) | ((scales[j] >> 6) << 4))
            d1 = np.float32(d * sc)
            m1 = np.float32(dmin * mn)
            qbase = (jgrp // 64) * 32
            lo = sub == 0
            for l in range(32):
                nib = (qs[qbase + l] & 0xF) if lo else (qs[qbase + l] >> 4)
                out.append(np.float32(d1 * np.float32(nib) - m1))
        is_idx += 2
    return np.asarray(out, dtype=np.float32)


def q4k_block(d: float, dmin: float, scales, qs) -> bytes:
    """Encode one 256-elem Q4_K block (144 bytes) from byte-level fields."""
    hdr = f16_bytes(d) + f16_bytes(dmin)
    sc = bytes(scales)
    qb = bytes(qs)
    assert len(sc) == 12 and len(qb) == 128
    return hdr + sc + qb


def dequant_q8_0_block(blk: bytes) -> np.ndarray:
    d = np.float32(np.frombuffer(blk[0:2], "<f2")[0])
    qs = np.frombuffer(blk[2:], dtype=np.int8)
    return (qs.astype(np.float32) * d).astype(np.float32)


def q8_0_block(d: float, qs) -> bytes:
    return f16_bytes(d) + bytes(int(x) & 0xFF for x in qs)


# --------------------------------------------------------------------------- #
# Deterministic weight generation (values irrelevant to exactness; keep simple)
# --------------------------------------------------------------------------- #
def gen_q4k(ne0: int, ne1: int, zero: bool = False) -> bytes:
    """Q4_K tensor, dims[0]=ne0 (multiple of 256), dims[1]=ne1 columns."""
    OUT_REF = None
    nblk = ne0 // 256
    out = b""
    seed = 0
    for col in range(ne1):
        for b in range(nblk):
            if zero:
                out += q4k_block(1.0, 0.0, [0] * 12, [0] * 128)
                continue
            scales = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]
            qs = []
            for i in range(128):
                v = (seed + col * 17 + b * 3 + i * 7) % 256
                qs.append(v)
                seed += 1
            out += q4k_block(1.0, 0.0, scales, qs)
    return out


def gen_q8_0(ne0: int, ne1: int, zero: bool = False) -> bytes:
    nblk = ne0 // 32
    out = b""
    for col in range(ne1):
        for b in range(nblk):
            if zero:
                out += q8_0_block(1.0, [0] * 32)
                continue
            qs = [((col * 3 + b * 5 + j) % 64) - 32 for j in range(32)]
            out += q8_0_block(1.0, qs)
    return out


def gen_f32(rows: int, cols: int, pattern) -> bytes:
    return f32_bytes([[pattern(r, c) for c in range(cols)] for r in range(rows)])


def gen_f32_1d(n: int, fn) -> bytes:
    return f32_bytes([fn(i) for i in range(n)])


# --------------------------------------------------------------------------- #
# Tensor descriptors (name, dims, type, bytes)
# --------------------------------------------------------------------------- #
def layer_tensors(layer: int):
    p = f"blk.{layer}."
    return [
        (p + "attn_norm.weight", [H], T_F32, gen_f32_1d(H, lambda i: 0.5 if i % 3 == 0 else 1.0)),
        (p + "attn_q.weight", [H, N_HEAD * HEAD_DIM], T_Q4_K, gen_q4k(H, N_HEAD * HEAD_DIM)),
        (p + "attn_q_norm.weight", [HEAD_DIM], T_F32, gen_f32_1d(HEAD_DIM, lambda i: 1.0)),
        (p + "attn_k.weight", [H, N_KV * HEAD_DIM], T_Q4_K, gen_q4k(H, N_KV * HEAD_DIM)),
        (p + "attn_k_norm.weight", [HEAD_DIM], T_F32, gen_f32_1d(HEAD_DIM, lambda i: 1.0)),
        (p + "attn_v.weight", [H, N_KV * HEAD_DIM], T_Q8_0, gen_q8_0(H, N_KV * HEAD_DIM)),
        (p + "attn_output.weight", [H, N_HEAD * HEAD_DIM], T_Q4_K, gen_q4k(H, N_HEAD * HEAD_DIM)),
        (p + "ffn_norm.weight", [H], T_F32, gen_f32_1d(H, lambda i: 0.25 + 0.5 * (i % 2))),
        (p + "ffn_gate.weight", [H, FFN], T_Q8_0, gen_q8_0(H, FFN)),
        (p + "ffn_up.weight", [H, FFN], T_Q8_0, gen_q8_0(H, FFN)),
        (p + "ffn_down.weight", [FFN, H], T_Q4_K, gen_q4k(FFN, H, zero=True)),
    ]


def embedding_bytes() -> bytes:
    # token_embd [H, VOCAB] F32. Pattern: small integers, deterministic.
    return gen_f32(H, VOCAB, lambda r, c: ((r * 7 + c * 13) % 9) - 4)


def norm_ones(n: int) -> bytes:
    return gen_f32_1d(n, lambda i: 1.0)


TENSOR_DEFS = [
    ("output_norm.weight", [H], T_F32, norm_ones(H)),
    ("token_embd.weight", [H, VOCAB], T_F32, embedding_bytes()),
] + layer_tensors(0) + layer_tensors(1)

# --------------------------------------------------------------------------- #
# GGUF v3 writer (mirrors engine-io GgufReader expectations)
# --------------------------------------------------------------------------- #
BLOCK_ELEMS = {T_Q4_K: 256, T_Q8_0: 32}
TYPE_SIZE = {T_F32: 4, T_F16: 2, T_Q8_0: 34, T_Q4_K: 144}


def tensor_size(dims, t):
    be = BLOCK_ELEMS.get(t, 1)
    tot = (dims[0] // be) * TYPE_SIZE[t]
    for d in dims[1:]:
        tot *= d
    return tot


def kv_str(v):
    b = v.encode()
    return struct.pack("<Q", len(b)) + b


def build_gguf(tensor_defs) -> bytes:
    for name, dims, t, data in tensor_defs:
        assert len(data) == tensor_size(dims, t), (name, len(data), tensor_size(dims, t))

    metadata = [
        ("general.architecture", 8, kv_str("qwen3")),
        ("general.type", 8, kv_str("model")),
        ("qwen3.block_count", 4, struct.pack("<I", N_LAYER)),
        ("qwen3.embedding_length", 4, struct.pack("<I", H)),
        ("qwen3.feed_forward_length", 4, struct.pack("<I", FFN)),
        ("qwen3.attention.head_count", 4, struct.pack("<I", N_HEAD)),
        ("qwen3.attention.head_count_kv", 4, struct.pack("<I", N_KV)),
        ("qwen3.attention.key_length", 4, struct.pack("<I", HEAD_DIM)),
        ("qwen3.attention.value_length", 4, struct.pack("<I", HEAD_DIM)),
        ("qwen3.context_length", 4, struct.pack("<I", CTX)),
        ("qwen3.rope.freq_base", 6, struct.pack("<f", FREQ_BASE)),
        ("qwen3.attention.layer_norm_rms_epsilon", 6, struct.pack("<f", EPS)),
        ("tokenizer.ggml.model", 8, kv_str("gpt2")),
        ("tokenizer.ggml.tokens", 9,
         struct.pack("<I", 8) + struct.pack("<Q", VOCAB) +
         b"".join(kv_str(f"tok{i}") for i in range(VOCAB))),
        ("tokenizer.ggml.eos_token_id", 4, struct.pack("<I", 0)),
        ("tokenizer.ggml.add_bos_token", 7, struct.pack("<B", 0)),
        ("general.alignment", 4, struct.pack("<I", 32)),
    ]

    n_tensors = len(tensor_defs)
    n_kv = len(metadata)
    offsets = []
    o = 0
    for _, dims, t, _ in tensor_defs:
        offsets.append(o)
        o += tensor_size(dims, t)
    total_data = o

    buf = bytearray()
    buf += b"GGUF"
    buf += struct.pack("<I", 3)
    buf += struct.pack("<Q", n_tensors)
    buf += struct.pack("<Q", n_kv)
    for k, vt, payload in metadata:
        buf += kv_str(k) + struct.pack("<I", vt) + payload
    for (name, dims, t, _), off in zip(tensor_defs, offsets):
        buf += kv_str(name)
        buf += struct.pack("<I", len(dims))
        buf += b"".join(struct.pack("<Q", d) for d in dims)
        buf += struct.pack("<I", t)
        buf += struct.pack("<Q", off)
    cur = len(buf)
    aligned = (cur + 31) & ~31
    buf += b"\x00" * (aligned - cur)
    assert len(buf) == aligned
    for (_, _, _, data) in tensor_defs:
        buf += data
    assert len(buf) == aligned + total_data
    return bytes(buf)


# --------------------------------------------------------------------------- #
# Independent FP32 reference forward (ggml.c semantics, numpy float32)
# --------------------------------------------------------------------------- #
def dequant_tensor(t, dims, data):
    ne1 = dims[1] if len(dims) > 1 else 1
    if t == T_F32:
        arr = np.frombuffer(data, dtype="<f4").astype(np.float32)
        if len(dims) > 1:
            arr = arr.reshape(dims[1], dims[0])
        return arr
    if t == T_Q4_K:
        blk = dims[0] // 256
        step = 144 * blk
        cols = []
        for c in range(ne1):
            seg = data[c * step:(c + 1) * step]
            vals = []
            for b in range(blk):
                vals.append(dequant_q4k_block(seg[b * 144:(b + 1) * 144]))
            cols.append(np.concatenate(vals))
        return np.stack(cols)  # [ne1, ne0]
    if t == T_Q8_0:
        blk = dims[0] // 32
        step = 34 * blk
        cols = []
        for c in range(ne1):
            seg = data[c * step:(c + 1) * step]
            vals = []
            for b in range(blk):
                vals.append(dequant_q8_0_block(seg[b * 34:(b + 1) * 34]))
            cols.append(np.concatenate(vals))
        return np.stack(cols)
    raise ValueError("bad type %d" % t)


def rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    """ggml.c: f64 sum of f32 squares, mean as f32, scale=1/sqrt(mean+eps),
    y=(x*scale)*w."""
    n = x.shape[0]
    ss = 0.0
    for i in range(n):
        xi = float(x[i])
        ss += float(np.float32(xi) * np.float32(xi))
    mean = np.float32(ss / n)
    scale = np.float32(1.0 / np.float32(np.sqrt(np.float32(mean + np.float32(eps)))))
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        out[i] = np.float32(np.float32(x[i] * scale) * w[i])
    return out


def matmul_q(weight: np.ndarray, x: np.ndarray) -> np.ndarray:
    """weight [out=ne1, in=ne0]; out[j] = sum_i w[j,i]*x[i], fp32 sequential."""
    out = np.empty(weight.shape[0], dtype=np.float32)
    for j in range(weight.shape[0]):
        acc = np.float32(0.0)
        for i in range(weight.shape[1]):
            acc = np.float32(acc + np.float32(weight[j, i] * x[i]))
        out[j] = acc
    return out


def softmax_1(x: np.ndarray) -> np.ndarray:
    m = np.float32(x.max())
    v = (x - m).astype(np.float32)
    e = np.exp(v).astype(np.float32)
    s = np.float32(e.sum(dtype=np.float32))
    return (e / s).astype(np.float32)


def reference_forward(tensors) -> dict:
    def W(name):
        t, dims, data = tensors[name]
        return dequant_tensor(t, dims, data)

    def W1(name):
        t, dims, data = tensors[name]
        return np.frombuffer(data, dtype="<f4").astype(np.float32)

    embd = W("token_embd.weight")  # [VOCAB, H]
    embed = embd[TOKEN_ID].astype(np.float32)  # [H]

    # ---- layer 0 ----
    h = rms_norm(embed, W1("blk.0.attn_norm.weight"), EPS)
    q = matmul_q(W("blk.0.attn_q.weight"), h)  # [256]
    k = matmul_q(W("blk.0.attn_k.weight"), h)  # [128]
    v = matmul_q(W("blk.0.attn_v.weight"), h)  # [128]

    wq_norm = W1("blk.0.attn_q_norm.weight")
    wk_norm = W1("blk.0.attn_k_norm.weight")
    qh = q.reshape(N_HEAD, HEAD_DIM)
    for hh in range(N_HEAD):
        qh[hh] = rms_norm(qh[hh], wq_norm, EPS)
    kh = k.reshape(N_KV, HEAD_DIM)
    for hh in range(N_KV):
        kh[hh] = rms_norm(kh[hh], wk_norm, EPS)
    # pos=0 => rota identity (structural restructure still applied)
    q_rope = qh.reshape(-1)
    k_rope = kh.reshape(-1)

    attn_heads = np.empty((N_HEAD, HEAD_DIM), dtype=np.float32)
    for hh in range(N_HEAD):
        s = np.float32(np.dot(q_rope[hh * HEAD_DIM:(hh + 1) * HEAD_DIM].astype(np.float64),
                              k_rope[0:HEAD_DIM].astype(np.float64)) / np.sqrt(HEAD_DIM))
        aw = softmax_1(np.asarray([s], dtype=np.float32))  # == 1.0
        attn_heads[hh] = (v.astype(np.float32) * np.float32(aw[0]))
    attn_concat = attn_heads.reshape(-1)  # [256]

    out_proj = matmul_q(W("blk.0.attn_output.weight"), attn_concat)
    h1 = (embed + out_proj).astype(np.float32)

    hn = rms_norm(h1, W1("blk.0.ffn_norm.weight"), EPS)
    gate = matmul_q(W("blk.0.ffn_gate.weight"), hn)
    up = matmul_q(W("blk.0.ffn_up.weight"), hn)
    silu = (gate / (np.float32(1.0) + np.exp(-gate.astype(np.float32))).astype(np.float32)).astype(np.float32)
    proj = (silu * up).astype(np.float32)
    down = matmul_q(W("blk.0.ffn_down.weight"), proj)  # all-zero weight -> exactly 0
    assert np.all(down == 0.0), "ffn down must contribute exactly 0"
    h2 = (h1 + down).astype(np.float32)  # layer-0 out

    ho = rms_norm(h2, W1("output_norm.weight"), EPS)
    logits = matmul_q(embd, ho)  # [VOCAB]

    return {
        "layer0": [float(x) for x in h2],
        "logits": [float(x) for x in logits],
    }


def parse_tensor_section(gguf, tensor_defs):
    """Recover per-tensor (type, dims, data) by re-walking the header."""
    o = 4 + 4 + 8 + 8
    nkv = struct.unpack_from("<Q", gguf, 16)[0]
    for _ in range(nkv):
        ln = struct.unpack_from("<Q", gguf, o)[0]; o += 8 + ln
        vt = struct.unpack_from("<I", gguf, o)[0]; o += 4
        if vt == 8:
            ln = struct.unpack_from("<Q", gguf, o)[0]; o += 8 + ln
        elif vt == 9:
            et = struct.unpack_from("<I", gguf, o)[0]
            cnt = struct.unpack_from("<Q", gguf, o + 4)[0]; o += 12
            for _ in range(cnt):
                if et == 8:
                    ln = struct.unpack_from("<Q", gguf, o)[0]; o += 8 + ln
        elif vt == 6 or vt == 4 or vt == 5:
            o += 4
        elif vt == 7:
            o += 1  # GGUF Bool is 1 byte
        elif vt == 0 or vt == 1:
            o += 1
        elif vt == 2 or vt == 3:
            o += 2
        elif vt == 10 or vt == 11 or vt == 12:
            o += 8
        else:
            raise ValueError("unhandled vtype %d" % vt)
    n_tensors = struct.unpack_from("<Q", gguf, 8)[0]
    names = []
    dims_l = []
    types = []
    for _ in range(n_tensors):
        ln = struct.unpack_from("<Q", gguf, o)[0]; o += 8
        name = gguf[o:o + ln].decode(); o += ln
        nd = struct.unpack_from("<I", gguf, o)[0]; o += 4
        dims = []
        for _ in range(nd):
            dims.append(struct.unpack_from("<Q", gguf, o)[0]); o += 8
        tt = struct.unpack_from("<I", gguf, o)[0]; o += 4
        off = struct.unpack_from("<Q", gguf, o)[0]; o += 8
        names.append(name); dims_l.append(dims); types.append(tt)
    aligned = (o + 31) & ~31
    pos = aligned
    out = {}
    for i, (name, dims, tt) in enumerate(zip(names, dims_l, types)):
        sz = tensor_size(dims, tt)
        out[name] = (tt, dims, gguf[pos:pos + sz])
        pos += sz
    return out


def main() -> int:
    SYN.mkdir(parents=True, exist_ok=True)
    gguf = build_gguf(TENSOR_DEFS)
    OUT_GGUF.write_bytes(gguf)
    print(f"wrote {OUT_GGUF.relative_to(ROOT)} ({len(gguf)} bytes)")

    tensors = parse_tensor_section(gguf, TENSOR_DEFS)
    ref = reference_forward(tensors)

    OUT_REF.write_text(json.dumps({
        "config": {
            "hidden": H, "n_head": N_HEAD, "head_dim": HEAD_DIM,
            "n_head_kv": N_KV, "ffn": FFN, "vocab": VOCAB,
            "layers": N_LAYER, "context": CTX, "eps": EPS,
            "freq_base": FREQ_BASE,
        },
        "token_id": TOKEN_ID, "pos": POS,
        "layer0": ref["layer0"],
        "logits": ref["logits"],
        "note": ("controlled known-constants model; single token @ pos 0; "
                 "RoPE identity; single-token softmax; FFN down weight zero "
                 "so silu is gated out of the exact path"),
    }, indent=1), encoding="utf-8")
    print(f"wrote {OUT_REF.relative_to(ROOT)} (layer0={len(ref['layer0'])}, "
          f"logits={len(ref['logits'])})")
    print("logits:", [f"{x:.6f}" for x in ref["logits"][:8]], "...")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
