"""Minimal GGUF metadata parser (mirrors titan engine-io GgufReader semantics).
Dumps scalar config values + tokenizer arrays for golden fixtures.
Format per GGUF spec v3: magic(4) version(u32) tensor_count(u64) kv_count(u64)
then KV pairs, then tensor infos.
"""
import json, struct

PATH = "C:/Users/niber/AppData/Local/hermes/workspace/titan/testdata/Qwen3-0.6B-Q4_K_M.gguf"

def u16(b, o): return struct.unpack_from("<H", b, o)[0]
def i16(b, o): return struct.unpack_from("<h", b, o)[0]
def u32(b, o): return struct.unpack_from("<I", b, o)[0]
def i32(b, o): return struct.unpack_from("<i", b, o)[0]
def u64(b, o): return struct.unpack_from("<Q", b, o)[0]
def f32(b, o): return struct.unpack_from("<f", b, o)[0]
def f64(b, o): return struct.unpack_from("<d", b, o)[0]

def read_str(b, o):
    ln = u64(b, o); o += 8
    return b[o:o+ln].decode("utf-8"), o + ln

def parse_value(b, o, vtype):
    if vtype == 0: return b[o], o+1                      # U8
    if vtype == 1: return struct.unpack_from("<b", b, o)[0], o+1  # I8
    if vtype == 2: return u16(b, o), o+2
    if vtype == 3: return i16(b, o), o+2
    if vtype == 4: return u32(b, o), o+4
    if vtype == 5: return i32(b, o), o+4
    if vtype == 6: return f32(b, o), o+4
    if vtype == 7: return bool(b[o]), o+1                # Bool
    if vtype == 8:                                        # String
        return read_str(b, o)
    if vtype == 9:                                        # Array
        et = u32(b, o); o += 4
        cnt = u64(b, o); o += 8
        out = []
        for _ in range(cnt):
            v, o = parse_value(b, o, et)
            out.append(v)
        return out, o
    if vtype == 10: return u64(b, o), o+8
    if vtype == 11: return struct.unpack_from("<q", b, o)[0], o+8
    if vtype == 12: return f64(b, o), o+8
    raise ValueError(f"bad vtype {vtype}")

raw = open(PATH, "rb").read()
o = 0
assert raw[o:o+4] == b"GGUF"; o += 4
version = u32(raw, o); o += 4
tensor_count = u64(raw, o); o += 8
kv_count = u64(raw, o); o += 8
meta = {}
order = []
for _ in range(kv_count):
    key, o = read_str(raw, o)
    vt = u32(raw, o); o += 4
    val, o = parse_value(raw, o, vt)
    meta[key] = val
    order.append(key)

out = {"GGUF.version": version, "GGUF.tensor_count": tensor_count,
       "GGUF.kv_count": kv_count, "kv_order": order}
for k in order:
    v = meta[k]
    if isinstance(v, list):
        print(f"LIST {k} len={len(v)}")
        out[k] = v
    else:
        print(f"{k} = {v!r}")
        out[k] = v

with open("C:/Users/niber/AppData/Local/hermes/workspace/titan/testdata/golden_meta.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, ensure_ascii=False, default=str)

# sanity
import hashlib
h = hashlib.sha256(raw).hexdigest()
print("FILE_SHA256", h)
print("END_HEADER_OFFSET", o)