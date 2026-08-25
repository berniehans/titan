import json
from gguf import GGUFReader

r = GGUFReader("C:/Users/niber/AppData/Local/hermes/workspace/titan/testdata/Qwen3-0.6B-Q4_K_M.gguf")
out = {}
for name, fields in r.fields.items():
    f = fields[0]
    val = f.contents() if hasattr(f, "contents") else None
    out[name] = val

with open("C:/Users/niber/AppData/Local/hermes/workspace/titan/testdata/golden_meta_full.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False, default=str)
print("WROTE", len(out), "keys")

# Summarize scalar config values (those we need for ModelConfig)
for k in sorted(out):
    v = out[k]
    if isinstance(v, str):
        s = v if len(v) < 60 else v[:57] + "..."
        print(f"{k} = STR({len(v)}) '{s}'")
    elif isinstance(v, list):
        print(f"{k} = LIST[{len(v)}]")
    else:
        print(f"{k} = {v!r}")