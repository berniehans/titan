import json
from gguf import GGUFReader

r = GGUFReader("C:/Users/niber/AppData/Local/hermes/workspace/titan/testdata/Qwen3-0.6B-Q4_K_M.gguf")
out = {}
for name, fields in r.fields.items():
    f = fields[0]
    if hasattr(f, "parts") and f.parts:
        parts = f.parts
        if len(parts) == 1:
            out[name] = parts[0]
        else:
            out[name] = [p for p in parts]
    else:
        out[name] = str(f)

with open("C:/Users/niber/AppData/Local/hermes/workspace/titan/testdata/golden_meta_dump.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False, default=str)
print("WROTE", len(out), "keys")