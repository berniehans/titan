from struct import pack

def get_scale_min_k4(j, q):
    if j < 4:
        return q[j] & 63, q[j+4] & 63
    d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
    m = (q[j+4] >> 4) | ((q[j] >> 6) << 4)
    return d, m

def dequant_block(q, scales, d, dmin):
    out = []
    is_idx = 0
    for j in range(0, 256, 64):
        sc, m = get_scale_min_k4(is_idx+0, scales)
        d1 = d*sc; m1 = dmin*m
        sc, m = get_scale_min_k4(is_idx+1, scales)
        d2 = d*sc; m2 = dmin*m
        for l in range(32):
            out.append(d1*(q[l] & 0xF) - m1)
        for l in range(32):
            out.append(d2*(q[l] >> 4) - m2)
        q = q[32:]; is_idx += 2
    return out

sub = [(10,2),(3,8),(31,1),(0,63),(1,1),(44,0),(5,20),(63,5)]
scales = [0]*12
for j in range(8):
    ls, lm = sub[j]
    if j < 4:
        scales[j] = ls
        scales[j+4] = lm
    else:
        scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4)
        scales[j-4] |= ((ls >> 4) << 6)
        scales[j-0] |= ((lm >> 4) << 6)
for j in range(8):
    assert get_scale_min_k4(j, scales) == sub[j]

d = 1.0; dmin = 0.5
qs = [0]*128
for g in range(4):
    for i in range(32):
        lo = (g*4 + i) & 0xF
        hi = (g*4 + i + 3) & 0xF
        qs[g*32 + i] = lo | (hi << 4)
exp = dequant_block(list(qs), scales, d, dmin)

block = bytearray(pack('<e', d)) + bytearray(pack('<e', dmin)) + bytearray(scales) + bytearray(qs)
assert len(block) == 144

def rs(arr):
    return "[" + ", ".join(str(round(float(x),4)).rstrip('0').rstrip('.') if float(x)%1==0 else str(round(float(x),4)) for x in arr) + "]"

with open(".ref/fixture_block.txt","w") as f: f.write(rs(list(block)))
with open(".ref/fixture_expected.txt","w") as f: f.write(rs(exp))
print("wrote", len(block), "bytes and", len(exp), "expected floats")