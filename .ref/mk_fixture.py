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

# ---- CLEAN FIXTURE: exact integer expected output ----
# d=1.0, dmin=0.5. scales/mins chosen small; nibbles 0..15.
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
# nibble pattern: low= i (0..15 pattern), high= 16-i-ish. deterministic, exact ints via d1*(q&F)-m1
for g in range(4):
    for i in range(32):
        lo = (g*4 + i) & 0xF
        hi = (g*4 + i + 3) & 0xF
        qs[g*32 + i] = lo | (hi << 4)

exp = dequant_block(list(qs), scales, d, dmin)
# all expected must be exact (integer or .5) -> dmin=0.5 makes .5 possible; d1 integer*sc
exp_d2 = []
is_idx = 0
for j in range(0,256,64):
    sc,m = get_scale_min_k4(is_idx+0, scales); d1=d*sc; m1=dmin*m
    sc,m = get_scale_min_k4(is_idx+1, scales); d2=d*sc; m2=dmin*m
    for i in range(32):
        b = qs[(j//64)*32+i]
        exp_d2.append(d1*(b&0xF)-m1)
    for i in range(32):
        b = qs[(j//64)*32+i]
        exp_d2.append(d2*(b>>4)-m2)
    is_idx += 2
assert all(abs(a-b)<1e-9 for a,b in zip(exp,exp_d2))

block = bytearray()
block += pack('<e', d); block += pack('<e', dmin)
block += bytes(scales); block += bytes(qs)
assert len(block) == 144

def rs(arr): return "[" + ", ".join(str(x) for x in arr) + "]"

print("== BLOCK: [d,dmin,scales,qs] ==")
print(rs(list(block)))
print()
print("== EXPECTED (256 floats) ==")
print(rs([round(x,4) for x in exp]))
print()
print("fract present?", any(x != int(x) for x in exp))