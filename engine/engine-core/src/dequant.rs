//! Q4_K block layout: 144 bytes per super-block of 256 weights (8 sub-blocks x 32).
//!
//! Layout details for Q4_K block quantization format (144 bytes per 256-element block):
//! - bytes[0..2]:   fp16 LE `d`    — super scale for the 6-bit sub-block scales
//! - bytes[2..4]:   fp16 LE `dmin` — super scale for the 6-bit sub-block mins
//! - bytes[4..16]:  `scales[12]`   — packs 8 sub-block 6-bit scale values + 8 sub-block 6-bit min values
//! - bytes[16..144]: `qs[128]`     — 256 weights packed 4-bit nibbles, 2 per byte

fn get_scale_min(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        let scale = scales[j] & 63;
        let min = scales[j + 4] & 63;
        (scale, min)
    } else {
        let scale = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        let min = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
        (scale, min)
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    let val = if exp == 0 {
        if mant == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }
        (mant as f32) * 2.0f32.powi(-24)
    } else if exp == 31 {
        f32::INFINITY
    } else {
        (1.0 + (mant as f32) / 1024.0) * 2.0f32.powi((exp as i32) - 15)
    };
    if sign == 1 { -val } else { val }
}

/// Dequantizes Q4_K quantized bytes into 32-bit floats.
///
/// Returns 256 floats per 144-byte Q4_K block.
pub fn dequant_q4k_cpu(src: &[u8]) -> Vec<f32> {
    assert!(
        src.len() == 144,
        "Q4_K block dequantization expects exactly 144 bytes, got {}",
        src.len()
    );

    let d = f16_to_f32(u16::from_le_bytes([src[0], src[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([src[2], src[3]]));
    let scales = &src[4..16];
    let qs = &src[16..144];

    let mut out = Vec::with_capacity(256);
    let mut is_idx = 0;

    for jgrp in (0..256).step_by(64) {
        let (sc0, m0) = get_scale_min(is_idx, scales);
        let d1 = d * sc0 as f32;
        let m1 = dmin * m0 as f32;

        let (sc1, m1s) = get_scale_min(is_idx + 1, scales);
        let d2 = d * sc1 as f32;
        let m2 = dmin * m1s as f32;

        let qbase = (jgrp / 64) * 32;

        for l in 0..32 {
            out.push(d1 * (qs[qbase + l] & 0xF) as f32 - m1);
        }
        for l in 0..32 {
            out.push(d2 * (qs[qbase + l] >> 4) as f32 - m2);
        }

        is_idx += 2;
    }

    out
}

/// Q6_K block layout (210 bytes per 256 weights, per llama.cpp `block_q6_K`):
/// - bytes[0..128):  `ql[128]`  — 256 quants, low 4 bits (2 nibbles per byte, 128 bytes)
/// - bytes[128..192): `qh[64]`  — 256 quants, high 2 bits (4 per byte)
/// - bytes[192..208): `scales[16]` — 16 int8 block scales (16 groups of 16)
/// - bytes[208..210): `d`       — fp16 LE super-block scale
///
/// Dequant (llama.cpp `dequantize_row_q6_K`): each 32-element lane relates to
/// `n` (0 or 128) and `l`; `qN = ((ql nibble) | ((qh bits)&3)<<4) - 32`; the
/// value is `y = d * scales[k] * q` (left-to-right fp32 multiplications).
pub fn dequant_q6k_cpu(src: &[u8]) -> Vec<f32> {
    assert_eq!(src.len(), 210, "Q6_K block dequant expects 210 bytes");
    let d = f16_to_f32(u16::from_le_bytes([src[208], src[209]]));
    let ql = &src[0..128];
    let qh = &src[128..192];
    let scales = &src[192..208];

    let mut out = vec![0.0f32; 256];
    let mut y = 0usize;
    let mut ql_off = 0usize;
    let mut qh_off = 0usize;
    let mut sc_off = 0usize;
    for _half in 0..2 {
        for l in 0..32 {
            let is = l / 16;
            let q1 = (((ql[ql_off + l] & 0x0F) as i32)
                | ((qh[qh_off + l] as i32 >> 0) & 3) << 4)
                - 32;
            let q2 = (((ql[ql_off + l + 32] & 0x0F) as i32)
                | ((qh[qh_off + l] as i32 >> 2) & 3) << 4)
                - 32;
            let q3 = (((ql[ql_off + l] as i32 >> 4) & 0x0F)
                | ((qh[qh_off + l] as i32 >> 4) & 3) << 4)
                - 32;
            let q4 = (((ql[ql_off + l + 32] as i32 >> 4) & 0x0F)
                | ((qh[qh_off + l] as i32 >> 6) & 3) << 4)
                - 32;
            out[y + l] = d * scales[sc_off + is] as i8 as f32 * q1 as f32;
            out[y + l + 32] = d * scales[sc_off + 2 + is] as i8 as f32 * q2 as f32;
            out[y + l + 64] = d * scales[sc_off + 4 + is] as i8 as f32 * q3 as f32;
            out[y + l + 96] = d * scales[sc_off + 6 + is] as i8 as f32 * q4 as f32;
        }
        y += 128;
        ql_off += 64;
        qh_off += 32;
        sc_off += 8;
    }
    out
}

/// Dequantizes the column `j` (of `ne0` elements, `Q6_K` blocks) — not used by
/// the CPU bank path but validated for round-trips and readability.
#[cfg(test)]
mod q6k_sanity {
    use super::*;
    #[test]
    fn dequant_q6k_all_zero_is_min_scaled() {
        // q = (0 nibble | 0 << 4) - 32 = -32; d = 1.0, all scales = 1
        let mut blk = [0u8; 210];
        blk[208..210].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = fp16 1.0
        for s in blk[192..208].iter_mut() {
            *s = 1; // scales (int8) = 1
        }
        let r = dequant_q6k_cpu(&blk);
        assert_eq!(r.len(), 256);
        for v in &r {
            assert_eq!(*v, -32.0, "all elements must be d*sc*q = 1*1*(-32)");
        }
    }

    #[test]
    fn dequant_q6k_matches_independent_block() {
        // Exact 210-byte block-0 of `token_embd.weight` row 9707 (Qwen3 fixture).
        // Values independently verified against llama.cpp `dequantize_row_q6_K`
        // semantics (fp16 d, int8 scales) — and cross-checked against the
        // 6.6 numpy reference that reproduces golden L0 to cos-sim 0.9998.
        const HEX: &str = "6139b402f8b067305faa6fcaf0ca08875deff35accf00784b7d2ae549685a601200434f9cbb540043076908f0fcf97ac7d6f09f3781085e17213a555605b692c1cb20649ab5f2f9e99f0a04f8c1b81f423d83fb8990ef0849e812f90577c4aa58c731394714c457f95d9fa9c1552dfe8d35c2103c4327911f42d6b3b99b3d14be21a9562996f5db4aa96dd5ae6754ba6e0a8a661d65e9760460a4e8615da5688a98d4a997054aa56a75959a1287874581d597eda368a672292f790d8aa696cd8b5ae3f4bc5afa94458c080bb494b48a46f01";
        let b: Vec<u8> = (0..HEX.len() / 2)
            .map(|i| u8::from_str_radix(&HEX[2 * i..2 * i + 2], 16).unwrap())
            .collect();
        assert_eq!(b.len(), 210);
        let r = dequant_q6k_cpu(&b);
        // Independent expectation (fp16 d=0x016f, int8 scales, q=-32..):
        let want = [
            -1.6406178474e-03, -1.4765560627e-02, 1.9687414169e-02, -3.2812356949e-03,
            1.3124942780e-02, -2.6249885559e-02, 1.4765560627e-02, 5.2499771118e-02,
        ];
        for (a, e) in r[..8].iter().zip(want.iter()) {
            assert!((a - e).abs() < 2e-7, "q6 elem mismatch: {a} vs {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dequant_q4k_reproduces_known_block() {
        const BLOCK_BYTES: [u8; 144] = [
            0, 60, 0, 56, 10, 131, 31, 192, 2, 8, 65, 63, 17, 12, 69, 95, 48, 65, 82, 99, 116, 133,
            150, 167, 184, 201, 218, 235, 252, 13, 30, 47, 48, 65, 82, 99, 116, 133, 150, 167, 184,
            201, 218, 235, 252, 13, 30, 47, 116, 133, 150, 167, 184, 201, 218, 235, 252, 13, 30,
            47, 48, 65, 82, 99, 116, 133, 150, 167, 184, 201, 218, 235, 252, 13, 30, 47, 48, 65,
            82, 99, 184, 201, 218, 235, 252, 13, 30, 47, 48, 65, 82, 99, 116, 133, 150, 167, 184,
            201, 218, 235, 252, 13, 30, 47, 48, 65, 82, 99, 116, 133, 150, 167, 252, 13, 30, 47,
            48, 65, 82, 99, 116, 133, 150, 167, 184, 201, 218, 235, 252, 13, 30, 47, 48, 65, 82,
            99, 116, 133, 150, 167, 184, 201, 218, 235,
        ];

        const EXPECTED: [f32; 256] = [
            -1.0, 9.0, 19.0, 29.0, 39.0, 49.0, 59.0, 69.0, 79.0, 89.0, 99.0, 109.0, 119.0, 129.0,
            139.0, 149.0, -1.0, 9.0, 19.0, 29.0, 39.0, 49.0, 59.0, 69.0, 79.0, 89.0, 99.0, 109.0,
            119.0, 129.0, 139.0, 149.0, 5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0, 26.0, 29.0, 32.0,
            35.0, 38.0, 41.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0, 26.0, 29.0,
            32.0, 35.0, 38.0, 41.0, -4.0, -1.0, 2.0, 123.5, 154.5, 185.5, 216.5, 247.5, 278.5,
            309.5, 340.5, 371.5, 402.5, 433.5, 464.5, -0.5, 30.5, 61.5, 92.5, 123.5, 154.5, 185.5,
            216.5, 247.5, 278.5, 309.5, 340.5, 371.5, 402.5, 433.5, 464.5, -0.5, 30.5, 61.5, 92.5,
            -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5,
            -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5,
            -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, -31.5, 7.5, 8.5, 9.5, 10.5, 11.5,
            12.5, 13.5, 14.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5,
            12.5, 13.5, 14.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 484.0, 528.0, 572.0, 616.0,
            660.0, 0.0, 44.0, 88.0, 132.0, 176.0, 220.0, 264.0, 308.0, 352.0, 396.0, 440.0, 484.0,
            528.0, 572.0, 616.0, 660.0, 0.0, 44.0, 88.0, 132.0, 176.0, 220.0, 264.0, 308.0, 352.0,
            396.0, 440.0, 50.0, 55.0, 60.0, 65.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0,
            30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0,
            20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 942.5, -2.5, 60.5, 123.5, 186.5, 249.5, 312.5,
            375.5, 438.5, 501.5, 564.5, 627.5, 690.5, 753.5, 816.5, 879.5, 942.5, -2.5, 60.5,
            123.5, 186.5, 249.5, 312.5, 375.5, 438.5, 501.5, 564.5, 627.5, 690.5, 753.5, 816.5,
            879.5,
        ];

        let actual = dequant_q4k_cpu(&BLOCK_BYTES);
        assert_eq!(actual.len(), EXPECTED.len());
        for (a, e) in actual.iter().zip(EXPECTED.iter()) {
            assert!(
                (a - e).abs() < 1e-6,
                "Mismatch: actual {} vs expected {}",
                a,
                e
            );
        }
    }
}
