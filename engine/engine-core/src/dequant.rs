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
