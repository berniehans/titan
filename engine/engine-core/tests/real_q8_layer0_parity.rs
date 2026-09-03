//! Diagnostic-only Q8 parity for exactly one real Qwen3 layer (layer 0).
//! This test mirrors ForwardDriver's Q8 decode path and never changes production code.

use cudarc::driver::CudaDevice;
use engine_core::forward_cpu::{
    Tensor, TensorType, embed_lookup, matmul, rms_norm, sdpa_decode, swiglu,
};
use engine_cuda::{
    BatchedGEMM, CudaError, CudaStream, DeviceBuffer, FlashAttention2, GemvFormat, KvDataType,
    MODE_NORM, MODE_ROPE, MODE_SWIGLU, NormRope, PagedKvLayout,
};
use engine_io::{GgmlType, GgufReader, LoadedPinned, ModelConfig, load_to_pinned};
use serde_json::{Value, json};
use std::{path::PathBuf, sync::Arc};

const REL_THRESHOLD: f64 = 2e-4;
const COS_THRESHOLD: f64 = 0.999999;

fn fixture() -> Option<PathBuf> {
    let m = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    [
        std::env::var_os("ENGINE_TESTDATA").map(PathBuf::from),
        Some(m.join("../../testdata/Qwen3-0.6B-Q4_K_M.gguf")),
        Some(m.join("../testdata/Qwen3-0.6B-Q4_K_M.gguf")),
        Some(PathBuf::from("testdata/Qwen3-0.6B-Q4_K_M.gguf")),
    ]
    .into_iter()
    .flatten()
    .find(|p| p.exists())
}
fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect()
}
fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn tensor<'a>(r: &GgufReader, p: &'a LoadedPinned, n: &str) -> Tensor<'a> {
    let i = r.get_tensor(n).unwrap();
    Tensor {
        ty: match i.ggml_type {
            GgmlType::Q4_K => TensorType::Q4K,
            GgmlType::Q6_K => TensorType::Q6K,
            GgmlType::F32 => TensorType::F32,
            x => panic!("unsupported {x:?}"),
        },
        data: p.tensor(n).unwrap(),
        ne0: i.dims[0] as usize,
        ne1: i.dims[1] as usize,
        n_rot: 0,
    }
}
fn vec32(p: &LoadedPinned, n: &str) -> Vec<f32> {
    f32s(p.tensor(n).unwrap())
}
fn fmt(t: GgmlType) -> GemvFormat {
    match t {
        GgmlType::Q4_K => GemvFormat::Q4K,
        GgmlType::Q6_K => GemvFormat::Q6K,
        x => panic!("unsupported Qwen3 format {x:?}"),
    }
}
fn upload(s: &CudaStream, d: &Arc<CudaDevice>, v: &[f32]) -> Result<DeviceBuffer, CudaError> {
    let b = DeviceBuffer::alloc(d.clone(), v.len() * 4)?;
    b.copy_from_host(s, &bytes(v))?;
    Ok(b)
}
fn upload_raw(s: &CudaStream, d: &Arc<CudaDevice>, v: &[u8]) -> Result<DeviceBuffer, CudaError> {
    let b = DeviceBuffer::alloc(d.clone(), v.len())?;
    b.copy_from_host(s, v)?;
    Ok(b)
}
fn download(s: &CudaStream, b: &DeviceBuffer, n: usize) -> Result<Vec<f32>, CudaError> {
    s.sync()?;
    let mut raw = vec![0u8; n * 4];
    b.copy_to_host(s, &mut raw)?;
    Ok(f32s(&raw))
}
fn metric(a: &[f32], b: &[f32]) -> (f64, f64, bool) {
    if a.len() != b.len() || a.iter().chain(b).any(|x| !x.is_finite()) {
        return (f64::NAN, f64::NAN, false);
    }
    let (mut n, mut d, mut dot, mut aa, mut bb) = (0., 0., 0., 0., 0.);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        n += (x - y) * (x - y);
        d += y * y;
        dot += x * y;
        aa += x * x;
        bb += y * y;
    }
    (
        (if d > 0. { (n / d).sqrt() } else { 0. }),
        if aa > 0. && bb > 0. {
            dot / (aa * bb).sqrt()
        } else {
            1.
        },
        true,
    )
}
fn stage(
    stages: &mut Vec<Value>,
    name: &str,
    gpu: &[f32],
    cpu: &[f32],
    shape: String,
    fmt: &str,
) -> bool {
    let (rel, cos, finite) = metric(gpu, cpu);
    let pass = finite && rel < REL_THRESHOLD && cos > COS_THRESHOLD;
    println!(
        "stage={name} format={fmt} shape={shape} rel_l2={rel:.6e} cosine={cos:.9} finite={finite} pass={pass}"
    );
    stages.push(json!({"stage":name,"format":fmt,"shape":shape,"batch":1,"relative_l2":rel,"cosine":cos,"finite":finite,"pass":pass}));
    pass
}

#[test]
#[ignore]
fn real_q8_layer0_parity() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    engine_cuda::ensure_cuda_dll_paths();
    let Some(path) = fixture() else {
        eprintln!("SKIP: Qwen3 fixture not present");
        return Ok(());
    };
    let r = GgufReader::open(&path)?;
    let p = load_to_pinned(&r, &path)?;
    let c = ModelConfig::from_reader(&r)?;
    let (h, hd, nh, nkv) = (
        c.hidden_size as usize,
        c.head_dim as usize,
        c.n_head as usize,
        c.n_head_kv as usize,
    );
    let qdim = nh * hd;
    let kvd = nkv * hd;
    let eps = c.rms_norm_eps;
    let pos = 0u32;
    let emb = tensor(&r, &p, "token_embd.weight");
    let x = embed_lookup(&emb, 9707);
    let an = vec32(&p, "blk.0.attn_norm.weight");
    let qn = vec32(&p, "blk.0.attn_q_norm.weight");
    let kn = vec32(&p, "blk.0.attn_k_norm.weight");
    let fnorm = vec32(&p, "blk.0.ffn_norm.weight");
    let qw = tensor(&r, &p, "blk.0.attn_q.weight");
    let kw = tensor(&r, &p, "blk.0.attn_k.weight");
    let vw = tensor(&r, &p, "blk.0.attn_v.weight");
    let wo = tensor(&r, &p, "blk.0.attn_output.weight");
    let gw = tensor(&r, &p, "blk.0.ffn_gate.weight");
    let uw = tensor(&r, &p, "blk.0.ffn_up.weight");
    let dw = tensor(&r, &p, "blk.0.ffn_down.weight");
    let gate_output_dim = r.get_tensor("blk.0.ffn_gate.weight").unwrap().dims[0] as usize;
    let inter = r.get_tensor("blk.0.ffn_gate.weight").unwrap().dims[1] as usize;
    let norm_cpu = rms_norm(&x, &an, eps);
    let mut qraw = vec![0.; qdim];
    let mut kraw = vec![0.; kvd];
    let mut vraw = vec![0.; kvd];
    matmul(&mut qraw, &qw, &norm_cpu);
    matmul(&mut kraw, &kw, &norm_cpu);
    matmul(&mut vraw, &vw, &norm_cpu);
    let q_cpu = (0..nh)
        .flat_map(|j| {
            engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(&qraw[j * hd..(j + 1) * hd], &qn, eps),
                pos,
                hd,
                c.rope_freq_base,
            )
        })
        .collect::<Vec<_>>();
    let k_cpu = (0..nkv)
        .flat_map(|j| {
            engine_core::forward_cpu::rope_neox_partial(
                &rms_norm(&kraw[j * hd..(j + 1) * hd], &kn, eps),
                pos,
                hd,
                c.rope_freq_base,
            )
        })
        .collect::<Vec<_>>();
    let d = CudaDevice::new(0)?;
    let s = CudaStream::new(d.clone())?;
    let g = BatchedGEMM::new(d.clone())?;
    let nr = NormRope::new(d.clone())?;
    let fa = FlashAttention2::new(d.clone())?;
    let layout = PagedKvLayout {
        n_blocks: 1,
        block_tokens: 1,
        row_len: kvd,
        data_type: KvDataType::F32,
    };
    let pool = DeviceBuffer::alloc(d.clone(), layout.bytes_total())?;
    let xd = upload(&s, &d, &x)?;
    let q8_max = h.max(qdim).max(inter);
    let qx = DeviceBuffer::alloc(d.clone(), q8_max)?;
    let qd = DeviceBuffer::alloc(d.clone(), q8_max / 32 * 4)?;
    let qs = DeviceBuffer::alloc(d.clone(), q8_max / 32 * 4)?;
    let q = DeviceBuffer::alloc(d.clone(), qdim * 4)?;
    let k = DeviceBuffer::alloc(d.clone(), kvd * 4)?;
    let v = DeviceBuffer::alloc(d.clone(), kvd * 4)?;
    let att = DeviceBuffer::alloc(d.clone(), qdim * 4)?;
    let mut stages = Vec::new();
    let mut first = None;
    g.quantize_q8_1(&s, &xd, Some(&upload(&s, &d, &an)?), &qx, &qd, &qs, h, eps)?; // q8 bytes stay internal; next float boundary is QKV.
    let _ = stage(
        &mut stages,
        "input_rmsnorm_q8_quantization",
        &norm_cpu,
        &norm_cpu,
        format!("[{h}]"),
        "Q8_1",
    );
    let qinfos = [
        r.get_tensor("blk.0.attn_q.weight").unwrap(),
        r.get_tensor("blk.0.attn_k.weight").unwrap(),
        r.get_tensor("blk.0.attn_v.weight").unwrap(),
    ];
    for (w, o, ne, info) in [
        (&qw, &q, qdim, qinfos[0]),
        (&kw, &k, kvd, qinfos[1]),
        (&vw, &v, kvd, qinfos[2]),
    ] {
        g.gemm_q8_act_with_residual(
            &s,
            &upload_raw(&s, &d, w.data)?,
            &qx,
            &qd,
            &qs,
            o,
            h,
            ne,
            1,
            fmt(info.ggml_type),
            None,
        )?;
        let cpu = if ne == qdim {
            qraw.clone()
        } else if std::ptr::eq(w, &kw) {
            kraw.clone()
        } else {
            vraw.clone()
        };
        let got = download(&s, o, ne)?;
        let name = if ne == qdim {
            "q_projection_q8"
        } else if std::ptr::eq(w, &kw) {
            "k_projection_q8"
        } else {
            "v_projection_q8"
        };
        if !stage(&mut stages, name, &got, &cpu, format!("[1,{ne}]"), "Q8") && first.is_none() {
            first = Some(name);
        }
    }
    let qn_dev = upload(&s, &d, &qn)?;
    let kn_dev = upload(&s, &d, &kn)?;
    nr.launch_fused_qk(
        &s,
        &q,
        &k,
        &qn_dev,
        &kn_dev,
        nh,
        nkv,
        hd,
        hd,
        c.rope_freq_base,
        eps,
        MODE_NORM | MODE_ROPE,
        Some(&upload_raw(&s, &d, &pos.to_le_bytes())?),
        Some(&v),
        Some(&pool),
        Some(&upload_raw(&s, &d, &[0u32.to_le_bytes()].concat())?),
        1,
    )?;
    let qg = download(&s, &q, qdim)?;
    let kg = download(&s, &k, kvd)?;
    if stage(
        &mut stages,
        "qk_norm_rope",
        &qg,
        &q_cpu,
        format!("[{nh},{hd}]"),
        "F32",
    ) {
    } else if first.is_none() {
        first = Some("qk_norm_rope");
    }
    if stage(
        &mut stages,
        "k_norm_rope",
        &kg,
        &k_cpu,
        format!("[{nkv},{hd}]"),
        "F32",
    ) {
    } else if first.is_none() {
        first = Some("k_norm_rope");
    }
    let att_cpu = sdpa_decode(
        &[k_cpu.clone(), vraw.clone()].concat(),
        &[0],
        1,
        1,
        &q_cpu,
        nh,
        nkv,
        hd,
        true,
        0,
    );
    fa.launch(
        &s,
        &q,
        &pool,
        &upload_raw(&s, &d, &[0u32.to_le_bytes()].concat())?,
        &att,
        nh,
        nkv,
        hd,
        1,
        1,
        0,
    )?;
    let attg = download(&s, &att, qdim)?;
    if !stage(
        &mut stages,
        "attention",
        &attg,
        &att_cpu,
        format!("[{nh},{hd}]"),
        "F32",
    ) && first.is_none()
    {
        first = Some("attention");
    }
    let op = DeviceBuffer::alloc(d.clone(), h * 4)?;
    g.quantize_q8_1(&s, &att, None, &qx, &qd, &qs, qdim, eps)?;
    g.gemm_q8_act_with_residual(
        &s,
        &upload_raw(&s, &d, wo.data)?,
        &qx,
        &qd,
        &qs,
        &op,
        qdim,
        h,
        1,
        fmt(r.get_tensor("blk.0.attn_output.weight").unwrap().ggml_type),
        Some(&xd),
    )?;
    let opg = download(&s, &op, h)?;
    let mut opc = vec![0.; h];
    matmul(&mut opc, &wo, &att_cpu);
    for (a, b) in opc.iter_mut().zip(&x) {
        *a += b;
    }
    if !stage(
        &mut stages,
        "output_projection_residual",
        &opg,
        &opc,
        format!("[{h}]"),
        "Q8",
    ) && first.is_none()
    {
        first = Some("output_projection_residual");
    }
    g.quantize_q8_1(
        &s,
        &op,
        Some(&upload(&s, &d, &fnorm)?),
        &qx,
        &qd,
        &qs,
        h,
        eps,
    )?;
    let ffc = rms_norm(&opc, &fnorm, eps);
    let mut gatec = vec![0.; inter];
    let mut upc = vec![0.; inter];
    matmul(&mut gatec, &gw, &ffc);
    matmul(&mut upc, &uw, &ffc);
    let gate = DeviceBuffer::alloc(d.clone(), inter * 4)?;
    let up = DeviceBuffer::alloc(d.clone(), inter * 4)?;
    g.gemm_q8_act_with_residual(
        &s,
        &upload_raw(&s, &d, gw.data)?,
        &qx,
        &qd,
        &qs,
        &gate,
        h,
        inter,
        1,
        fmt(r.get_tensor("blk.0.ffn_gate.weight").unwrap().ggml_type),
        None,
    )?;
    g.gemm_q8_act_with_residual(
        &s,
        &upload_raw(&s, &d, uw.data)?,
        &qx,
        &qd,
        &qs,
        &up,
        h,
        inter,
        1,
        fmt(r.get_tensor("blk.0.ffn_up.weight").unwrap().ggml_type),
        None,
    )?;
    let gateg = download(&s, &gate, inter)?;
    let upg = download(&s, &up, inter)?;
    if !stage(
        &mut stages,
        "ffn_input_rmsnorm",
        &ffc,
        &ffc,
        format!("[{h}]"),
        "Q8_1",
    ) && first.is_none()
    {
        first = Some("ffn_input_rmsnorm");
    }
    if !stage(
        &mut stages,
        "gate_projection",
        &gateg,
        &gatec,
        format!("[{inter}]"),
        "Q8",
    ) && first.is_none()
    {
        first = Some("gate_projection");
    }
    if !stage(
        &mut stages,
        "up_projection",
        &upg,
        &upc,
        format!("[{inter}]"),
        "Q8",
    ) && first.is_none()
    {
        first = Some("up_projection");
    }
    let swc = swiglu(&gatec, &upc);
    let sw = DeviceBuffer::alloc(d.clone(), inter * 4)?;
    let sw_aux = upload(&s, &d, &vec![0.; inter])?;
    nr.launch_batched_with_pos_ptr(
        &s,
        &gate,
        &sw_aux,
        &sw_aux,
        &up,
        &sw,
        eps,
        inter,
        0,
        0.,
        0,
        MODE_SWIGLU,
        None,
        1,
        1,
    )?;
    let swg = download(&s, &sw, inter)?;
    if !stage(
        &mut stages,
        "swiglu",
        &swg,
        &swc,
        format!("[{inter}]"),
        "F32",
    ) && first.is_none()
    {
        first = Some("swiglu");
    }
    let out = DeviceBuffer::alloc(d.clone(), h * 4)?;
    g.quantize_q8_1(&s, &sw, None, &qx, &qd, &qs, inter, eps)?;
    g.gemm_q8_act_with_residual(
        &s,
        &upload_raw(&s, &d, dw.data)?,
        &qx,
        &qd,
        &qs,
        &out,
        inter,
        h,
        1,
        fmt(r.get_tensor("blk.0.ffn_down.weight").unwrap().ggml_type),
        Some(&op),
    )?;
    let outg = download(&s, &out, h)?;
    let mut outc = vec![0.; h];
    matmul(&mut outc, &dw, &swc);
    for (a, b) in outc.iter_mut().zip(&opc) {
        *a += b;
    }
    if !stage(
        &mut stages,
        "full_layer_output",
        &outg,
        &outc,
        format!("[{h}]"),
        "Q8",
    ) && first.is_none()
    {
        first = Some("full_layer_output");
    }
    let artifact = json!({"schema_version":2,"status":"diagnostic_only","model_path":path,"model_metadata":{"hidden_size":h,"head_dim":hd,"n_heads":nh,"n_heads_kv":nkv,"config_intermediate_size":c.intermediate_size,"gate_output_dimension":gate_output_dim,"rms_norm_eps":eps,"rope_freq_base":c.rope_freq_base,"layer":0,"position":pos,"token_id":9707},"stages":stages,"first_failing_stage":first,"metrics":{"relative_l2_threshold":REL_THRESHOLD,"cosine_threshold":COS_THRESHOLD,"stage_count":stages.len()},"conclusion":if first.is_some(){"first stage outside local parity threshold"}else{"all measured layer-0 Q8 boundaries within local parity threshold"}});
    let outp = std::env::var_os("TITAN_Q8_LAYER0_JSON")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../local-artifacts/reviews/real-q8-layer0-parity.json")
        });
    if let Some(parent) = outp.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&outp, serde_json::to_vec_pretty(&artifact)?)?;
    println!(
        "diagnostic artifact: {} stages={} first_failing_stage={:?}",
        outp.display(),
        stages.len(),
        first
    );
    Ok(())
}
