# Titan — Estado y organización del workspace

**Corte:** 2026-09-02
**Rama:** `master`
**HEAD histórico:** `97993d0`
**Estado:** checkout local ampliamente modificado; no es un commit release.

## Regla de verdad

Este documento resume el estado operativo actual. Los números de rendimiento y correctness solo son válidos cuando apuntan a un artefacto JSON/log concreto. README y tablas históricas no sustituyen la evidencia de `local-artifacts/`.

## Planos arquitectónicos

| Plano | Responsabilidad | Estado actual |
|---|---|---|
| Ingestión/representación | GGUF, metadatos, tokenizer, cuantización | Implementado; Q8 requiere contrato explícito |
| Ejecución GPU/CPU/memoria | kernels CUDA, GEMV/GEMM, attention, KV | Q6_K ABI reparado; gates GPU verdes |
| Runtime/generación/scheduling | ForwardDriver, graph, batching, decode | F32 operativo; Q8 sigue experimental |
| Servidor/API/CLI | HTTP, SSE, JSON, CLI, tool calling | E2E verificado en la configuración actual |
| Evidencia/observabilidad/release | telemetría, benchmarks, gates, OpenSpec | Instrumentación completa; release bloqueado |

## Evidencia vigente

### Correctness

- `local-artifacts/reviews/real-q8-layer0-parity-final-20260902.json`
  - 13 stages.
  - Primer fallo restante: `q_projection_q8`.
  - Full layer: `rel_l2=0.01261698`, `cosine=0.999922688`.
- `local-artifacts/reviews/real-q8-single-decode-differential-20260902.json`
  - Single-step full-driver F32 vs Q8.
  - `rel_l2=0.0465303149`, `cosine=0.9989238592`, finito.
- Reparación Q6_K en `engine/engine-cuda/src/batched_gemm.rs`:
  - Antes V: `rel_l2=0.667568`, `cosine=0.744549`.
  - Después V: `rel_l2=0.006830045`, `cosine=0.999976680`.

### Observabilidad

- `local-artifacts/reviews/q8-dispatch-probe-final-20260902.json`
  - 211 launches, 8 records, 0 roles/variantes incompletos.
- La telemetría de atribución y el benchmark de aceptación deben continuar separados.

### Performance/reference

- Benchmark Q8 post-fix: `local-artifacts/benchmarks/final-titan-q8-after-q6k-fix-20260902_180740.json`.
- Head-to-head fresco: `local-artifacts/benchmarks/final-head-to-head-20260902_165459.json`.
- Gate fresco: `local-artifacts/reviews/fresh-head-to-head-release-gate-20260902.json`.
- Resultado agregado actual: `0.4955685x`; objetivo: `>=0.95x` por modelo y agregado.
- Candidato FFN-down 2col: `rejected_neutral`, delta `+0.0491%`.

### Toolchain/profile

- llama.cpp fuente: `local-artifacts/llama.cpp`.
- llama.cpp build mínimo/runtime: `local-artifacts/llama-build-cuda/`.
- CUDA Toolkit/nvcc: `13.3.73`.
- Smoke `llama-server.exe --version`: no verificado; Bash y launcher nativo devolvieron `exit 127` después de conservar el binario. No se declara runtime funcional hasta resolverlo.
- Nsight: bloqueado por `ERR_NVGPUCTRPERM`; no se infieren counters.

## Organización

```text
engine/                    código y tests Titan
openspec/                  especificaciones y change activo
docs/                      documentación estable y estado operativo
tools/                     tooling reproducible de benchmark/gates
local-artifacts/
  benchmarks/              JSON/log de corridas
  reviews/                 decisiones, gates y diagnósticos
  profiles/                capturas Nsight y perfiles parciales
  llama.cpp/               fuente de referencia externa
  llama-build-cuda/bin/   binarios mínimos de llama.cpp
  nvrtc-cu12/runtime/      DLLs requeridas por Titan
  release-candidate/       snapshots RC autocontenidos
  archive/                 material histórico/no operativo
.hermes/plans/              planes activos; planes superseded en archive/
```

## Política de retención

### Conservar en ubicación operativa

- Código y tests, aunque estén no commiteados, hasta revisión explícita.
- OpenSpec activo y sus logs de investigación.
- JSON/log finales de benchmarks y gates.
- Diagnósticos que justifican una decisión técnica.
- Fuente `llama.cpp`, runtime NVRTC y DLLs ejecutables.
- Capturas Nsight que contienen evidencia válida o bloqueo documentado.

### Archivar, no borrar

- Planes reemplazados.
- Notas de instrucciones de agentes/coder.
- Bundles RC superseded.
- Logs auxiliares que explican un experimento, pero no son evidencia primaria.

### Eliminar como reconstruible

Solo intermediarios de build de llama.cpp después de conservar binarios, scripts, source commit y manifest. No eliminar GGUF, secretos, código, tests ni evidencia única.

## Bloqueantes de release

1. Drift Q8 frente al contrato estricto; Q8 no es default de producción.
2. Rendimiento agregado `0.4955685x` frente al gate `0.95x`.
3. Nsight sin counters por `ERR_NVGPUCTRPERM`.

## Verificaciones mínimas antes de cualquier commit futuro

```bash
cargo fmt --all -- --check
cargo check --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
openspec validate --all
```

Las suites `#[ignore]` GPU/E2E se reportan por separado y solo cuentan cuando se ejecutan explícitamente.

## Cambios de higiene de este corte

La limpieza de workspace debe quedar registrada en:

```text
local-artifacts/manifests/workspace-cleanup-20260902.json
```

No se autoriza `git clean`, `git reset`, `git restore`, commit ni push como parte de esta higiene.
