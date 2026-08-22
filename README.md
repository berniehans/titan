# Titan

Motor de inferencia LLM en Rust + CUDA para modelos GGUF cuyos pesos **no caben en VRAM**: los tensores residen en RAM pinned (NVMe → host una sola vez) y se transmiten por capa hacia la GPU con pipeline de doble búfer.

## Estado actual — Fase 0-1 (bootstrap)

| Componente | Estado |
|---|---|
| Cargo workspace (5 crates) + toolchain estable + CI | ✅ |
| Parser GGUF v3 (header, metadata KV, tensor infos, layer index) | ✅ TDD |
| RAM pinned RAII (`cuMemAllocHost`/`cuMemFreeHost`, alineado 4096 B) | ✅ TDD, GPU |
| Loader single-pass NVMe → pinned con métrica GB/s | ✅ (~400 MB en <1 s) |
| Streaming por capa, doble búfer CUDA, kernels dequant, KV cache | ⏳ próximas fases |

**Hardware de referencia:** RTX 3060 6 GB · objetivo: denso 14B Q4_K_M (~8.5 GB en RAM) a ≈1.4 tok/s medidos sobre PCIe x8.

## Arquitectura

```
engine/
├── engine-api        # contratos públicos del motor
├── engine-core       # orquestación y loop de generación
├── engine-io         # parser GGUF v3 + loader a pinned memory
├── engine-cuda       # FFI CUDA: pinned host RAII (próx.: streams, kernels)
└── engine-kvcache    # cache KV para atención
```

Principio central ([spec](openspec/specs/layer-streaming-engine/spec.md)): los pesos se leen del disco **una sola vez** al inicio; durante la generación no hay `read()`. La copia H2D de la capa N+1 se solapa con el cómputo de la capa N mediante dos streams sincronizados por eventos. Los pesos Q4_K_M se descuantizan dentro de los kernels GPU, sin materializar FP16 en VRAM.

## Uso

```bash
# 1. Descargar fixture de prueba (Qwen3-0.6B Q4_K_M, ~400 MB, idempotente, con SHA256)
bash tools/download_fixture.sh

# 2. Build + lint
cd engine
cargo build --workspace
cargo clippy --workspace -- -D warnings

# 3. Tests CPU
cargo test --workspace

# 4. Tests GPU (requieren CUDA device local; marcados #[ignore])
cargo test --workspace -- --ignored
```

Los tests que dependen del fixture GGUF hacen *skip* automático si el archivo no está presente (p. ej. en CI); localmente corren completo.

## CI

GitHub Actions: `cargo fmt --check`, `clippy -D warnings`, `cargo test` (CPU). Los tests GPU corren en hardware local.

## Desarrollo

Spec-driven con [OpenSpec](openspec/constitution.md): cada fase es un change en `openspec/changes/` con proposal, tasks y gate verificable antes de marcar done.
