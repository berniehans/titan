# Change: Bootstrap del workspace (Fase 0-1)

## Why
Punto de partida del motor: infraestructura de build, CI y las primitivas de I/O (parser GGUF + memoria pinned) sobre las que descansa todo lo demás. Es el change más pequeño que produce valor verificable.

## What Changes
- Crear Cargo workspace con 5 crates vacíos (`engine-api`, `engine-core`, `engine-io`, `engine-cuda`, `engine-kvcache`) + rust-toolchain.toml + CI GitHub Actions.
- Implementar parser GGUF v3 (metadatos + tensor infos) en engine-io con tests contra fixture.
- Implementar reserva RAII de RAM pinned alineada a 4096 B vía cudarc/FFI en engine-cuda.
- Cargar fixture Qwen3-0.6B Q4_K_M completo a pinned con métrica GB/s.

## Non-goals
- Nada de kernels, forward pass, streaming ni servidor HTTP (changes posteriores).
- No soportar formatos distintos de GGUF.

## Impact
- **Affected specs:** layer-streaming-engine (requisito "Carga única de pesos a RAM pinned")
- **Affected code:** nuevo workspace completo bajo `engine/`, fixtures en `testdata/`
- **Gate:** `cargo test` verde + fixture cargado <5 s + clippy limpio

## Tasks (resumen — detalle en tasks.md)
1. Workspace + toolchain + CI
2. Fixture descargable con checksums
3. Parser GGUF (TDD)
4. Pinned memory RAII (TDD)
5. Loader completo con métricas
