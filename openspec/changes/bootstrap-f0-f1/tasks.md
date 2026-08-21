# Tasks: bootstrap-f0-f1

> Ejecutar con harness (agy gemini-3.7-flash-high vía perfil coder). TDD estricto. Commit por task.

## 1. Workspace
- [x] 1.1 Crear `engine/Cargo.toml` workspace con members = los 5 crates; deps compartidas en workspace.dependencies (cudarc 0.12 cuda-12000, tokio, axum, anyhow, thiserror, tracing)
- [x] 1.2 Crear crates vacíos con src/lib.rs + rust-toolchain.toml (stable)
- [x] 1.3 CI GitHub Actions: fmt --check, clippy -D warnings, test (GPU tests con #[ignore])
- [x] 1.4 Verificar: `cargo build && cargo clippy -- -D warnings && cargo test` verde

## 2. Fixture
- [x] 2.1 Script tools/download_fixture.sh: descarga Qwen3-0.6B Q4_K_M GGUF (~400 MB) a testdata/, idempotente, registra SHA256 en testdata/CHECKSUMS.md
- [x] 2.2 Verificar descarga y checksum

## 3. Parser GGUF (engine-io)
- [x] 3.1 Test fallido: parsear header fixture → magic "GGUF", version 3
- [x] 3.2 Implementar lectura de header + metadata KV (tipos u8..f64, string, array)
- [x] 3.3 Test fallido: tensor infos → nombre/dims/tipo/offset de todos los tensores del fixture
- [x] 3.4 Implementar tensor infos; validar contra gguf-dump de referencia
- [x] 3.5 Test fallido: mapear tensores por patrón de nombre (blk.N.*, token_embd, output)
- [x] 3.6 Implementar indexación por capa para carga streaming posterior
- [x] 3.7 Verificar: cargo test -p engine-io verde

## 4. Pinned memory (engine-cuda)
- [ ] 4.1 Test fallido (#[ignore] sin GPU): reservar 256 MB pinned, escribir patrón, leer igual, Drop libera (contador debug)
- [ ] 4.2 Implementar wrapper RAII cudaMallocHost/cudaFreeHost alineado a 4096 B con // SAFETY:
- [ ] 4.3 Verificar test PASS en GPU local

## 5. Loader completo
- [ ] 5.1 Test fallido: cargar fixture completo → suma bytes == tamaño archivo, tensores en regiones contiguas por capa
- [ ] 5.2 Implementar loader: lee GGUF una vez, escribe tensores a pinned, loguea GB/s
- [ ] 5.3 Gate F0-F1: fixture cargado <5 s, clippy limpio, todos los tests verdes
