# Constitution — Motor de Inferencia LLM en Rust

Reglas inmutables del proyecto. Toda spec y change referencia este documento. Cualquier cambio aquí requiere aprobación explícita de Bernie.

## 1. Propósito
Motor de inferencia LLM local en Rust para hardware restringido (RTX 3060 Laptop, 6 GB VRAM, PCIe 4.0 x8). Prioridad: correctness verificable > throughput > features. El caso de alto valor es MoE expert-streaming; el baseline es modelo denso residente.

## 2. Stack y convenciones (no negociables)
- **Rust stable** con `rust-toolchain.toml`. Repos Python auxiliares (tools/): UV obligatorio.
- Cargo workspace: `engine-api`, `engine-core`, `engine-io`, `engine-cuda`, `engine-kvcache`. Sin dependencias circulares entre crates.
- `cargo clippy -- -D warnings` limpio en todo commit.
- Formato de pesos: GGUF only para MVP.
- Errores con `thiserror` en libs, `anyhow` en bins. Sin `unwrap()` fuera de tests.
- `unsafe` solo en `engine-cuda`/`engine-io`, con comment `// SAFETY:` obligatorio y test que lo ejercite.

## 3. Proceso de desarrollo
- **TDD**: test antes que implementación en toda task. Paridad numérica contra referencia (llama.cpp) para todo kernel.
- **Gates por fase**: cada fase tiene criterio medible; NO se avanza sin gate verde. Gates de riesgo (kernels nuevos, cambios de pipeline CUDA) requieren visto bueno humano antes de correr en GPU.
- **Codegen**: implementación delegada a agy (gemini-3.7-flash-high) vía harness; review independiente (modelo distinto al writer) antes de merge. El orquestador nunca escribe código directamente.
- **Verificación honesta**: "listo" = tests corridos + output leído. Prohibido reportar éxito sin evidencia.
- **Git**: commits frecuentes por task. NUNCA `git push` sin orden explícita de Bernie.

## 4. Restricciones de hardware (supuestos fijos)
- VRAM usable: ~5.2 GB de 6 GB. Presupuesto: buffers ~0.9 GB, activaciones/driver ~1.3 GB, resto KV-cache.
- Bus: PCIe 4.0 x8 ≈ 12 GB/s efectivos.
- Los pesos cargan del NVMe a RAM pinned UNA sola vez; el streaming es SIEMPRE RAM→VRAM.
- Toda estimación de throughput se valida con benchmark real antes de usarse en specs.

## 5. Calidad de specs
- Requisitos en formato SHALL con escenarios WHEN/THEN concretos.
- Toda spec numérica cita su fuente (cálculo propio verificado o benchmark medido, nunca estimación sin etiquetar).
- Non-goals explícitos en cada change para evitar scope creep.
