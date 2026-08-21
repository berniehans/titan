# Specification: Layer-Streaming Engine Core

Capacidad central del sistema: ejecutar LLMs cuyos pesos no caben en VRAM, transmitiéndolos por capa (o experto) desde RAM pinned hacia buffers dobles en GPU.

## ADDED Requirements

### Requirement: Carga única de pesos a RAM pinned
El sistema SHALL cargar todos los tensores del modelo GGUF desde el NVMe a memoria host pinned (cudaMallocHost) UNA sola vez al inicio, y SHALL NUNCA leer del disco durante la generación.

#### Scenario: Carga de fixture 0.6B
- **WHEN** se inicia la carga de un GGUF Q4_K_M de ~400 MB
- **THEN** los tensores quedan en regiones pinned contiguas por capa en <5 s
- **AND** la suma de bytes cargados es igual al tamaño del archivo
- **AND** ningún read() ocurre durante el loop de generación (verificable con trace)

### Requirement: Pipeline doble búfer con solapamiento
El sistema SHALL mantener dos streams CUDA (transferencia y cómputo) sincronizados por eventos, de modo que la copia H2D de la capa N+1 se solape con la ejecución del kernel de la capa N.

#### Scenario: Solapamiento medido con Nsight
- **WHEN** se ejecuta el benchmark de pipeline con modelo dummy de 8 capas
- **THEN** el trace muestra ≥80% de la ventana de cómputo cubierta por transferencia concurrente
- **AND** el tiempo total por capa < (t_copy + t_kernel) secuencial

#### Scenario: Orden de dependencias
- **WHEN** compute evalúa la capa N
- **THEN** espera el evento copy_done[N] antes de lanzar kernels sobre ese buffer
- **AND** no hay busy-waiting en la CPU

### Requirement: Descuantización al vuelo en GPU
Los pesos Q4_K_M SHALL descuantizarse dentro de los kernels GPU (shared memory/registers), sin materializar versiones FP16 completas en VRAM.

#### Scenario: Paridad numérica
- **WHEN** se compara dequant GPU vs referencia CPU bloque a bloque
- **THEN** error máximo < 0.01 por elemento

### Requirement: Throughput predecible por escala
El sistema SHALL documentar el throughput real medido por escala de modelo, sin promesas teóricas sin benchmark.

#### Scenario: Denso 14B Q4 en PCIe x8
- **WHEN** se genera texto con un denso 14B Q4 (~8.5 GB) residente en RAM
- **THEN** throughput esperado ≈ 1.4 tok/s (medido, no estimado)
- **AND** el primer token generado es válido end-to-end (topología completa de capas)
