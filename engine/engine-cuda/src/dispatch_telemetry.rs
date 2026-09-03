//! Benchmark-only accounting for successful GEMV/GEMM kernel dispatches.
//!
//! Module registration (`cuModuleGetFunction`) only proves that a symbol exists;
//! it is not evidence that the kernel was dispatched. Records are therefore
//! appended by the launch wrappers after `cuLaunchKernel` returns success.

use serde::Serialize;
use std::collections::BTreeMap;
use std::sync::Mutex;

/// Metadata for one observed successful kernel-launch call site.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DispatchRecord {
    pub operation: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_role: Option<String>,
    pub format: String,
    pub ne0: usize,
    pub ne1: usize,
    pub batch_size: usize,
    pub grid_x: u32,
    pub grid_y: u32,
    pub grid_z: u32,
    pub block_x: u32,
    pub block_y: u32,
    pub block_z: u32,
    pub selected_variant: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
    pub launches: usize,
}

impl DispatchRecord {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        operation: impl Into<String>,
        tensor_role: Option<String>,
        format: impl Into<String>,
        ne0: usize,
        ne1: usize,
        batch_size: usize,
        grid: [u32; 3],
        block: [u32; 3],
        selected_variant: impl Into<String>,
        fallback_reason: Option<String>,
    ) -> Self {
        Self {
            operation: operation.into(),
            tensor_role,
            format: format.into(),
            ne0,
            ne1,
            batch_size,
            grid_x: grid[0],
            grid_y: grid[1],
            grid_z: grid[2],
            block_x: block[0],
            block_y: block[1],
            block_z: block[2],
            selected_variant: selected_variant.into(),
            fallback_reason,
            launches: 1,
        }
    }
}

/// Serializable snapshot emitted by benchmark telemetry.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct DispatchTelemetrySnapshot {
    pub observed_launches: usize,
    pub records: Vec<DispatchRecord>,
}

#[derive(Debug, Default)]
struct DispatchState {
    current_tensor_role: Option<String>,
    records: BTreeMap<String, DispatchRecord>,
    observed_launches: usize,
}

/// Optional recorder. `None` is used by default so disabled production paths do
/// not allocate or lock when launching kernels.
pub struct DispatchTelemetry {
    state: Mutex<DispatchState>,
}

impl DispatchTelemetry {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(DispatchState::default()),
        }
    }

    /// Sets the logical role attached to subsequent launches on this recorder.
    pub fn set_tensor_role(&self, role: Option<&str>) {
        if let Ok(mut state) = self.state.lock() {
            state.current_tensor_role = role.map(str::to_owned);
        }
    }

    pub fn has_tensor_role(&self, role: &str) -> bool {
        self.state
            .lock()
            .map(|state| state.current_tensor_role.as_deref() == Some(role))
            .unwrap_or(false)
    }

    /// Aggregates an observed successful launch by its full dispatch shape.
    pub fn record(&self, mut record: DispatchRecord) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        // A role labels the next successful launch only. Logical operations that
        // dispatch multiple kernels must set the role again before each launch;
        // consuming it here prevents attribution leaking into a later operation.
        record.tensor_role = match record.tensor_role {
            Some(role) => Some(role),
            None => state.current_tensor_role.take(),
        };
        state.observed_launches += 1;
        let key = format!(
            "{}|{:?}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
            record.operation,
            record.tensor_role,
            record.format,
            record.ne0,
            record.ne1,
            record.batch_size,
            record.grid_x,
            record.grid_y,
            record.grid_z,
            record.block_x,
            record.block_y,
            record.selected_variant,
        );
        if let Some(existing) = state.records.get_mut(&key) {
            existing.launches += 1;
        } else {
            state.records.insert(key, record);
        }
    }

    pub fn snapshot(&self) -> DispatchTelemetrySnapshot {
        let Ok(state) = self.state.lock() else {
            return DispatchTelemetrySnapshot::default();
        };
        DispatchTelemetrySnapshot {
            observed_launches: state.observed_launches,
            records: state.records.values().cloned().collect(),
        }
    }
}

impl Default for DispatchTelemetry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{DispatchRecord, DispatchTelemetry};

    fn record(variant: &str) -> DispatchRecord {
        DispatchRecord::new(
            "gemm",
            None,
            "Q6K",
            4096,
            11008,
            1,
            [4, 1, 1],
            [256, 1, 1],
            variant,
            None,
        )
    }

    #[test]
    fn aggregates_identical_launches_and_keeps_variants_separate() {
        let telemetry = DispatchTelemetry::new();
        telemetry.record(record("gemm_q6k_multi_row_kernel"));
        telemetry.record(record("gemm_q6k_multi_row_kernel"));
        telemetry.record(record("gemm_q6k_2col_kernel"));

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.observed_launches, 3);
        assert_eq!(snapshot.records.len(), 2);
        assert_eq!(
            snapshot
                .records
                .iter()
                .find(|r| r.selected_variant == "gemm_q6k_multi_row_kernel")
                .unwrap()
                .launches,
            2
        );
    }

    #[test]
    fn records_selected_variant_role_and_serializes_without_cuda() {
        let telemetry = DispatchTelemetry::new();
        telemetry.set_tensor_role(Some("ffn_down"));
        telemetry.record(record("gemm_q6k_splitk2_kernel"));

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.records[0].tensor_role.as_deref(), Some("ffn_down"));
        assert_eq!(
            snapshot.records[0].selected_variant,
            "gemm_q6k_splitk2_kernel"
        );
        let json = serde_json::to_value(snapshot).expect("dispatch snapshot serializes");
        assert_eq!(json["observed_launches"], 1);
        assert_eq!(
            json["records"][0]["selected_variant"],
            "gemm_q6k_splitk2_kernel"
        );
        assert_eq!(json["records"][0]["tensor_role"], "ffn_down");
    }

    #[test]
    fn consumes_tensor_role_after_recording_once() {
        let telemetry = DispatchTelemetry::new();
        telemetry.set_tensor_role(Some("ffn_down"));
        telemetry.record(record("first"));
        telemetry.record(record("second"));

        let snapshot = telemetry.snapshot();
        assert_eq!(
            snapshot
                .records
                .iter()
                .find(|record| record.selected_variant == "first")
                .unwrap()
                .tensor_role
                .as_deref(),
            Some("ffn_down")
        );
        assert_eq!(
            snapshot
                .records
                .iter()
                .find(|record| record.selected_variant == "second")
                .unwrap()
                .tensor_role,
            None
        );
    }

    #[test]
    fn models_multi_launch_logical_operation_role_lifecycle() {
        let telemetry = DispatchTelemetry::new();

        // This is the FFN sequence: quantize activations, then dispatch GEMM.
        // The role belongs to the quantization launch, not to the following
        // launch unless the caller explicitly assigns it again.
        telemetry.set_tensor_role(Some("activation_quantization"));
        telemetry.record(DispatchRecord::new(
            "quantize",
            None,
            "Q8_1",
            3072,
            96,
            1,
            [1, 1, 1],
            [256, 1, 1],
            "quantize_row_q8_1_kernel",
            None,
        ));
        telemetry.record(DispatchRecord::new(
            "gemm",
            None,
            "Q4K",
            3072,
            8192,
            1,
            [1024, 1, 1],
            [256, 1, 1],
            "gemm_q4k_mma_kernel",
            None,
        ));

        let snapshot = telemetry.snapshot();
        assert_eq!(snapshot.observed_launches, 2);
        assert_eq!(
            snapshot
                .records
                .iter()
                .find(|record| record.operation == "quantize")
                .and_then(|record| record.tensor_role.as_deref()),
            Some("activation_quantization")
        );
        assert_eq!(
            snapshot
                .records
                .iter()
                .find(|record| record.operation == "gemm")
                .and_then(|record| record.tensor_role.as_deref()),
            None
        );
    }
}
